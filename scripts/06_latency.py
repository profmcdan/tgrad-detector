"""Inference latency on Elliptic, every model timed in the same way (Table of the paper).

GNNs: one forward pass over the graph; Random Forest: prediction on the labelled
transactions; T-GRAD: encoder pass over the event stream, energy head, guided reverse
diffusion and forest prediction, timed stage by stage. Median of three runs, CPU only.
The hardware record uses sysctl, so it is filled in on macOS only.

    python scripts/06_latency.py

Writes results/latency_elliptic.json. Needs the Elliptic CSV files (see README).
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_curve,
                             precision_recall_curve, confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from tgrad.data.dataset import EllipticStandardDataset
from tgrad.models.temporal_encoder_v2 import TemporalEncoderV2
from tgrad.models.diffusion import VariancePreservingSDE, ScoreNetworkGNN
from tgrad.models.cycle_guidance import EnergyCycleGuidance
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "results")
os.makedirs(ART, exist_ok=True)
DEVICE = "cpu"
print(f"Device: {DEVICE}")

# ----------------------------------------------------------------------------
# Metrics helper: threshold tuned on TRAIN, applied to TEST (no leakage).
# ----------------------------------------------------------------------------
def best_threshold(y, s):
    p, r, th = precision_recall_curve(y, s)
    f1 = 2 * p * r / (p + r + 1e-12)
    k = int(np.argmax(f1[:-1])) if len(th) else 0
    return th[k] if len(th) else 0.5

def evaluate(y, s, thr):
    yp = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    k = max(1, int(0.05 * len(s)))
    topk = np.argsort(s)[::-1][:k]
    s01 = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return {
        "ROC-AUC": roc_auc_score(y, s),
        "PR-AUC": average_precision_score(y, s),
        "Precision": precision_score(y, yp, zero_division=0),
        "Recall": recall_score(y, yp, zero_division=0),
        "F1": f1_score(y, yp, zero_division=0),
        "Recall@5%": float(np.sum(y[topk]) / (np.sum(y) + 1e-12)),
        "Precision@5%": float(np.mean(y[topk])),
        "FPR": float(fp / (fp + tn + 1e-12)),
        "Brier": brier_score_loss(y, s01),
    }

# ============================================================================
# 1. DATA  (Objective a)
# ============================================================================
print("\n=== Loading Elliptic (standard temporal protocol) ===")
ds = EllipticStandardDataset(os.path.join(ROOT, "data/elliptic_bitcoin_dataset"))
ds.load_data()

# Standardise the 165 features on TRAIN nodes only, then rebuild the edge stream.
scaler = StandardScaler().fit(ds.node_feat[ds.train_mask.numpy()])
node_feat = scaler.transform(ds.node_feat).astype(np.float32)
node_feat_t = torch.tensor(node_feat)
ds.msg = node_feat_t[ds.src]                     # edge message = scaled src features
node_label = ds.node_label_t.numpy()
train_mask = ds.train_mask.numpy()
test_mask = ds.test_mask.numpy()

y_train = node_label[train_mask]
y_test = node_label[test_mask]
pos_weight = torch.tensor([(y_train == 0).sum() / max(1, (y_train == 1).sum())], device=DEVICE)
print(f"pos_weight (licit/illicit) = {pos_weight.item():.2f}")


# ============================================================================
# Paper 3 revision: per-node INFERENCE latency, every model timed the same way
# (the published comparison timed GNN training + inference against a single
# energy-head forward pass for T-GRAD). Weights do not change the cost of a
# forward pass; forests are fitted first because prediction cost depends on trees.
# ============================================================================
import time, statistics, json, platform, subprocess
from torch_geometric.nn import SAGEConv, GATConv, GINConv, TransformerConv
torch.set_num_threads(int(os.environ.get("TGRAD_THREADS", "8")))
gnn_ei = to_undirected(ds.edge_index); gnn_x = node_feat_t
class GNN(nn.Module):
    """Unified 2-layer GNN backbone for the SOTA baselines."""
    def __init__(self, kind, in_dim, hid=64):
        super().__init__()
        self.act = nn.ReLU(); self.drop = nn.Dropout(0.3)
        if kind == "GCN":
            self.c1, self.c2 = GCNConv(in_dim, hid), GCNConv(hid, 2)
        elif kind == "GraphSAGE":
            self.c1, self.c2 = SAGEConv(in_dim, hid), SAGEConv(hid, 2)
        elif kind == "GAT":
            self.c1 = GATConv(in_dim, hid, heads=4, concat=True, dropout=0.3)
            self.c2 = GATConv(hid * 4, 2, heads=1, concat=False, dropout=0.3)
        elif kind == "GIN":
            self.c1 = GINConv(nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid)))
            self.c2 = GINConv(nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 2)))
        elif kind == "Graph Transformer":
            self.c1 = TransformerConv(in_dim, hid, heads=4, concat=True, dropout=0.3)
            self.c2 = TransformerConv(hid * 4, 2, heads=1, concat=False, dropout=0.3)
    def forward(self, x, ei):
        return self.c2(self.drop(self.act(self.c1(x, ei))), ei)


def timed(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)
N = ds.num_nodes; lab_idx = np.where(node_label >= 0)[0]; out = {"num_nodes": int(N), "num_labelled": int(len(lab_idx)), "num_edges": int(ds.num_edges)}
lat = {}
for kind in ["GCN", "GraphSAGE", "GAT", "GIN", "Graph Transformer"]:
    torch.manual_seed(SEED); net = GNN(kind, node_feat.shape[1]).eval()
    def f():
        with torch.no_grad(): net(gnn_x, gnn_ei)
    lat[kind] = {"total_sec": timed(f)}; print(kind, lat[kind], flush=True)
from sklearn.ensemble import RandomForestClassifier
rf_a = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED).fit(node_feat[train_mask], y_train)
lat["RF, attributes"] = {"total_sec": timed(lambda: rf_a.predict_proba(node_feat[lab_idx]))}; print("RF attr", lat["RF, attributes"], flush=True)

# ---- T-GRAD inference, stage by stage (Algorithm 2) ----
NODE_DIM, TIME_DIM, EDGE_DIM = 128, 32, ds.edge_dim
encoder = TemporalEncoderV2(ds.num_nodes, NODE_DIM, EDGE_DIM, TIME_DIM).eval()
sde = VariancePreservingSDE(ScoreNetworkGNN(NODE_DIM)).eval(); guidance = EnergyCycleGuidance(NODE_DIM).eval()
BATCH = 1024
def encode():
    encoder.memory.reset_memory(); es = torch.zeros(N, NODE_DIM); ec = torch.zeros(N)
    with torch.no_grad():
        for st in range(0, ds.num_edges, BATCH):
            b = ds.get_batch(st, BATCH); sz, dz = encoder(b["src"], b["dst"], b["t"], b["msg"])
            es.index_add_(0, b["src"], sz); ec.index_add_(0, b["src"], torch.ones(len(b["t"])))
            es.index_add_(0, b["dst"], dz); ec.index_add_(0, b["dst"], torch.ones(len(b["t"])))
    return es / ec.clamp(min=1).unsqueeze(1)
t0 = time.perf_counter(); node_emb = encode(); t_enc = time.perf_counter() - t0
def recon_error(emb, lam, steps=40, chunk=8192):
    out = np.zeros(emb.shape[0], dtype=np.float32)
    t_steps = torch.linspace(1.0, 1e-3, steps, device=DEVICE)
    dt = 1.0 / steps
    for s in range(0, emb.shape[0], chunk):
        z0 = emb[s:s+chunk]
        e = torch.arange(z0.shape[0], device=DEVICE).unsqueeze(0).repeat(2, 1)
        tmax = torch.ones(z0.shape[0], device=DEVICE)
        mean, std = sde.marginal_prob(z0, tmax)
        x = (mean + std[:, None] * torch.randn_like(z0)).requires_grad_(True)
        for tv in t_steps:
            tb = torch.ones(x.shape[0], device=DEVICE) * tv
            beta = sde.beta_t(tv)
            with torch.no_grad():
                score = sde.score_net(x, e, tb)
            if lam > 0:
                g = guidance.compute_cycle_gradient(x).clamp(-5, 5)
                score = score + lam * g
            drift = -0.5 * beta * x - beta * score
            x = (x - drift * dt + torch.sqrt(torch.tensor(beta)) *
                 torch.sqrt(torch.tensor(dt)) * (torch.randn_like(x) if tv > t_steps[-1] else 0)).detach().requires_grad_(True)
        out[s:s+chunk] = torch.mean((z0 - x) ** 2, dim=1).detach().cpu().numpy()
    return out


t0 = time.perf_counter()
with torch.no_grad(): energy_logit = guidance.fraud_logit(node_emb)
t_energy = time.perf_counter() - t0
emb_lab = node_emb[lab_idx]
t0 = time.perf_counter(); rg = recon_error(emb_lab, lam=10.0); t_recon = time.perf_counter() - t0
X_full = np.hstack([node_feat, node_emb.numpy(), np.zeros((N, 1), np.float32), energy_logit.numpy().reshape(-1, 1)])
X_full[lab_idx, -2] = rg
rf_f = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED).fit(X_full[train_mask], y_train)
t_rf = timed(lambda: rf_f.predict_proba(X_full[lab_idx]))
lat["T-GRAD"] = {"total_sec": t_enc + t_energy + t_recon + t_rf, "encoder_sec": t_enc, "energy_head_sec": t_energy,
                 "guided_reconstruction_sec": t_recon, "forest_predict_sec": t_rf,
                 "head_only_sec": t_energy}
print("T-GRAD", lat["T-GRAD"], flush=True)
g = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
out["hardware"] = {"cpu": g("machdep.cpu.brand_string"), "cores": g("hw.ncpu"), "ram_gb": round(int(g("hw.memsize")) / 2**30, 1),
                   "os": "macOS " + platform.mac_ver()[0], "threads": torch.get_num_threads()}
for k, v in lat.items():
    v["ms_per_labelled_node"] = 1000 * v["total_sec"] / len(lab_idx)
out["latency"] = lat
json.dump(out, open(os.path.join(ART, "latency_elliptic.json"), "w"), indent=2)
print(json.dumps({k: round(v["ms_per_labelled_node"], 5) for k, v in lat.items()}, indent=1))

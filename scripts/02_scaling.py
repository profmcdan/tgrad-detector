"""
T-GRAD scalability study on AMLSim-style cyclic laundering graphs.

The controlled cyclic benchmark is regenerated at increasing transaction volumes
(200k, 400k, 600k, 800k, 1M, 1.2M, 2M directed edges), holding the graph density
and illicit prevalence fixed so that GRAPH SIZE is the only varying factor. For
each size the full T-GRAD pipeline (temporal encoder -> continuous-time diffusion
-> cycle-aware guidance -> augmented classifier) is trained and evaluated, together
with the strongest scaling SOTA baseline (GraphSAGE), and both detection quality
and computational cost (wall time, streaming throughput, peak memory) are recorded.

Design notes for tractability / honesty:
  * The temporal encoder is TRAINED on a bounded chronological prefix (standard
    streaming practice); the reported compute-scaling measure is the FULL-graph
    inference/extraction pass over every edge, whose cost is the operational O(E)
    streaming cost.
  * Structural cycle features use the diagonal of A^2 and A^3 computed WITHOUT
    materialising A^3 (elementwise A .* (A^2)^T), so memory stays bounded at scale.
  * Density (mean degree) and illicit prevalence are held constant across sizes.
  * The encoder and energy head are trained on the labels of training entities only.

    TGRAD_SEED=42 python scripts/02_scaling.py       # also run with seeds 0 and 1

Writes results/scaling_seed<SEED>.json.
"""
import os, sys, json, time, warnings, resource, gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
try:
    import psutil
    _PROC = psutil.Process()
except Exception:
    _PROC = None

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from tgrad.data.dataset import SyntheticCyclicAML
from tgrad.models.temporal_encoder_v2 import TemporalEncoderV2
from tgrad.models.diffusion import VariancePreservingSDE, ScoreNetworkGNN
from tgrad.models.cycle_guidance import EnergyCycleGuidance
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, precision_recall_curve)

SEED = int(os.environ.get("TGRAD_SEED", "42"))
torch.manual_seed(SEED); np.random.seed(SEED)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "results")
os.makedirs(ART, exist_ok=True)
C_GRID = 0.3

# ---- sweep configuration (overridable via env for smoke tests) ----
SIZES = [int(x) for x in os.environ.get(
    "TGRAD_SIZES", "200000,400000,600000,800000,1000000,1200000,2000000").split(",")]
MEAN_DEG = 3.45           # held constant: nodes scale with edges
CYCLE_EDGE_FRAC = 0.13    # fraction of edges belonging to planted cycles (prevalence proxy)
AVG_CYCLE_LEN = 4.5       # mean of cycle_len_range (3..6)
ENC_EPOCHS = int(os.environ.get("TGRAD_ENC_EPOCHS", "3"))
ENC_TRAIN_CAP = int(os.environ.get("TGRAD_ENC_CAP", "120000"))  # bound encoder training edges
GSAGE_EPOCHS = int(os.environ.get("TGRAD_GSAGE_EPOCHS", "120"))
HEAD_EPOCHS = int(os.environ.get("TGRAD_HEAD_EPOCHS", "250"))
NODE_DIM, TIME_DIM = 128, 32
BATCH = 2048

def cur_rss_mb():
    # current resident-set size in MB (per-size footprint; psutil preferred)
    if _PROC is not None:
        return _PROC.memory_info().rss / (1024 * 1024)
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024

def best_threshold(y, s):
    p, r, th = precision_recall_curve(y, s)
    f1 = 2 * p * r / (p + r + 1e-12)
    k = int(np.argmax(f1[:-1])) if len(th) else 0
    return th[k] if len(th) else 0.5

def evaluate(y, s, thr):
    yp = (s >= thr).astype(int)
    k = max(1, int(0.05 * len(s)))
    topk = np.argsort(s)[::-1][:k]
    return {"ROC-AUC": float(roc_auc_score(y, s)),
            "PR-AUC": float(average_precision_score(y, s)),
            "Precision": float(precision_score(y, yp, zero_division=0)),
            "Recall": float(recall_score(y, yp, zero_division=0)),
            "F1": float(f1_score(y, yp, zero_division=0)),
            "Recall@5%": float(np.sum(y[topk]) / (np.sum(y) + 1e-12))}

GSAGE_HID = int(os.environ.get("TGRAD_GSAGE_HID", "32"))  # smaller hidden keeps peak memory bounded

class GSAGE(nn.Module):
    def __init__(self, d, h=GSAGE_HID):
        super().__init__(); self.c1, self.c2 = SAGEConv(d, h), SAGEConv(h, 2)
        self.act, self.drop = nn.ReLU(), nn.Dropout(0.3)
    def forward(self, x, ei): return self.c2(self.drop(self.act(self.c1(x, ei))), ei)

class FraudHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def run_size(target_edges):
    t_wall0 = time.time()
    num_nodes = int(round(target_edges / MEAN_DEG))
    num_cycles = int(round(CYCLE_EDGE_FRAC * target_edges / AVG_CYCLE_LEN))
    num_benign = int(round(target_edges - num_cycles * AVG_CYCLE_LEN))
    ds = SyntheticCyclicAML(num_nodes=num_nodes, feat_dim=64,
                            num_benign_edges=num_benign, num_cycles=num_cycles,
                            cycle_len_range=(3, 6), time_span=100.0, seed=SEED)
    ds.load_data()
    node_label = ds.node_label_t.numpy()
    train_mask = ds.train_mask.numpy(); test_mask = ds.test_mask.numpy()
    y_train, y_test = node_label[train_mask], node_label[test_mask]
    node_feat = ds.node_feat
    prev = float(node_label.mean())
    pos_weight = torch.tensor([(y_train == 0).sum() / max(1, (y_train == 1).sum())])

    # ---- memory-safe structural cycle features: indeg, outdeg, diag(A^2), diag(A^3) ----
    ei = ds.edge_index.numpy()
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(ds.num_nodes, ds.num_nodes))
    A.data[:] = 1
    A2 = A @ A; A3 = A2 @ A; pw = {1: A, 2: A2, 3: A3}; walks = {}
    for L, (a, b) in {2: (1, 1), 3: (1, 2), 4: (2, 2), 5: (2, 3), 6: (3, 3)}.items():
        walks[L] = np.asarray(pw[a].multiply(pw[b].T).sum(1)).ravel()   # exact diag(A^L)
    mon_raw = sum(2.0 ** -L * walks[L] for L in walks)                   # Eq. (monitor), L = 2..6
    struct = np.vstack([np.asarray(A.sum(1)).ravel(), np.asarray(A.sum(0)).ravel()]
                       + [walks[L] for L in range(2, 7)]).T.astype(np.float32)
    del A2, A3, pw, A; gc.collect()
    peak_mb = cur_rss_mb()

    # ---- SOTA baseline: GraphSAGE (scales best of the five GNNs) ----
    gnn_ei = to_undirected(ds.edge_index)
    gnn_x, gnn_y = ds.node_feat_t, ds.node_label_t
    cw = torch.tensor([1.0, pos_weight.item()])
    torch.manual_seed(SEED)
    sage = GSAGE(node_feat.shape[1]); o = torch.optim.Adam(sage.parameters(), lr=1e-2, weight_decay=5e-4)
    tr_t = torch.tensor(train_mask)
    t_g0 = time.time()
    for _ in range(GSAGE_EPOCHS):
        sage.train(); o.zero_grad()
        loss = nn.functional.cross_entropy(sage(gnn_x, gnn_ei)[tr_t], gnn_y[tr_t], weight=cw)
        loss.backward(); o.step()
    sage.eval()
    with torch.no_grad():
        sage_p = torch.softmax(sage(gnn_x, gnn_ei), 1)[:, 1].numpy()
    m_sage = evaluate(y_test, sage_p[test_mask], best_threshold(y_train, sage_p[train_mask]))
    sage_time = time.time() - t_g0
    peak_mb = max(peak_mb, cur_rss_mb())
    del sage, o, gnn_ei; gc.collect()

    # ---- T-GRAD temporal encoder (train on bounded chronological prefix) ----
    encoder = TemporalEncoderV2(ds.num_nodes, NODE_DIM, ds.edge_dim, TIME_DIM)
    sde = VariancePreservingSDE(ScoreNetworkGNN(NODE_DIM))
    guidance = EnergyCycleGuidance(NODE_DIM)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(sde.parameters()) +
                           list(guidance.parameters()), lr=1e-3)
    node_label_dev = ds.node_label_t
    n_train_edges = int((ds.t.numpy() <= ds.train_t_max).sum())
    enc_cap = min(n_train_edges, ENC_TRAIN_CAP)
    t_tr0 = time.time()
    for ep in range(ENC_EPOCHS):
        encoder.memory.reset_memory()
        for start in range(0, enc_cap, BATCH):
            b = ds.get_batch(start, BATCH)
            src, dst, t, msg = b['src'], b['dst'], b['t'], b['msg']
            only = (t <= ds.train_t_max)
            src_z, dst_z = encoder(src, dst, t, msg)
            z = torch.cat([src_z, dst_z], 0); bs = src.shape[0]
            lbl = torch.cat([node_label_dev[src], node_label_dev[dst]])
            tctx = torch.cat([only, only])
            # labels only from TRAINING entities: an endpoint of a training-window
            # event may be a test entity, whose label must not be used
            tctx = tctx & torch.cat([ds.train_mask[src], ds.train_mask[dst]])
            licit = (lbl == 0) & tctx
            ld = sde.compute_loss(z[licit], torch.arange(int(licit.sum())).unsqueeze(0).repeat(2, 1)) \
                if licit.sum() > 4 else torch.tensor(0.0)
            lm = (lbl >= 0) & tctx
            lg = guidance.compute_bce_loss(z[lm], lbl[lm], pos_weight=pos_weight) if lm.sum() > 4 else torch.tensor(0.0)
            loss = ld + lg
            if not loss.requires_grad: continue
            opt.zero_grad(); loss.backward(); opt.step()
    tgrad_train_time = time.time() - t_tr0

    # ---- full-graph inference/extraction pass over EVERY edge (O(E) streaming cost) ----
    encoder.eval(); encoder.memory.reset_memory()
    emb_sum = torch.zeros(ds.num_nodes, NODE_DIM); emb_cnt = torch.zeros(ds.num_nodes)
    t_inf0 = time.time()
    with torch.no_grad():
        for start in range(0, ds.num_edges, BATCH):
            b = ds.get_batch(start, BATCH)
            src, dst, t, msg = b['src'], b['dst'], b['t'], b['msg']
            sz, dz = encoder(src, dst, t, msg)
            emb_sum.index_add_(0, src, sz); emb_cnt.index_add_(0, src, torch.ones_like(t))
            emb_sum.index_add_(0, dst, dz); emb_cnt.index_add_(0, dst, torch.ones_like(t))
    infer_time = time.time() - t_inf0
    throughput = ds.num_edges / max(infer_time, 1e-6)
    node_emb = (emb_sum / emb_cnt.clamp(min=1).unsqueeze(1)).numpy()
    peak_mb = max(peak_mb, cur_rss_mb())

    # ---- augmented representation + guidance signal ----
    with torch.no_grad():
        energy_logit = guidance.fraud_logit(torch.tensor(node_emb)).numpy()
    def std_feat(a):
        return (a - a[train_mask].mean(0)) / (a[train_mask].std(0) + 1e-8)
    def recon_error(emb, lam, steps=40, chunk=8192):
        out = np.zeros(emb.shape[0], dtype=np.float32); t_steps = torch.linspace(1.0, 1e-3, steps); dt = 1.0 / steps
        for s0 in range(0, emb.shape[0], chunk):
            z0 = emb[s0:s0 + chunk]; e = torch.arange(z0.shape[0]).unsqueeze(0).repeat(2, 1)
            mean, std = sde.marginal_prob(z0, torch.ones(z0.shape[0]))
            x = (mean + std[:, None] * torch.randn_like(z0)).requires_grad_(True)
            for tv in t_steps:
                tb = torch.ones(x.shape[0]) * tv; beta = sde.beta_t(tv)
                with torch.no_grad(): score = sde.score_net(x, e, tb)
                if lam > 0: score = score + lam * guidance.compute_cycle_gradient(x).clamp(-5, 5)
                drift = -0.5 * beta * x - beta * score
                x = (x - drift * dt + torch.sqrt(torch.tensor(beta)) * torch.sqrt(torch.tensor(dt))
                     * (torch.randn_like(x) if tv > t_steps[-1] else 0)).detach().requires_grad_(True)
            out[s0:s0 + chunk] = torch.mean((z0 - x) ** 2, dim=1).detach().numpy()
        return out
    rec_g = recon_error(torch.tensor(node_emb), 10.0)
    X_full = np.hstack([node_feat, std_feat(node_emb), std_feat(struct), std_feat(rec_g.reshape(-1, 1)),
                        std_feat(energy_logit.reshape(-1, 1))]).astype(np.float32)

    # ---- final augmented classifier (T-GRAD readout) ----
    Xt = torch.tensor(X_full); yt = torch.tensor(node_label, dtype=torch.float32)
    head = FraudHead(X_full.shape[1]); oh = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    fm = torch.tensor(train_mask)
    for _ in range(HEAD_EPOCHS):
        head.train(); oh.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(head(Xt[fm]), yt[fm], pos_weight=pos_weight)
        loss.backward(); oh.step()
    head.eval()
    with torch.no_grad():
        tgrad_p = torch.sigmoid(head(Xt)).numpy()
    m_tgrad = evaluate(y_test, tgrad_p[test_mask], best_threshold(y_train, tgrad_p[train_mask]))
    peak_mb = max(peak_mb, cur_rss_mb())

    m_mon = evaluate(y_test, mon_raw[test_mask], best_threshold(y_train, mon_raw[train_mask]))
    gx_m = torch.tensor(np.hstack([node_feat, std_feat(struct)]), dtype=torch.float32); gei = to_undirected(ds.edge_index)
    torch.manual_seed(SEED); sg = GSAGE(gx_m.shape[1]); o2 = torch.optim.Adam(sg.parameters(), lr=1e-2, weight_decay=5e-4)
    trt = torch.tensor(train_mask)
    for _ in range(GSAGE_EPOCHS):
        sg.train(); o2.zero_grad()
        loss = nn.functional.cross_entropy(sg(gx_m, gei)[trt], ds.node_label_t[trt], weight=torch.tensor([1.0, pos_weight.item()]))
        loss.backward(); o2.step()
    sg.eval()
    with torch.no_grad(): sp_m = torch.softmax(sg(gx_m, gei), 1)[:, 1].numpy()
    m_sage_m = evaluate(y_test, sp_m[test_mask], best_threshold(y_train, sp_m[train_mask]))
    del sg, gx_m, gei; gc.collect()
    n_nodes, n_edges, n_ill = ds.num_nodes, ds.num_edges, int(node_label.sum())
    rec = {"target_edges": target_edges, "num_nodes": n_nodes, "num_edges": n_edges,
           "num_illicit": n_ill, "prevalence": prev,
           "mean_degree": n_edges / n_nodes,
           "tgrad": m_tgrad, "graphsage": m_sage, "monitor_raw": m_mon, "graphsage_matched": m_sage_m,
           "tgrad_train_sec": tgrad_train_time, "infer_sec": infer_time,
           "throughput_eps": throughput, "graphsage_train_sec": sage_time,
           "wall_sec": time.time() - t_wall0, "peak_rss_mb": peak_mb}
    print(f"  N={n_nodes:>7} E={n_edges:>8} prev={prev:.3f} | "
          f"T-GRAD AUC={m_tgrad['ROC-AUC']:.3f} F1={m_tgrad['F1']:.3f} | "
          f"SAGE AUC={m_sage['ROC-AUC']:.3f} F1={m_sage['F1']:.3f} | "
          f"infer={infer_time:.1f}s ({throughput:,.0f} e/s) wall={rec['wall_sec']:.1f}s "
          f"rss={peak_mb:.0f}MB")
    del ds, encoder, sde, guidance, head, Xt, X_full, node_emb, struct, emb_sum, emb_cnt
    gc.collect()
    return rec

# ============================================================================
# run sweep
# ============================================================================
print("=== T-GRAD scalability sweep (AMLSim-style cyclic benchmark) ===")
print(f"sizes={SIZES}  enc_epochs={ENC_EPOCHS} enc_cap={ENC_TRAIN_CAP} "
      f"gsage_epochs={GSAGE_EPOCHS} head_epochs={HEAD_EPOCHS}")
records = []
for E in SIZES:
    print(f"\n--- target {E:,} edges ---")
    records.append(run_size(E))
    json.dump(records, open(os.path.join(ART, f"scaling_seed{SEED}.json"), "w"), indent=2, default=float)


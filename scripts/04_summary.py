"""Numbers, tables and the per-time-step figure of the paper, from the saved runs.

Reads results/benchmark_full.json, results/elliptic_seed_*/paper3_final.json,
results/scaling_seed*.json and results/timing.json (latency and closed-walk timing, measured
by scripts/11_timing.py of https://github.com/profmcdan/tgrad) and results/latency_elliptic.json
(scripts/06_latency.py). Writes
results/paper3_summary.json, results/tables.md and figures/robustness_paper3.png.

    python scripts/04_summary.py
"""
import glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(ROOT, "results")
B = json.load(open(f"{R}/benchmark_full.json"))
E = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/elliptic_seed_*/paper3_final.json"))]
S = {int(re.search(r"seed(\d+)", f).group(1)): json.load(open(f)) for f in sorted(glob.glob(f"{R}/scaling_seed*.json"))}
TM = json.load(open(f"{R}/timing.json"))["rows"]
assert len(E) == 3 and len(S) == 3, (len(E), len(S))
M = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Recall@5%", "FPR", "Brier_prob"]
pm = lambda m, s: "%.3f ± %.3f" % (m, s)


def agg(rows, m):
    v = np.array([r[m] for r in rows], float); return v.mean(), v.std(ddof=1)


md = ["# Tables of the paper (mean ± standard deviation over seeds)", ""]

# ---------------------------------------------------------------- complete detector
chosen = [e["chosen_readout"] for e in E]
sel = max(set(chosen), key=chosen.count)
other = "perceptron" if sel == "forest" else "forest"
EA = {k: {m: agg([e["test"][k] for e in E], m) for m in M} for k in ("forest", "perceptron")}
name = {"forest": "Random Forest", "perceptron": "perceptron"}
val = {k: np.mean([e["selection"][k]["val_PR-AUC"] for e in E]) for k in ("forest", "perceptron")}
gg_e = agg([{"g": e["guidance_gain_auc"]} for e in E], "g")
md += ["## Complete detector", "", "| Setting | " + " | ".join(M) + " |", "|" + "---|" * (len(M) + 1),
       "| Benchmark, perceptron | " + " | ".join(pm(*B[m]) for m in M) + " |"]
for k in (sel, other):
    md.append("| Elliptic, %s%s | " % (name[k], " (selected)" if k == sel else "") + " | ".join(pm(*EA[k][m]) for m in M) + " |")
md += ["", "Readout chosen on the validation window: %s (%s); validation PR-AUC %.3f (forest) and %.3f (perceptron)."
       % (name[sel], ", ".join(chosen), val["forest"], val["perceptron"]),
       "Guidance gain (ROC-AUC of the guided minus the unguided residual): benchmark %+.3f ± %.3f, Elliptic %+.3f ± %.3f."
       % (*B["guidance_gain_auc"], *gg_e), ""]

# ---------------------------------------------------------------- scaling
seeds = sorted(S); n = min(len(v) for v in S.values())
KEYS = [("monitor_raw", "Closed-walk score"), ("tgrad", "T-GRAD"), ("graphsage", "GraphSAGE"), ("graphsage_matched", "GraphSAGE, matched")]
md += ["## Scaling (ROC-AUC, three seeds)", "",
       "| Edges | Entities | " + " | ".join(l for _, l in KEYS) + " | Encoder (events/s) | Closed walks (s) |", "|" + "---|" * 8]
summ = []
for i in range(n):
    recs = [S[s][i] for s in seeds]
    Ed = int(np.mean([r["num_edges"] for r in recs])); Nd = int(np.mean([r["num_nodes"] for r in recs]))
    vals = {k: np.array([r[k]["ROC-AUC"] for r in recs]) for k, _ in KEYS}
    tm = min(TM, key=lambda r: abs(r["edges"] - Ed))
    md.append("| %s | %s | %s | %s | %.2f |" % (f"{Ed:,}", f"{Nd:,}", " | ".join(pm(v.mean(), v.std(ddof=1)) for v in vals.values()),
                                                f"{tm['encoder_events_per_sec']:,.0f}", tm["monitor_sec"]))
    summ.append({"edges": Ed, **{k: float(v.mean()) for k, v in vals.items()}})
md.append("")

# ---------------------------------------------------------------- per-step F1 on Elliptic
steps = list(range(35, 50))
F = {k: np.array([[e["per_step"][k][str(s)]["F1"] for s in steps] for e in E]) for k in (sel, "GraphSAGE")}
nill = np.array([E[0]["per_step"][sel][str(s)]["n_illicit"] for s in steps])
fig, ax = plt.subplots(figsize=(6.2, 3.4))
for k, c, lab in [(sel, "#1f78b4", "T-GRAD (%s readout)" % name[sel]), ("GraphSAGE", "#e31a1c", "GraphSAGE")]:
    mu = F[k].mean(0); ax.plot(steps, mu, "o-", color=c, lw=1.8, ms=4, label=lab)
    ax.fill_between(steps, F[k].min(0), F[k].max(0), color=c, alpha=0.18, lw=0)
ax.axvline(42.5, ls="--", color="grey", lw=1); ax.text(42.7, 0.95, "step 43", fontsize=8, color="grey", va="top")
ax.set_xlabel("time step"); ax.set_ylabel("illicit-class $F_1$"); ax.set_ylim(0, 1); ax.set_xticks(steps)
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "robustness_paper3.png"), dpi=300); plt.close()
pre = [i for i, s in enumerate(steps) if s <= 42 and nill[i] > 0]; post = [i for i, s in enumerate(steps) if s >= 43]
ahead = int((F[sel].mean(0)[pre] > F["GraphSAGE"].mean(0)[pre]).sum())
post_max = max(F[sel][:, post].max(), F["GraphSAGE"][:, post].max())
md += ["## Elliptic over time", "", "T-GRAD ahead of GraphSAGE at %d of %d test steps before step 43; "
       "from step 43 the illicit-class F1 of both models is at most %.2f in every seed." % (ahead, len(pre), post_max), ""]

# ---------------------------------------------------------------- latency
LAT = json.load(open(f"{R}/latency_elliptic.json"))
md += ["## Inference latency on Elliptic (median of three runs, CPU)", "",
       "| Model | Time (s) | ms per labelled transaction |", "|---|---|---|"]
for k, v in LAT["latency"].items():
    md.append("| %s | %.3f | %.4f |" % (k, v["total_sec"], v["ms_per_labelled_node"]))
t = LAT["latency"]["T-GRAD"]
md += ["", "T-GRAD stages: encoder %.3f s, guided reverse diffusion %.3f s, forest %.3f s, energy head %.3f s."
       % (t["encoder_sec"], t["guided_reconstruction_sec"], t["forest_predict_sec"], t["energy_head_sec"]),
       "T-GRAD is %.0f to %.0f times slower per transaction than the GNNs." % (
           t["total_sec"] / max(LAT["latency"][g]["total_sec"] for g in ("GCN", "GraphSAGE", "GAT", "GIN", "Graph Transformer")),
           t["total_sec"] / min(LAT["latency"][g]["total_sec"] for g in ("GCN", "GraphSAGE", "GAT", "GIN", "Graph Transformer"))), ""]

json.dump({"selected": sel, "chosen_per_seed": chosen, "validation_PR-AUC": val,
           "elliptic": {k: {m: list(v) for m, v in d.items()} for k, d in EA.items()},
           "guidance_gain_elliptic": list(gg_e), "scaling": summ, "per_step_ahead": [ahead, len(pre)],
           "post43_max_F1": float(post_max)}, open(f"{R}/paper3_summary.json", "w"), indent=2)
open(f"{R}/tables.md", "w").write("\n".join(md))
print("\n".join(md))

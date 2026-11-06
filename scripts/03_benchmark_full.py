"""Complete detector on the cyclic benchmark, aggregated over five seeds.

The benchmark runs are produced by scripts/03_benchmark.py of the companion repository
https://github.com/profmcdan/tgrad (closed walks of length 2 to 6, seeds 42, 0, 1, 2, 3);
their outputs are copied in results/benchmark/: revision.json per seed, whose row
"+ Diffusion guidance (Full)" is the complete detector, and the run logs, which record the
guidance gain.

    python scripts/03_benchmark_full.py

Writes results/benchmark_full.json.
"""
import json, os, re
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(ROOT, "results", "benchmark"); SEEDS = [42, 0, 1, 2, 3]
M = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Precision@5%", "Recall@5%", "FPR", "Brier_prob"]
rows = [json.load(open(f"{R}/seed_{s}_L6/revision.json"))["published_rows"]["+ Diffusion guidance (Full)"] for s in SEEDS]
gg = [float(re.search(r"guidance gain=([-+0-9.]+)", open(f"{R}/log_seed{s}_L6.txt").read()).group(1)) for s in SEEDS]
out = {m: [float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows], ddof=1))] for m in M}
out["guidance_gain_auc"] = [float(np.mean(gg)), float(np.std(gg, ddof=1))]
out["guidance_gain_per_seed"] = dict(zip(map(str, SEEDS), gg))
json.dump(out, open(os.path.join(ROOT, "results", "benchmark_full.json"), "w"), indent=2)
print(json.dumps(out, indent=1))

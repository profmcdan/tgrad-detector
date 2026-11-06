# T-GRAD: fraud detection on an augmented relational state

Code and results for the paper

> D. T. Ale, E. O. Ogunti, S. A. Oyetunji, A. A. Ponnle and K. B. Adedeji,
> *T-GRAD: Fraud Detection on an Augmented Relational State – Design, Evaluation, and Scaling
> for Cyclic Financial Crime*, manuscript, 2026.

T-GRAD scores each account from an augmented relational state of five named blocks: the
account's attributes, the state of an event-driven encoder, counts of the closed walks through
the account, the residual of a guided reverse diffusion, and the fraud logit of an energy model.
Removing a block and refitting only the readout gives an exact ablation. This repository
contains everything needed to reproduce the paper's tables and figures: the Elliptic
experiment with readout selection on a validation window, the scaling study on graphs of up to
two million edges, the latency measurements, and the saved per-seed results.

The encoder, the energy model and the closed-walk statistic are developed in the companion
paper on the nodal probability distribution model and the cyclic monitor, whose code and
benchmark generator are at https://github.com/profmcdan/tgrad. The model code in `src/` is
identical to that repository's.

## Layout

```
src/tgrad/data/dataset.py               benchmark generator (SyntheticCyclicAML) and Elliptic loader
src/tgrad/models/temporal_encoder_v2.py event-driven encoder (memory, time encoding, GRU update)
src/tgrad/models/cycle_guidance.py      energy head: the fraud logit f(z)
src/tgrad/models/diffusion.py           VP-SDE score network for the guided reverse diffusion
scripts/01_elliptic.py                  Elliptic: GNN baselines, T-GRAD, readout selection, per-step F1
scripts/02_scaling.py                   scaling study, 0.2M to 2M edges
scripts/03_benchmark_full.py            complete detector on the cyclic benchmark (five seeds)
scripts/04_summary.py                   tables (results/tables.md) and the per-step figure
scripts/05_detector_figure.py           Figure 1, the detector diagram
scripts/06_latency.py                   inference latency on Elliptic, every model timed the same way
results/                                the saved outputs behind every reported number
figures/                                the paper's figures
```

## Setup

Python 3.13 on CPU. The reported results were produced with torch 2.11.0, torch-geometric
2.7.0, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, pandas 3.0.2, matplotlib 3.10.9,
seaborn 0.13.2 and networkx 3.6.1, on an Apple M5 (10 cores, 24 GB).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Elliptic.** The Elliptic Bitcoin dataset is not redistributed here. Download it from Kaggle
(dataset `ellipticco/elliptic-data-set`) and place the three CSV files in
`data/elliptic_bitcoin_dataset/`:

```
data/elliptic_bitcoin_dataset/elliptic_txs_features.csv
data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv
data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv
```

The synthetic graphs need no download; they are generated from their seeds.

## Reproducing the paper

The tables can be rebuilt from the saved results in seconds:

```bash
python scripts/03_benchmark_full.py     # results/benchmark_full.json
python scripts/04_summary.py            # results/paper3_summary.json, results/tables.md, figures/robustness_paper3.png
```

To rerun the experiments themselves (seeds 42, 0 and 1; several hours on a laptop CPU):

```bash
bash scripts/run_all.sh
```

| Paper | Script | Output |
|---|---|---|
| Complete detector, cyclic benchmark | `03_benchmark_full.py` | `results/benchmark_full.json` |
| Complete detector, Elliptic; readout selection | `01_elliptic.py` | `results/elliptic_seed_*/paper3_final.json` |
| Inference latency | `06_latency.py` | `results/latency_elliptic.json` |
| Scaling study | `02_scaling.py` | `results/scaling_seed*.json` |
| Per-time-step F1 on Elliptic (figure) | `04_summary.py` | `figures/robustness_paper3.png` |
| Detector diagram (figure) | `05_detector_figure.py` | `figures/tgrad_detector_overview.png` |

The benchmark rows reuse the five-seed runs of the companion repository
(`scripts/03_benchmark.py` there, closed walks of length 2 to 6); their outputs and logs are
copied in `results/benchmark/`. The timings of the encoder and of the closed-walk block in the
scaling table come from `scripts/11_timing.py` there and are copied in `results/timing.json`.

## Protocol

On Elliptic the standard temporal split is used (train on time steps 1–34, test on 35–49,
labelled transactions only). The readout of T-GRAD, a Random Forest or a perceptron, is chosen
by PR-AUC on a validation window inside the training period (fit on steps 1–27, validate on
28–34); thresholds are set on training data. No choice uses the test split. On the synthetic
graphs the encoder and the energy head are trained on the labels of training entities only.

## Licence and citation

MIT licence. If you use this code, please cite the paper above; citation metadata are in
`CITATION.cff`.

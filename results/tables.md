# Tables of the paper (mean ± standard deviation over seeds)

## Complete detector

| Setting | Precision | Recall | F1 | ROC-AUC | PR-AUC | Recall@5% | FPR | Brier_prob |
|---|---|---|---|---|---|---|---|---|
| Benchmark, perceptron | 0.895 ± 0.026 | 0.533 ± 0.027 | 0.668 ± 0.024 | 0.919 ± 0.013 | 0.858 ± 0.019 | 0.141 ± 0.005 | 0.033 ± 0.010 | 0.129 ± 0.014 |
| Elliptic, Random Forest (selected) | 0.976 ± 0.007 | 0.614 ± 0.019 | 0.754 ± 0.014 | 0.932 ± 0.002 | 0.787 ± 0.003 | 0.696 ± 0.004 | 0.001 ± 0.000 | 0.024 ± 0.000 |
| Elliptic, perceptron | 0.910 ± 0.013 | 0.397 ± 0.028 | 0.552 ± 0.025 | 0.897 ± 0.004 | 0.637 ± 0.012 | 0.570 ± 0.015 | 0.003 ± 0.001 | 0.036 ± 0.002 |

Readout chosen on the validation window: Random Forest (forest, forest, forest); validation PR-AUC 0.985 (forest) and 0.958 (perceptron).
Guidance gain (ROC-AUC of the guided minus the unguided residual): benchmark +0.007 ± 0.019, Elliptic +0.016 ± 0.012.

## Scaling (ROC-AUC, three seeds)

| Edges | Entities | Closed-walk score | T-GRAD | GraphSAGE | GraphSAGE, matched | Encoder (events/s) | Closed walks (s) |
|---|---|---|---|---|---|---|---|
| 199,997 | 57,971 | 0.994 ± 0.000 | 0.994 ± 0.001 | 0.686 ± 0.002 | 0.994 ± 0.000 | 17,684 | 0.08 |
| 399,984 | 115,942 | 0.997 ± 0.000 | 0.997 ± 0.000 | 0.690 ± 0.004 | 0.997 ± 0.000 | 17,765 | 0.18 |
| 600,091 | 173,913 | 0.998 ± 0.000 | 0.998 ± 0.000 | 0.693 ± 0.011 | 0.998 ± 0.000 | 17,805 | 0.32 |
| 800,054 | 231,884 | 0.998 ± 0.000 | 0.998 ± 0.000 | 0.695 ± 0.008 | 0.998 ± 0.000 | 17,656 | 0.55 |
| 999,996 | 289,855 | 0.999 ± 0.000 | 0.999 ± 0.000 | 0.697 ± 0.009 | 0.999 ± 0.000 | 17,425 | 0.88 |
| 1,199,888 | 347,826 | 0.999 ± 0.000 | 0.999 ± 0.000 | 0.697 ± 0.008 | 0.999 ± 0.000 | 17,276 | 0.84 |
| 1,999,820 | 579,710 | 0.999 ± 0.000 | 0.999 ± 0.000 | 0.700 ± 0.010 | 0.999 ± 0.000 | 17,400 | 1.68 |

## Elliptic over time

T-GRAD ahead of GraphSAGE at 8 of 8 test steps before step 43; from step 43 the illicit-class F1 of both models is at most 0.07 in every seed.

## Inference latency on Elliptic (median of three runs, CPU)

| Model | Time (s) | ms per labelled transaction |
|---|---|---|
| GCN | 0.075 | 0.0016 |
| GraphSAGE | 0.189 | 0.0041 |
| GAT | 0.181 | 0.0039 |
| GIN | 0.105 | 0.0023 |
| Graph Transformer | 0.459 | 0.0099 |
| RF, attributes | 0.094 | 0.0020 |
| T-GRAD | 16.891 | 0.3627 |

T-GRAD stages: encoder 14.081 s, guided reverse diffusion 2.649 s, forest 0.128 s, energy head 0.033 s.
T-GRAD is 37 to 225 times slower per transaction than the GNNs.

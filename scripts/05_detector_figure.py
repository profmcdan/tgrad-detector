"""End-to-end diagram of the T-GRAD detector (Figure 1 of the paper); writes figures/tgrad_detector_overview.png."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 8, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(figsize=(7.6, 5.0)); ax.set_xlim(0, 100); ax.set_ylim(-7, 67); ax.axis("off")
C_IN, C_LRN, C_STR, C_OUT = "#eef2f7", "#e6f0e3", "#fbeee0", "#f4f4f4"
def box(x, y, w, h, title, body, fc, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec="#333333", lw=lw))
    nt = title.count("\n") + 1
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontweight="bold", fontsize=8.1, linespacing=1.15)
    ax.text(x + w / 2, y + h - 1.6 - 3.3 * nt, body, ha="center", va="top", fontsize=7.3, linespacing=1.4)
def arrow(p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#222222",
                                 connectionstyle=f"arc3,rad={rad}"))
def label(x, y, s, ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=6.9, style="italic", color="#222222")
box(0.5, 24, 13, 16, "Transaction\nstream", "events\n$e=(u,v,t,\\mathbf{a})$\nin time order", C_IN)
box(18, 40, 18, 17, "Event-driven\nencoder", "GRU memory,\ntime encoding\n$\\to$ state $\\mathbf{z}_i$", C_LRN)
box(18, 2, 18, 16, "Observed\ngraph", "adjacency $A$\nof all events\n(transductive)", C_IN)
box(41, 52, 19, 9, "Attributes", "$\\mathbf{a}_i$ of entity $i$", C_IN)
box(41, 38.5, 19, 10, "Energy head", "fraud logit $f(\\mathbf{z}_i)$", C_LRN)
box(41, 19, 19, 16, "Guided reverse\ndiffusion", "from noised $\\mathbf{z}_i$,\nguided by $-\\nabla f$\n$\\to$ residual $r_i$", C_LRN)
box(41, 0, 19, 16.5, "Closed walks\nand degrees", "$(A^L)_{ii}$, $L=2,\\dots,6$;\nin-, out-degree\n$\\to\\mathbf{m}_i$", C_STR)
box(65, 13, 15, 33, "Augmented\nstate $\\zeta_i$", "$[\\,\\mathbf{a}_i\\,\\|\\,\\mathbf{z}_i\\,\\|$\n$\\mathbf{m}_i\\,\\|\\,r_i\\,\\|$\n$f(\\mathbf{z}_i)\\,]$\n\nfive named\nblocks,\nstandardised\non training\nstatistics", C_OUT, lw=1.3)
box(84, 17.5, 15.5, 23.5, "Readout $g_\\omega$", "perceptron or\nforest\n$\\to s_i\\in[0,1]$\n\nthreshold $\\tau$\nfrom training\nPR curve", C_OUT)

arrow((13.8, 36), (17.7, 47)); label(13.2, 45, "events", ha="right")
arrow((13.8, 28), (17.7, 11)); label(13.2, 17, "events", ha="right")
arrow((36.3, 49), (40.7, 43.5)); arrow((36.3, 44), (40.7, 29))
arrow((50.5, 38.2), (50.5, 35.4)); label(54.5, 36.8, "$\\nabla f$")
arrow((36.3, 10), (40.7, 9))
arrow((60.3, 56), (64.7, 44)); arrow((60.3, 43.5), (64.7, 39)); arrow((60.3, 27), (64.7, 28)); arrow((60.3, 8.5), (64.7, 16))
ax.plot([27, 27, 72.5], [57.3, 64.5, 64.5], color="#222222", lw=0.9)
arrow((72.5, 64.5), (72.5, 46.4)); label(50, 66, "state $\\mathbf{z}_i$ (block E)")
arrow((80.3, 30), (83.7, 30))
ax.text(50, -2.4, "Trained jointly: class-weighted cross-entropy on $f$ and denoising score matching on licit states.",
        ha="center", va="center", fontsize=7.0, color="#2c5d2c")
ax.text(50, -5.6, "Exact ablation: delete a named block of $\\zeta_i$ and refit only $g_\\omega$; the encoder and the other blocks stay fixed.",
        ha="center", va="center", fontsize=7.0, color="#8a4b00")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "tgrad_detector_overview.png")
fig.savefig(out, dpi=300, bbox_inches="tight"); print("wrote", out)

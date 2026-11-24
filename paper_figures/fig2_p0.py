"""
Figure 2: first-attempt pass rate P(0).
(a) view-on heatmap, (b) view-off heatmap (models x tasks + mean column),
(c) per-model Delta P(0) diverging bar (no significance marks).

Style follows AtomWorld Fig. 2: viridis-like green-to-yellow scale,
per-cell percentage, Mean column, compact sans-serif typography.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Helvetica",
})

from plot_data import display_data, MODELS, DISPLAY_N, DISPLAY_POOLED_N, save_numeric
models = MODELS
rows = ["Mean"] + models
task_labels = ["delete\nbelow\nIncl.", "delete\nbelow\nExcl.",
               "delete\naround", "insert\nbetween", "move\ntowards",
               "rotate\naround", "remove\nmolecule"]
tasks = task_labels + ["Mean"]
on = np.array([display_data[m,"ON"][0] for m in models],dtype=float)
off = np.array([display_data[m,"OFF"][0] for m in models],dtype=float)
on_pool = on.sum(axis=0)
off_pool = off.sum(axis=0)
task_delta = (on_pool - off_pool) / DISPLAY_POOLED_N * 100.0

def pct(mat):
    m = mat / DISPLAY_N * 100
    col_mean = mat.sum(axis=0,keepdims=True) / DISPLAY_POOLED_N * 100
    weighted_mean = mat.sum(axis=1,keepdims=True) / DISPLAY_N.sum() * 100
    body = np.hstack([m, weighted_mean])
    # The overall mean stays sample-weighted across the original 180 instances.
    return np.vstack([np.hstack([col_mean, [[np.nan]]]), body])

ON, OFF = pct(on), pct(off)
delta = ON[1:, -1] - OFF[1:, -1]   # model rows only

# AtomWorld uses a green->yellow scale; extend to a deep low end
cmap = plt.get_cmap("viridis")

def heat(ax, data, title, cax, show_ylabels=True):
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_facecolor("white")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows if show_ylabels else [], fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xticks(np.arange(-.5, len(tasks), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    for i in range(data.shape[0]):
        is_mean_row = (i == 0)
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "", ha="center", va="center")
                continue
            rgba = cmap(v / 100.0)
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            color = "black" if lum > 0.6 else "white"
            fw = "bold" if (is_mean_row or j == data.shape[1] - 1) else "normal"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=8.6 if not is_mean_row else 9.4,
                    color=color, fontweight=fw)
    # separate Mean row (top) and Mean column (right)
    ax.axhline(0.5, color="#333333", lw=1.4)
    ax.axvline(data.shape[1] - 1.5, color="#333333", lw=1.4)
    cb = plt.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label("First-attempt pass rate (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return im

fig = plt.figure(figsize=(13.0, 6.8))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05],
                      height_ratios=[1.05, 0.92], hspace=0.82, wspace=0.28,
                      left=0.15, right=0.9, top=0.92, bottom=0.10)
axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1])
cxa = fig.add_subplot(gs[0, 2])
heat(axa, ON, "(a) View ON: P(0)", cxa, show_ylabels=True)
heat(axb, OFF, "(b) View OFF: P(0)", cxa, show_ylabels=False)
axb.set_ylabel(""); axa.set_ylabel("")

# (c) per-model delta
axc = fig.add_subplot(gs[1, 0])
order = np.argsort(delta)
dm = delta[order]; ml = [models[i] for i in order]
colors = ["#c0504d" if v < 0 else "#2e7d32" for v in dm]
axc.barh(range(len(dm)), dm, color=colors, edgecolor="#333", lw=.5, height=.62)
axc.set_yticks(range(len(dm))); axc.set_yticklabels(ml, fontsize=8.5)
axc.axvline(0, color="#333", lw=1)
axc.set_xlim(-16, 16)
axc.set_xlabel("ΔP(0) (percentage points)", fontsize=8.5)
axc.set_title("(c) Per-model effect", fontsize=10.5, fontweight="bold", pad=5)
for i, v in enumerate(dm):
    axc.text(v + (0.5 if v >= 0 else -0.5), i, f"{v:+.1f}",
             va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
axc.grid(axis="x", color="#dddddd", lw=.6); axc.set_axisbelow(True)
for sp in ["top", "right"]: axc.spines[sp].set_visible(False)
axc.tick_params(labelsize=8)

# (d) per-task delta (pooled over the five models)
axd = fig.add_subplot(gs[1, 1])
order2 = np.argsort(task_delta)
dt = task_delta[order2]; tl = [('delete below\nIncl.' if i==0 else 'delete below\nExcl.' if i==1 else task_labels[i]) for i in order2]
colors2 = ["#c0504d" if v < 0 else "#2e7d32" for v in dt]
axd.barh(range(len(dt)), dt, color=colors2, edgecolor="#333", lw=.5, height=.62)
axd.set_yticks(range(len(dt))); axd.set_yticklabels(tl, fontsize=8.5)
axd.axvline(0, color="#333", lw=1)
axd.set_xlim(-11, 11)
axd.set_xlabel("ΔP(0) (percentage points)", fontsize=8.5)
axd.set_title("(d) Per-task effect (5 models pooled)", fontsize=10.5,
              fontweight="bold", pad=5)
for i, v in enumerate(dt):
    axd.text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.1f}",
             va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
axd.grid(axis="x", color="#dddddd", lw=.6); axd.set_axisbelow(True)
for sp in ["top", "right"]: axd.spines[sp].set_visible(False)
axd.tick_params(labelsize=8)

from original_style import apply_original_typography
# Explicit math font roles avoid the Apple Chancery fallback in recent Matplotlib.
plt.rcParams.update({"mathtext.it":"DejaVu Sans:italic", "mathtext.bf":"DejaVu Sans:bold",
                     "mathtext.cal":"DejaVu Sans", "mathtext.sf":"DejaVu Sans",
                     "mathtext.tt":"DejaVu Sans Mono"})
apply_original_typography(fig)
fig.savefig(os.path.join(OUT, "fig2_p0.pdf"), bbox_inches="tight",
            pad_inches=0.03)
fig.savefig(os.path.join(OUT, "fig2_p0.png"), dpi=300, bbox_inches="tight",
            pad_inches=0.03)
print("saved fig2_p0.pdf / .png")

save_numeric("fig2_p0", {"on_counts":on.astype(int).tolist(),"off_counts":off.astype(int).tolist(),"model_delta":delta.tolist(),"task_delta":task_delta.tolist()})

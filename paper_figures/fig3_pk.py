"""
Figure 3: retained successes by historical repair round.
Top row x5: per-model cumulative retained-rate curves, view ON/OFF, k=0..5.
Bottom: arithmetic per-round retained fraction (all 1800 configs pooled).
Data come from the specified Markdown; not a new online protocol rerun.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

from plot_data import data, MODELS, total, save_numeric
models = MODELS

def pk(model, cond):
    a = np.array(data[(model, cond)], dtype=float).sum(axis=1)
    return np.cumsum(a[:6]) / a.sum() * 100.0, a

# The original bar form is preserved. Denominator is arithmetic after final filtering,
# not the observed number of enhanced-protocol repair entrants.
round_pass = total[1:6].astype(float)
entering = np.array([total.sum()-total[:k].sum() for k in range(1,6)],dtype=float)
rescue_rate = round_pass / entering * 100
print("arithmetic denominator", entering.astype(int), "retained fraction%", np.round(rescue_rate,1))

C_ON = "#1f6f43"; C_OFF = "#b25928"
fig = plt.figure(figsize=(11.0, 4.3))
# (a)-(e) line panels; (f) bars; (g) summary

gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1.0], hspace=0.72, wspace=0.0,
                      left=0.055, right=0.985, top=0.90, bottom=0.13)
axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
for ax, m in zip(axes, models):
    on,_ = pk(m, "ON"); off,_ = pk(m, "OFF")
    ks = np.arange(6)
    ax.plot(ks, on, "-o", color=C_ON, lw=1.6, ms=3.4, label="view ON")
    ax.plot(ks, off, "--s", color=C_OFF, lw=1.6, ms=3.2, mfc="white",
            label="view OFF")
    letter = ["(a)","(b)","(c)","(d)","(e)"][axes.index(ax)]
    ax.set_title(f"{letter} {m}", fontsize=9.5, fontweight="bold", pad=4)
    ax.set_ylim(45, 100); ax.set_xticks(ks); ax.tick_params(labelsize=7.5)
    ax.grid(color="#dddddd", lw=.5); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    if ax is not axes[0]: ax.set_yticklabels([])
axes[0].set_ylabel("Cumulative rate $P(k)$ (%)", fontsize=9)
axes[2].set_xlabel("Repair round $k$ (0 = first attempt)", fontsize=9)
handles = [plt.Line2D([0],[0], color=C_ON, marker="o", ls="-", lw=1.6, label="view ON"),
           plt.Line2D([0],[0], color=C_OFF, marker="s", ls="--", lw=1.6, mfc="white",
                      label="view OFF")]
axes[0].legend(handles=handles, fontsize=7.5, frameon=False, loc="lower right",
               borderaxespad=0.2)

# bottom: rescue bars, horizontally centered (spans middle columns)
axb = fig.add_subplot(gs[1, :])
colors = ["#2e8b57", "#4fa372", "#73b88b", "#99cda6", "#bfe1c2"]
bars = axb.bar(np.arange(1,6), rescue_rate, color=colors, edgecolor="#333",
              lw=.5, width=.62)
for k, (v, n) in enumerate(zip(rescue_rate, round_pass)):
    axb.text(k+1, v+1.2, f"{v:.1f}%\n({int(n)})", ha="center", va="bottom",
             fontsize=7.8, linespacing=1.15)
ax.set_ylim(45,100)
axb.set_xticks(np.arange(1,6)); axb.set_xticklabels([f"repair {i}" for i in range(1,6)], fontsize=8)
axb.set_ylim(0,46); axb.set_ylabel("Per-round success fraction (%)", fontsize=9)
axb.set_title("(f) New successes per repair round (all configs pooled)", fontsize=9.5,
              fontweight="bold", pad=4)
axb.grid(axis="y", color="#dddddd", lw=.5); axb.set_axisbelow(True)
for sp in ["top","right"]: axb.spines[sp].set_visible(False)
axb.tick_params(labelsize=7.8)

from original_style import apply_original_typography
apply_original_typography(fig)
fig.savefig(os.path.join(OUT, "fig3_pk.pdf"), bbox_inches="tight", pad_inches=0.03)
fig.savefig(os.path.join(OUT, "fig3_pk.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
print("saved fig3_pk")

save_numeric("fig3_pk", {"model_curves":{m:{v:pk(m,v)[0].tolist() for v in ["ON","OFF"]} for m in models},"total_seven":total.tolist(),"repair_pass":round_pass.astype(int).tolist(),"arithmetic_denominators":entering.astype(int).tolist(),"repair_fraction":rescue_rate.tolist()})

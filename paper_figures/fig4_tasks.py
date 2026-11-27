"""
Figure 4: task-level differences, split by visual condition.
(a) P(k) per display group, view ON (5 models pooled; 50/100/150 instances)
(b) P(k) per display group, view OFF
(c) final failures by type, view ON
(d) final failures by type, view OFF
P(k) data: six-task Markdown plus a prompt-labelled Delete Below split;
cumulative retained success counts by historical repair bucket,
not a new enhanced-online-protocol rerun.
Failure types: separately supplied archived evidence, checked against all
14 display-group/view final failure totals (217 final failures total).
"""
import os, json, collections
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

from plot_data import DISPLAY_TASKS, DISPLAY_POOLED_N, display_pooled, display_failures, save_numeric
tasks = [t.lower() for t in DISPLAY_TASKS]
colors = ["#4C78A8", "#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
rows = {v:{name:display_pooled[v][:,i].tolist() for i,name in enumerate(tasks)} for v in ["ON","OFF"]}
denominators = dict(zip(tasks,DISPLAY_POOLED_N))
fail = {v:{name:tuple(display_failures[v,original,t] for t in ["AtomCountMismatch","StructureMismatch","NearInitialStructure"]) for original,name in zip(DISPLAY_TASKS,tasks)} for v in ["ON","OFF"]}
for v in ["ON","OFF"]:
    for name in tasks: assert sum(fail[v][name]) == rows[v][name][6]

C_ACM, C_SM, C_NOOP = "#C65911", "#3A6EA5", "#9B5858"
fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22,
                      left=0.075, right=0.985, top=0.93, bottom=0.075)
fig.delaxes(axes[0][0]); fig.delaxes(axes[0][1])
fig.delaxes(axes[1][0]); fig.delaxes(axes[1][1])
ax_curve = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
ax_bar   = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
ks = np.arange(6)

for j, view in enumerate(["ON", "OFF"]):
    ax = ax_curve[j]
    for name, c in zip(tasks, colors):
        arr = np.array(rows[view][name], float)
        pk = np.cumsum(arr[:6]) / denominators[name] * 100
        # The two Delete Below groups share the original task colour;
        # a dashed hollow-square line distinguishes Excluding.
        excluding = '(excluding)' in name
        ax.plot(ks, pk, "--s" if excluding else "-o", color=c, lw=1.8, ms=4.0,
                mfc='white' if excluding else c)
    ax.set_title(f"({'ab'[j]}) Cumulative pass rate, view {view}",
                 fontsize=11, fontweight="bold", pad=6)
    ax.set_xticks(ks); ax.set_ylim(0, 103); ax.set_xlim(-0.15, 5.15)
    ax.set_xlabel("Repair round $k$ (0 = first attempt)", fontsize=9.5)
    if j == 0: ax.set_ylabel("Cumulative pass rate $P(k)$ (%)", fontsize=9.5)
    ax.grid(color="#dddddd", lw=.5); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    # legend carries P(0)->P(5) so no endpoint labels are needed
    leg = [f"{n.replace('(including)','(Incl.)').replace('(excluding)','(Excl.)')}  ({rows[view][n][0]/denominators[n]*100:.1f} $\u2192$ "
           f"{sum(rows[view][n][:6])/denominators[n]*100:.1f})" for n in tasks]
    ax.legend(leg, fontsize=7.6, frameon=False, loc="lower right",
              borderaxespad=.25, handlelength=1.6, labelspacing=.45,
              title="action  ($P(0) \u2192 P(5)$)", title_fontsize=7.8)

for j, view in enumerate(["ON", "OFF"]):
    ax = ax_bar[j]
    x = np.arange(len(tasks))
    acm = np.array([fail[view][t][0] for t in tasks])
    sm  = np.array([fail[view][t][1] for t in tasks])
    noop = np.array([fail[view][t][2] for t in tasks])
    ax.bar(x, acm, color=C_ACM, edgecolor="white", lw=.8, width=.66,
           label="AtomCountMismatch (composition)")
    ax.bar(x, sm, bottom=acm, color=C_SM, edgecolor="white", lw=.8,
           width=.66, label="StructureMismatch (geometry)")
    if noop.sum():
        ax.bar(x, noop, bottom=acm+sm, color=C_NOOP, edgecolor="white", lw=.8,
               width=.66, label="NearInitialStructure (no-op)")
    # Tiny segments cannot fit an 8.5pt digit inside a 1-2 count bar.
    # Keep the original bars, but place these labels beside the segment.
    for i in range(len(tasks)):
        if acm[i] >= 3:
            ax.text(i, acm[i]/2, str(int(acm[i])), ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")
        elif acm[i] > 0:
            ax.annotate(str(int(acm[i])), xy=(i,acm[i]/2), xytext=(i-.32,acm[i]/2+1),
                        fontsize=8.5, ha="right", va="center", color=C_ACM,
                        arrowprops=dict(arrowstyle="-",color=C_ACM,lw=.5))
        if sm[i] >= 3:
            ax.text(i, acm[i] + sm[i]/2, str(int(sm[i])), ha="center",
                    va="center", fontsize=8.5, color="white", fontweight="bold")
        elif sm[i] > 0:
            ax.annotate(str(int(sm[i])), xy=(i,acm[i]+sm[i]/2), xytext=(i-.32,acm[i]+sm[i]/2+1),
                        fontsize=8.5, ha="right", va="center", color=C_SM,
                        arrowprops=dict(arrowstyle="-",color=C_SM,lw=.5))
        if noop[i] >= 3:
            ax.text(i, acm[i] + sm[i] + noop[i]/2, str(int(noop[i])), ha="center",
                    va="center", fontsize=8.5, color="white", fontweight="bold")
        elif noop[i] > 0:
            ax.annotate(str(int(noop[i])), xy=(i,acm[i]+sm[i]+noop[i]/2), xytext=(i+.32,acm[i]+sm[i]+noop[i]/2+1),
                        fontsize=8.5, ha="left", va="center", color=C_NOOP,
                        arrowprops=dict(arrowstyle="-",color=C_NOOP,lw=.5))
        ax.text(i, acm[i] + sm[i] + noop[i] + 1.6, f"n={int(acm[i]+sm[i]+noop[i])}",
                ha="center", fontsize=8.5, color="#333")
    ax.set_xticks(x)
    bar_labels = ["delete\nbelow\nIncl.", "delete\nbelow\nExcl."] + [t.replace(' ','\n') for t in tasks[2:]]
    ax.set_xticklabels(bar_labels, fontsize=8.2)
    ax.set_ylim(0, 68)
    ax.set_title(f"({'cd'[j]}) Final failures by error type, view {view}",
                 fontsize=11, fontweight="bold", pad=6)
    ax.set_ylabel("Number of final failures", fontsize=9.5)
    ax.grid(axis="y", color="#dddddd", lw=.5); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left", borderaxespad=.2)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

from original_style import apply_original_typography
apply_original_typography(fig)
fig.savefig(os.path.join(OUT, "fig4_tasks.pdf"), bbox_inches="tight",
            pad_inches=0.03)
fig.savefig(os.path.join(OUT, "fig4_tasks.png"), dpi=300,
            bbox_inches="tight", pad_inches=0.03)
print("saved fig4_tasks")

save_numeric("fig4_tasks", {"rows":rows,"failure_types":fail,"denominators":{k:int(v) for k,v in denominators.items()}})

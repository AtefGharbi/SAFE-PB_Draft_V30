import sys; sys.path.insert(0, "/home/claude/work/paper")
import numpy as np, pandas as pd
from fig_common import *
df = pd.read_csv("/home/claude/work/safe_pb_v23/results/safety_revalidation_cases.csv")
df = df.sort_values("candidate_min_v_ac").reset_index(drop=True)
x = np.arange(len(df))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [2.3, 1]})
ax = axes[0]
ax.axhline(0.95, color=INK, lw=0.9, ls="--", zorder=1)
ax.text(len(df) - 1, 0.9503, "$v_{\\min}$ = 0.95 p.u.", ha="right", va="bottom", color=MUTED, fontsize=7.5)
ax.scatter(x, df.candidate_min_v_ac, s=9, color=GRAY, label="Candidate (unprojected)", zorder=2)
ax.scatter(x, df.soft_min_v_ac, s=9, color=ORANGE, marker="s", label="Soft-penalty QP (legacy)", zorder=3)
ax.scatter(x, df.lexicographic_min_v_ac, s=9, color=VIOLET, marker="^", label="Lexicographic, no AC correction", zorder=3)
ax.scatter(x, df.ac_corrected_min_v_ac, s=10, color=BLUE, marker="o", label="AC-corrected (proposed)", zorder=4)
bad = ~df.authority_feasible
ax.scatter(x[bad], df.authority_best_min_v_ac[bad], s=40, facecolors="none", edgecolors=INK, lw=0.8,
           label="Authority bound (infeasible case)", zorder=5)
ax.set_xlabel("Qualifying stress case (sorted by candidate voltage)")
ax.set_ylabel("Minimum AC bus voltage (p.u.)")
ax.legend(loc="lower right", fontsize=6.8, ncol=1, handletextpad=0.3)
ax.set_title("(a) Post-projection AC voltage, 168 cases", loc="left")
ax2 = axes[1]
fa = df.authority_feasible
names = ["Soft-penalty QP", "Lexicographic", "AC-corrected"]
vals = [int((df.soft_ac_feasible & fa).sum()), int((df.lexicographic_ac_feasible & fa).sum()),
        int((df.ac_corrected_ac_feasible & fa).sum())]
bars = ax2.barh(names[::-1], vals[::-1], color=[BLUE, VIOLET, ORANGE], height=0.55, edgecolor="white", linewidth=1.5)
for b, v in zip(bars, vals[::-1]):
    ax2.text(v + 4, b.get_y() + b.get_height() / 2, f"{v}/{int(fa.sum())}", va="center", fontsize=7.5, color=INK)
ax2.set_xlim(0, 215); ax2.set_xlabel("AC-feasible after projection")
ax2.set_title("(b) Restored, authority-feasible cases", loc="left", fontsize=8.5); ax2.grid(axis="y", visible=False); ax2.set_axisbelow(True)
fig.tight_layout(); save(fig, "fig3_safety_revalidation")
print("ok")

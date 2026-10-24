import sys; sys.path.insert(0, "/home/claude/work/paper")
from fig_common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(7.2, 3.7)); ax.set_xlim(0, 104); ax.set_ylim(0, 54); ax.axis("off")
def box(x, y, w, h, title, body, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc="white", ec=edge, lw=1.3))
    ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=6.6, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=6, color=MUTED, linespacing=1.4)
def arrow(p1, p2, label="", off=(0, 1.0), ls="-", ha="center"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=INK, linestyle=ls))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1], label, ha=ha, va="bottom", fontsize=6, color=INK)
# top row (execution, one time slot)
box(0.5, 32, 22, 18, "Utility SAC actor", "time, wholesale price,\nload/PV forecast,\nlast AC min. voltage\n$\\rightarrow$ category prices $\\eta_{c,t}$", BLUE)
box(27, 32, 22, 18, "Household actor", "local demand, SoC,\navailability, price\n(parameter-shared)\n$\\rightarrow$ candidate $\\hat{a}_t$", BLUE)
box(53.5, 32, 23, 18, "Safety projection", "lexicographic: (i) min-slack LP\n(ii) min-distance QP\nLinDistFlow + bias $\\delta^{(k)}$\n$\\rightarrow a_t^{(k)}$", ORANGE)
box(81, 32, 22.5, 18, "Feeder & devices", "apply $a_t^{\\star}$; update\nEV/battery SoC,\nbills, discomfort,\nreward $r_t$", GRAY)
# bottom row
box(0.5, 3, 48.5, 18, "Centralised training (CTDE)", "receives transitions $(s_t,\\hat{a}_t,a_t^{\\star},r_t)$ and daily losses;\ncentralised critics, replay buffer, alternating updates (K = 24);\ncomplete-day CVaR and discomfort-variance terms", GRAY)
box(53.5, 3, 23, 18, "AC power flow", "Newton–Raphson,\nIEEE 33-bus feeder\n$\\delta^{(k+1)}=V^{AC}-V^{lin}$\nstop when feasible", ORANGE)
box(81, 3, 22.5, 18, "Authority certificate", "all devices at lower\nbound: is any action\nAC-feasible? Labels\nviolations (no control)", AQUA)
arrow((23.1, 41), (26.4, 41), "$\\eta_{c,t}$")
arrow((49.6, 41), (52.9, 41), "$\\hat{a}_t$")
arrow((77.1, 41), (80.4, 41), "$a_t^{\\star}$")
arrow((62, 31.4), (62, 21.6), "$a_t^{(k)}$", off=(-1, -1), ha="right")
arrow((69, 21.6), (69, 31.4), "$\\delta^{(k+1)}$", off=(1, -1), ha="left")
arrow((92, 31.4), (92, 21.6), "device bounds", off=(1, -1), ha="left")
arrow((14, 21.6), (11, 31.4), "", ls="--")
arrow((30, 21.6), (36, 31.4), "", ls="--")
ax.text(22, 25.2, "gradient updates", fontsize=6, color=INK, ha="center")
save(fig, "fig1_architecture")

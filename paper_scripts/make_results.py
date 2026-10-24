"""Build manuscript tables (markdown) and figures from a publication run directory."""
import sys, json, ast
sys.path.insert(0, "/home/claude/work/paper")
import numpy as np, pandas as pd
from fig_common import *

RUN = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/work/paper/results_tables.md"
ps = pd.read_csv(f"{RUN}/per_seed_means.csv")
pst = pd.read_csv(f"{RUN}/paired_statistics.csv")
ORDER = ["SAFE-PB", "SAFE-PB_no_safety", "SAFE-PB_no_CVaR", "SAFE-PB_no_fairness", "SAFE-PB_single_price",
         "No-DR", "TOU", "Tabular_Q_learning", "MPC"]
LABEL = {"SAFE-PB": "SAFE-PB", "SAFE-PB_no_safety": "SAFE-PB, no safety", "SAFE-PB_no_CVaR": "SAFE-PB, no CVaR",
         "SAFE-PB_no_fairness": "SAFE-PB, no fairness", "SAFE-PB_single_price": "SAFE-PB, single price",
         "No-DR": "No-DR", "TOU": "TOU", "Tabular_Q_learning": "Tabular Q-learning [1]", "MPC": "MPC"}
methods = [m for m in ORDER if m in set(ps.method)]
nseeds = ps.training_seed.nunique()


def ms(prof, m, col, fmt="{:,.0f}"):
    x = ps[(ps.scenario_profile == prof) & (ps.method == m)][col].astype(float)
    if x.isna().all():
        return "–"
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return f"{fmt.format(x.mean())} ± {fmt.format(sd)}"


def mean(prof, m, col):
    return float(ps[(ps.scenario_profile == prof) & (ps.method == m)][col].astype(float).mean())


lines = []
# Table: nominal economics
lines.append("NOMINAL_TABLE")
hdr = "| Method | Daily loss $L_d$ | Customer cost $C_d$ | Utility profit $U_d$ | PAR | Discomfort variance | Jain index |"
lines += [hdr, "|---|---|---|---|---|---|---|"]
for m in methods:
    lines.append(f"| {LABEL[m]} | {ms('nominal', m, 'weighted_social_loss')} | {ms('nominal', m, 'customer_cost')} | "
                 f"{ms('nominal', m, 'utility_profit')} | {ms('nominal', m, 'PAR', '{:.3f}')} | "
                 f"{ms('nominal', m, 'normalized_discomfort_variance', '{:.4f}')} | {ms('nominal', m, 'jain_discomfort', '{:.3f}')} |")
lines.append("")
lines.append("BINDING_TABLE")
hdr = ("| Method | Violation min/day | Controllable min/day | Exogenous min/day | Restored (%) | "
       "Min AC voltage (p.u.) | Projection active (%) | Daily loss $L_d$ |")
lines += [hdr, "|---|---|---|---|---|---|---|---|"]
for m in methods:
    rr = ps[(ps.scenario_profile == "binding_voltage") & (ps.method == m)]["candidate_violation_restored_rate"].astype(float)
    rr_s = "–" if rr.isna().all() else f"{100*rr.mean():.1f}"
    lines.append(f"| {LABEL[m]} | {ms('binding_voltage', m, 'voltage_violation_minutes', '{:.1f}')} | "
                 f"{ms('binding_voltage', m, 'controllable_violation_minutes', '{:.1f}')} | "
                 f"{ms('binding_voltage', m, 'exogenous_violation_minutes', '{:.1f}')} | {rr_s} | "
                 f"{ms('binding_voltage', m, 'min_voltage_ac', '{:.4f}')} | "
                 f"{100*mean('binding_voltage', m, 'projection_active_rate'):.1f} | "
                 f"{ms('binding_voltage', m, 'weighted_social_loss')} |")
lines.append("")
lines.append("PAIRED_TABLE")
lines += ["| Comparator | Nominal: daily loss, Δ [95% CI] | $p_{Holm}$ | Binding: controllable violation min/day, Δ [95% CI] | $p_{Holm}$ | Binding: daily loss, Δ [95% CI] | $p_{Holm}$ |",
          "|---|---|---|---|---|---|---|"]
def cell(prof, comp, met, f):
    r = pst[(pst.scenario_profile == prof) & (pst.comparison == comp) & (pst.metric == met)]
    if r.empty: return "–", "–"
    r = r.iloc[0]
    ci = "" if pd.isna(r.ci_low) else f" [{f.format(r.ci_low)}, {f.format(r.ci_high)}]"
    p = "–" if pd.isna(r.p_holm) else (f"{r.p_holm:.3f}" if r.p_holm >= 0.001 else "<0.001")
    return f"{f.format(r.mean_paired_difference)}{ci}", p
for comp in methods[1:]:
    a = cell("nominal", comp, "weighted_social_loss", "{:,.0f}")
    b = cell("binding_voltage", comp, "controllable_violation_minutes", "{:,.0f}")
    c = cell("binding_voltage", comp, "weighted_social_loss", "{:,.0f}")
    lines.append(f"| {LABEL[comp]} | {a[0]} | {a[1]} | {b[0]} | {b[1]} | {c[0]} | {c[1]} |")
open(OUT, "w").write("\n".join(lines) + "\n")

# ---- numbers for text
facts = {"n_seeds": int(nseeds)}
for prof in ("nominal", "binding_voltage"):
    for m in methods:
        for col in ("weighted_social_loss", "customer_cost", "utility_profit", "PAR", "voltage_violation_minutes",
                    "controllable_violation_minutes", "exogenous_violation_minutes", "min_voltage_ac",
                    "candidate_violation_restored_rate", "projection_active_rate", "normalized_discomfort_variance",
                    "jain_discomfort", "CVaR90", "candidate_voltage_violation_minutes", "projection_distance_sum"):
            facts[f"{prof}|{m}|{col}"] = mean(prof, m, col)
json.dump(facts, open(OUT.replace(".md", "_facts.json"), "w"), indent=1)

# ---- Fig 4: binding-profile violation attribution
fig, ax = plt.subplots(figsize=(7.2, 2.9))
bm = methods
ctrl = [mean("binding_voltage", m, "controllable_violation_minutes") for m in bm]
exo = [mean("binding_voltage", m, "exogenous_violation_minutes") for m in bm]
y = np.arange(len(bm))[::-1]
ax.barh(y, exo, color=GRAY, height=0.62, label="Exogenous (no feasible action)", edgecolor="white", linewidth=1.5)
ax.barh(y, ctrl, left=exo, color=ORANGE, height=0.62, label="Controllable (feasible action existed)", edgecolor="white", linewidth=1.5)
for yi, c, e in zip(y, ctrl, exo):
    ax.text(e + c + 6, yi, f"{c:.0f}", va="center", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels([LABEL[m] for m in bm], fontsize=7.5)
ax.set_xlabel("AC voltage-violation minutes per day, binding profile (mean over seeds)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
ax.set_xlim(0, max(np.add(ctrl, exo)) * 1.12)
fig.tight_layout(); save(fig, "fig4_violation_attribution")

# ---- Fig 5: hourly voltage, binding profile
ep = pd.read_csv(f"{RUN}/episode_results.csv", usecols=["method", "scenario_profile", "post_min_voltage_ac_by_step",
                                                         "candidate_min_voltage_ac_by_step"])
ep = ep[ep.scenario_profile == "binding_voltage"]
def hourly(m, col):
    arr = np.array([ast.literal_eval(s) for s in ep[ep.method == m][col]], float)
    return arr.mean(0), arr.min(0)
fig, ax = plt.subplots(figsize=(7.2, 2.8))
hrs = np.arange(24)
show = [("SAFE-PB", "candidate_min_voltage_ac_by_step", GRAY, "SAFE-PB candidate (before projection)", "-"),
        ("SAFE-PB_no_safety", "post_min_voltage_ac_by_step", ORANGE, "SAFE-PB, no safety", "-"),
        ("TOU", "post_min_voltage_ac_by_step", VIOLET, "TOU", "--"),
        ("SAFE-PB", "post_min_voltage_ac_by_step", BLUE, "SAFE-PB (applied)", "-")]
for m, col, c, lab, ls in show:
    if m in set(ep.method):
        mu, _ = hourly(m, col)
        ax.plot(hrs, mu, color=c, ls=ls, label=lab, lw=1.8)
# authority bound per hour from SAFE-PB not stored; shade exogenous hours via certificate counts unavailable here
ax.axhline(0.95, color=INK, lw=0.9, ls=":")
ax.text(0.2, 0.9503, "$v_{\\min}$", color=MUTED, fontsize=7.5, va="bottom")
ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean daily minimum\nAC voltage (p.u.)")
ax.set_xticks(range(0, 24, 3)); ax.legend(fontsize=7, ncol=2, loc="lower left")
fig.tight_layout(); save(fig, "fig5_hourly_voltage")

# ---- Fig 6: training curves
import glob
tr = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RUN}/train_*.csv")])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
cols = {"SAFE-PB": BLUE, "SAFE-PB_no_safety": ORANGE, "SAFE-PB_no_CVaR": AQUA, "SAFE-PB_no_fairness": VIOLET,
        "SAFE-PB_single_price": GRAY}
for ax, prof, title in ((axes[0], "nominal", "(a) Nominal-profile episodes"),
                        (axes[1], "binding_voltage", "(b) Binding-profile episodes")):
    for m, c in cols.items():
        d = tr[(tr.method == m) & (tr.scenario_profile == prof)]
        if d.empty:
            continue
        piv = d.pivot_table(index="episode", columns="training_seed", values="weighted_social_loss")
        sm = piv.rolling(15, min_periods=1).mean()
        mu, sd = sm.mean(1), sm.std(1)
        ax.plot(sm.index, mu, color=c, lw=1.6, label=LABEL[m])
        ax.fill_between(sm.index, mu - sd, mu + sd, color=c, alpha=0.12, lw=0)
    ax.set_title(title, loc="left"); ax.set_xlabel("Training episode")
axes[0].set_ylabel("Daily loss $L_d$ (15-episode moving mean)")
axes[1].legend(fontsize=6.6, loc="upper right")
fig.tight_layout(); save(fig, "fig6_training_curves")

# training stability numbers
stab = {}
for m in cols:
    d = tr[tr.method == m]
    if d.empty:
        continue
    E = d.episode.max() + 1
    last = d[d.episode >= E - 100].groupby("training_seed").weighted_social_loss.mean()
    prev = d[(d.episode >= E - 200) & (d.episode < E - 100)].groupby("training_seed").weighted_social_loss.mean()
    stab[m] = {"last100": float(last.mean()), "prev100": float(prev.mean()),
               "rel_change_pct": float(100 * (last.mean() - prev.mean()) / abs(prev.mean()))}
json.dump(stab, open(OUT.replace(".md", "_stability.json"), "w"), indent=1)
print(open(OUT).read()); print(json.dumps(stab, indent=1))

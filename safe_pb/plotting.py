"""Figure generation for the baseline-validation report. All figures are
produced directly from computed arrays -- no values are typed in by hand."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_price_and_load(scenario, results_by_method: dict, save_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    hours = np.arange(scenario.T)
    for name, results in results_by_method.items():
        load = np.array([r.p_non.sum() + r.p_gen.sum() + r.p_pev.sum() for r in results])
        axes[0].plot(hours, load, marker="o", label=name, linewidth=1.5, markersize=3)
    axes[0].plot(hours, scenario.wholesale_price / scenario.wholesale_price.max() *
                 axes[0].get_ylim()[1], "k--", alpha=0.3, label="wholesale price (scaled)")
    axes[0].set_ylabel("Total load (kW)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Aggregate load by method (Tier A, 15-household reconstruction)")

    axes[1].plot(hours, scenario.wholesale_price, label="wholesale price $\\lambda_t$", color="black")
    axes[1].set_ylabel("$/kWh")
    axes[1].set_xlabel("Hour")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_q_convergence(agent, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    deltas = agent.max_delta_history
    ax.plot(deltas)
    ax.axhline(agent.cfg.delta, color="red", linestyle="--",
               label=f"stopping tolerance delta={agent.cfg.delta} (not reached within {len(deltas)} episodes)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("max|Q^k - Q^(k-1)| across all (state, action)")
    ax.set_yscale("log")
    # Renamed from "convergence" -- the stopping criterion (delta<=0.01)
    # was never actually met within the training budget
    # (converged_at_episode=None in the JSON output), so calling this plot
    # a "convergence" figure would overclaim; it shows the update
    # magnitude trend, not a certified converged result.
    ax.set_title("Tabular Q-learning update magnitude over training", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_voltage_profile(vm_pu: dict, v_min: float, v_max: float, save_path: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    buses = sorted(vm_pu.keys())
    vals = [vm_pu[b] for b in buses]
    colors = ["red" if (v < v_min or v > v_max) else "steelblue" for v in vals]
    ax.bar(buses, vals, color=colors)
    ax.axhline(v_min, color="black", linestyle="--", linewidth=1)
    ax.axhline(v_max, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Bus")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_linear_vs_ac(linear_v: dict, ac_v: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 7))
    buses = sorted(set(linear_v) & set(ac_v))
    lv = [linear_v[b] for b in buses]
    av = [ac_v[b] for b in buses]
    ax.scatter(av, lv, s=18)
    lims = [min(min(lv), min(av)) - 0.01, max(max(lv), max(av)) + 0.01]
    ax.plot(lims, lims, "k--", linewidth=1, label="perfect agreement")
    ax.set_xlabel("AC power-flow voltage (p.u.)")
    ax.set_ylabel("Linearized (LinDistFlow) voltage (p.u.)")
    ax.set_title("Linear approximation vs. independent AC verification")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)

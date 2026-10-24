"""Generate manuscript-ready tables and figures only from completed outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "SAFE-PB", "SAFE-PB_no_CVaR", "SAFE-PB_no_fairness",
    "SAFE-PB_no_safety", "SAFE-PB_single_price", "No-DR", "TOU",
    "Tabular_Q_learning", "MPC",
]


def _ordered(block: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    return block.assign(_order=block.method.map(order)).sort_values("_order")


def _metric_figure(summary: pd.DataFrame, metric: str, title: str,
                   ylabel: str, path: Path) -> None:
    profiles = sorted(summary.scenario_profile.unique())
    fig, axes = plt.subplots(1, len(profiles), figsize=(max(8, 6 * len(profiles)), 5),
                             squeeze=False)
    for ax, profile in zip(axes[0], profiles):
        block = _ordered(summary[summary.scenario_profile == profile])
        mean = block[f"{metric}_mean"].to_numpy(float)
        lo = block[f"{metric}_ci_low"].to_numpy(float)
        hi = block[f"{metric}_ci_high"].to_numpy(float)
        err = np.vstack([np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)])
        x = np.arange(len(block))
        ax.bar(x, mean, color=["#1f77b4" if m == "SAFE-PB" else "#9ebcda"
                               for m in block.method])
        if np.isfinite(err).any():
            ax.errorbar(x, mean, yerr=err, fmt="none", ecolor="black",
                        capsize=3, linewidth=1)
        ax.set_xticks(x, block.method, rotation=40, ha="right", fontsize=8)
        ax.set_title(profile.replace("_", " ").title())
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _learning_figure(outdir: Path, path: Path) -> None:
    frames = []
    for train_path in sorted(outdir.glob("train_*.csv")):
        frame = pd.read_csv(train_path)
        if "weighted_social_loss" not in frame:
            continue
        stem = train_path.stem.removeprefix("train_")
        method, seed_text = stem.rsplit("_seed", 1)
        frame["method"] = method
        frame["training_seed"] = int(seed_text)
        frames.append(frame[["method", "training_seed", "episode",
                             "weighted_social_loss"]])
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in METHOD_ORDER:
        block = data[data.method == method]
        if block.empty:
            continue
        pivot = block.pivot(index="episode", columns="training_seed",
                            values="weighted_social_loss").sort_index()
        smooth = pivot.rolling(20, min_periods=1).mean()
        mean = smooth.mean(axis=1)
        if smooth.shape[1] > 1:
            half = 1.96 * smooth.std(axis=1, ddof=1) / np.sqrt(smooth.shape[1])
            ax.fill_between(mean.index, mean - half, mean + half, alpha=0.12)
        ax.plot(mean.index, mean, label=method, linewidth=1.5)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Weighted social loss (20-episode moving mean)")
    ax.set_title("Training diagnostics across independent seeds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_digest(summary: pd.DataFrame, statistics: pd.DataFrame,
                  manifest: dict, path: Path) -> None:
    selected = ["weighted_social_loss", "PAR", "CVaR90",
                "voltage_violation_minutes", "normalized_discomfort_variance"]
    lines = ["# SAFE-PB experiment results digest", "",
             f"- Run status: `{manifest.get('status', 'unknown')}`",
             f"- Publication eligible: `{manifest.get('publication_eligible', False)}`",
             f"- Inferential unit: `{manifest.get('inferential_unit', 'training_seed')}`",
             f"- Independent training seeds: `{len(manifest.get('seeds', []))}`", ""]
    if not manifest.get("publication_eligible", False):
        lines += ["**Do not copy these numbers into the manuscript. This is an "
                  "engineering/pilot run, not publication evidence.**", ""]
    for profile in sorted(summary.scenario_profile.unique()):
        lines += [f"## {profile.replace('_', ' ').title()}", "",
                  "| Method | Social loss | PAR | CVaR90 | Voltage violation min | Discomfort variance |",
                  "|---|---:|---:|---:|---:|---:|"]
        for _, row in _ordered(summary[summary.scenario_profile == profile]).iterrows():
            values = [row[f"{m}_mean"] for m in selected]
            lines.append("| " + str(row.method) + " | " + " | ".join(
                f"{value:.4g}" for value in values) + " |")
        lines.append("")
    if not statistics.empty:
        key = statistics[statistics.metric.isin(selected)].copy()
        significant = key[np.isfinite(key.p_holm) & (key.p_holm < 0.05)]
        lines += ["## Multiplicity-corrected paired results", "",
                  f"Significant SAFE-PB contrasts among selected outcomes: "
                  f"{len(significant)} of {len(key)} (Holm-adjusted p < 0.05).", ""]
    path.write_text("\n".join(lines))


def generate_publication_outputs(outdir: Path) -> None:
    summary = pd.read_csv(outdir / "publication_summary.csv")
    statistics = pd.read_csv(outdir / "paired_statistics.csv")
    manifest = json.loads((outdir / "run_manifest.json").read_text())
    figures = outdir / "figures"
    figures.mkdir(exist_ok=True)
    _metric_figure(summary, "weighted_social_loss",
                   "Held-out economic performance across training seeds",
                   "Weighted social loss", figures / "fig7_1_social_loss.png")
    _metric_figure(summary, "PAR", "Held-out peak-to-average ratio",
                   "PAR", figures / "fig7_2_par.png")
    _metric_figure(summary, "voltage_violation_minutes",
                   "Independent AC voltage verification",
                   "Violation minutes per held-out day",
                   figures / "fig7_3_ac_voltage_violations.png")
    _metric_figure(summary, "CVaR90", "Held-out tail-risk comparison",
                   "CVaR90 of weighted social loss",
                   figures / "fig7_4_cvar90.png")
    _metric_figure(summary, "normalized_discomfort_variance",
                   "Normalized discomfort fairness",
                   "Variance of normalized discomfort",
                   figures / "fig7_5_fairness.png")
    _learning_figure(outdir, figures / "fig7_6_training_diagnostics.png")
    _write_digest(summary, statistics, manifest, outdir / "results_digest.md")

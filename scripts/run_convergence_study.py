"""SAFE-PB training-budget convergence study.

Runs the full SAFE-PB policy at multiple training budgets across independent
training seeds, evaluates each trained policy on the same nominal and
binding-voltage held-out protocol, and writes a compact comparison table.

This is an engineering/model-selection study only. It must be completed before
freezing the training budget for the final >=20-seed publication experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_pb.feeder_model import build_feeder
from safe_pb.data_loader import build_tier_a_scenario
from scripts.run_v17_experiments import (
    CONFIG,
    calibrate_binding_profile,
    evaluate_learned,
    train_one,
)


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def summarize_seed(budget: int, seed: int, train_log: list[dict], rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    tr = pd.DataFrame(train_log)
    nominal = df[df["scenario_profile"] == "nominal"]
    binding = df[df["scenario_profile"] == "binding_voltage"]
    tail = tr.tail(min(50, len(tr)))
    early = tr.head(min(50, len(tr)))

    def mean(frame, col):
        return float(frame[col].mean()) if len(frame) and col in frame else float("nan")

    return {
        "budget": budget,
        "training_seed": seed,
        "train_last50_loss": mean(tail, "weighted_social_loss"),
        "train_first50_loss": mean(early, "weighted_social_loss"),
        "train_last50_reward": mean(tail, "shaped_reward"),
        "train_last50_projection_active_steps": mean(tail, "projection_active_steps"),
        "nominal_loss": mean(nominal, "weighted_social_loss"),
        "nominal_PAR": mean(nominal, "PAR"),
        "nominal_CVaR90": mean(nominal, "CVaR90"),
        "nominal_fairness_variance": mean(nominal, "normalized_discomfort_variance"),
        "nominal_min_voltage_ac": mean(nominal, "min_voltage_ac"),
        "binding_loss": mean(binding, "weighted_social_loss"),
        "binding_PAR": mean(binding, "PAR"),
        "binding_CVaR90": mean(binding, "CVaR90"),
        "binding_fairness_variance": mean(binding, "normalized_discomfort_variance"),
        "binding_projection_active_rate": mean(binding, "projection_active_rate"),
        "binding_voltage_violation_minutes": mean(binding, "voltage_violation_minutes"),
        "binding_controllable_violation_minutes": mean(binding, "controllable_violation_minutes"),
        "binding_exogenous_violation_minutes": mean(binding, "exogenous_violation_minutes"),
        "binding_candidate_voltage_violation_minutes": mean(binding, "candidate_voltage_violation_minutes"),
        "binding_min_voltage_ac": mean(binding, "min_voltage_ac"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", default="300,500,750,1000")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--eval-days", type=int, default=10)
    p.add_argument("--updates-per-episode", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay-capacity", type=int, default=5000)
    p.add_argument("--beta-r", type=float, default=0.20)
    p.add_argument("--beta-f", type=float, default=2000.0)
    p.add_argument("--cvar-alpha", type=float, default=0.90)
    p.add_argument("--risk-warmup", type=int, default=40)
    p.add_argument("--output-dir", default=str(ROOT / "results" / "convergence_study"))
    args = p.parse_args()

    budgets = parse_ints(args.budgets)
    seeds = parse_ints(args.seeds)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    topo = build_feeder()
    nominal_scenario = build_tier_a_scenario(str(CONFIG), random_seed=9_000_000)
    nominal_base_scale = float(nominal_scenario.network_config["base_scale"])

    common = Namespace(
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        beta_r=args.beta_r,
        beta_f=args.beta_f,
        cvar_alpha=args.cvar_alpha,
        risk_warmup=args.risk_warmup,
        updates_per_episode=args.updates_per_episode,
        eval_days=args.eval_days,
        eval_seed_base=9_000_000,
        stress_calibration_seed=8_000_000,
        v_min=0.95,
        stress_v_min=0.95,
        stress_base_scale=0.35,
        stress_homes_candidates="2,4,8,12,16",
        save_checkpoints=False,
    )
    stress, calibration = calibrate_binding_profile(topo, common, nominal_base_scale)
    (out / "stress_calibration.json").write_text(json.dumps({
        "selected": stress, "audit": calibration
    }, indent=2))

    summaries = []
    all_eval = []
    started = time.time()
    for budget in budgets:
        for seed in seeds:
            run_args = Namespace(**vars(common))
            run_args.train_episodes = budget
            run_dir = out / f"budget_{budget}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[budget={budget}] seed={seed}: training", flush=True)
            t0 = time.time()
            trainer, train_log = train_one(seed, "SAFE-PB", run_args, topo, run_dir, stress)
            print(f"[budget={budget}] seed={seed}: held-out evaluation", flush=True)
            rows = evaluate_learned(
                trainer, seed, "SAFE-PB", run_args, topo, stress,
                ["nominal", "binding_voltage"],
            )
            elapsed = time.time() - t0
            for r in rows:
                r["budget"] = budget
            all_eval.extend(rows)
            s = summarize_seed(budget, seed, train_log, rows)
            s["elapsed_seconds"] = elapsed
            summaries.append(s)
            pd.DataFrame(rows).to_csv(run_dir / "evaluation.csv", index=False)
            print(f"[budget={budget}] seed={seed}: {elapsed/60:.1f} min", flush=True)

    seed_df = pd.DataFrame(summaries)
    seed_df.to_csv(out / "convergence_by_seed.csv", index=False)
    pd.DataFrame(all_eval).to_csv(out / "all_evaluation_rows.csv", index=False)

    numeric_cols = [c for c in seed_df.columns if c not in {"budget", "training_seed"}]
    grouped = seed_df.groupby("budget")[numeric_cols].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    grouped.reset_index().to_csv(out / "convergence_summary.csv", index=False)

    # Relative change between successive budgets for the key held-out metrics.
    key = ["nominal_loss", "nominal_PAR", "nominal_CVaR90",
           "nominal_fairness_variance", "binding_loss",
           "binding_projection_active_rate", "binding_voltage_violation_minutes"]
    means = seed_df.groupby("budget")[key].mean().sort_index()
    rel = means.pct_change().abs().replace([np.inf, -np.inf], np.nan)
    rel.columns = [f"abs_relative_change_{c}" for c in rel.columns]
    rel.reset_index().to_csv(out / "budget_relative_changes.csv", index=False)

    manifest = {
        "status": "passed",
        "purpose": "training-budget model selection; not publication evidence",
        "budgets": budgets,
        "seeds": seeds,
        "eval_days": args.eval_days,
        "elapsed_seconds": time.time() - started,
        "stress_profile": stress,
        "recommendation_rule": (
            "Choose the smallest budget for which held-out loss/CVaR/PAR/fairness "
            "changes are small relative to the next larger budget and training "
            "curves are stable across all three seeds; then freeze that budget "
            "before the final >=20-seed publication run."
        ),
    }
    (out / "convergence_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("PASS. Outputs written to", out)


if __name__ == "__main__":
    main()

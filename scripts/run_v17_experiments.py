#!/usr/bin/env python
"""Publication experiment runner for SAFE-PB.

The protocol uses independent training seeds as the inferential unit, evaluates
every method on matched held-out scenarios, includes a calibrated binding-
voltage profile, and records candidate and applied schedules separately.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
from pathlib import Path

# Parallelism is at the independent job level. Large nested BLAS/OpenMP pools
# make these small networks slower and can oversubscribe desktop CPUs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_pb.baseline_evaluation import summarize_baseline_episode
from safe_pb.baselines import (QLearningConfig, TabularQLearningAgent,
                               run_no_dr, run_price_schedule,
                               run_rolling_horizon_mpc, run_tou)
from safe_pb.data_loader import build_tier_a_scenario
from safe_pb.environment import EnvConfig, SafePBEnv
from safe_pb.episode_metrics import aggregate_episodes
from safe_pb.feeder_model import build_feeder
from safe_pb.metrics import cvar
from safe_pb.publication_reporting import generate_publication_outputs
from safe_pb.sac_trainer import HierarchicalSACTrainer, SACConfig
from safe_pb.statistical_analysis import (bootstrap_ci, cohens_dz,
                                          holm_correction, paired_test)

CONFIG = ROOT / "config" / "tier_a_wan2022.yaml"

LEARNED_METHODS = {
    "SAFE-PB": dict(risk=True, fair=True, safety=True, single=False),
    "SAFE-PB_no_CVaR": dict(risk=False, fair=True, safety=True, single=False),
    "SAFE-PB_no_fairness": dict(risk=True, fair=False, safety=True, single=False),
    "SAFE-PB_no_safety": dict(risk=True, fair=True, safety=False, single=False),
    "SAFE-PB_single_price": dict(risk=True, fair=True, safety=True, single=True),
}
BASELINE_METHODS = ("No-DR", "TOU", "Tabular_Q_learning", "MPC")
ALL_METHODS = tuple(LEARNED_METHODS) + BASELINE_METHODS

STAT_METRICS = (
    "weighted_social_loss", "PAR", "CVaR90",
    "candidate_voltage_violation_minutes", "voltage_violation_minutes",
    "controllable_violation_minutes", "exogenous_violation_minutes",
    "projection_active_rate", "normalized_discomfort_variance",
    "jain_discomfort", "gini_discomfort",
)

SEED_MEAN_METRICS = tuple(dict.fromkeys((
    "weighted_social_loss", "utility_profit", "customer_cost", "peak_kw",
    "mean_kw", "PAR", "voltage_violation_minutes", "min_voltage_ac",
    "controllable_violation_minutes", "exogenous_violation_minutes",
    "authority_infeasible_steps", "candidate_violation_restored_rate",
    "candidate_voltage_violation_minutes", "candidate_min_voltage_ac",
    "projection_active_steps", "projection_active_rate",
    "projection_distance_sum", "projection_delta_l1_sum", "slack_sum",
    "normalized_discomfort_variance", "jain_discomfort", "gini_discomfort",
)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_seeds(text: str) -> list[int]:
    if ":" in text:
        start, stop = map(int, text.split(":"))
        return list(range(start, stop))
    return [int(item) for item in text.split(",") if item.strip()]


def profile_config(profile: str, stress: dict) -> dict:
    if profile == "nominal":
        return {"base_scale": stress["nominal_base_scale"],
                "homes_per_bus": stress["nominal_homes_per_bus"],
                "v_min": stress["nominal_v_min"]}
    if profile == "binding_voltage":
        return {"base_scale": stress["base_scale"],
                "homes_per_bus": stress["homes_per_bus"],
                "v_min": stress["v_min"]}
    raise ValueError(f"unknown scenario profile: {profile}")


def make_env(topo, scenario_seed: int, variant: dict, profile: str, stress: dict,
             verify_candidate_ac: bool = True):
    scenario = build_tier_a_scenario(str(CONFIG), random_seed=scenario_seed)
    cfg = profile_config(profile, stress)
    env_cfg = EnvConfig(
        enable_safety_projection=variant["safety"],
        single_price=variant["single"],
        base_scale=cfg["base_scale"], homes_per_bus=cfg["homes_per_bus"],
        v_min=cfg["v_min"], scenario_label=profile,
        verify_candidate_ac=verify_candidate_ac,
        # Authority classification is an evaluation diagnostic; it does not
        # affect transitions or rewards, so it is disabled during training.
        classify_authority=verify_candidate_ac,
    )
    return SafePBEnv(scenario, topo, env_cfg)


def calibrate_binding_profile(topo, args, nominal_base_scale: float) -> tuple[dict, list[dict]]:
    """Select the smallest declared multiplier that activates projection."""
    candidates = [float(x) for x in args.stress_homes_candidates.split(",")]
    audit = []
    safe_variant = LEARNED_METHODS["SAFE-PB"]
    for homes in candidates:
        stress = {
            "nominal_base_scale": nominal_base_scale,
            "nominal_homes_per_bus": 1.0,
            "nominal_v_min": args.v_min,
            "base_scale": args.stress_base_scale,
            "homes_per_bus": homes,
            "v_min": args.stress_v_min,
        }
        env = make_env(topo, args.stress_calibration_seed, safe_variant,
                       "binding_voltage", stress)
        env.reset()
        active_steps = 0
        candidate_violations = 0
        min_candidate_ac = float("inf")
        done = False
        while not done:
            env.utility_announce_price(np.zeros(env.cfg.n_utility_act))
            _, _, _, done, info = env.household_step(
                np.ones((env.n_households, env.cfg.n_household_act)))
            active_steps += int(info["projection_active"])
            candidate_violations += int(info["candidate_ac_safety_violated"])
            min_candidate_ac = min(min_candidate_ac, info["candidate_min_voltage_ac"])
        record = {"homes_per_bus": homes, "active_steps": active_steps,
                  "candidate_violation_steps": candidate_violations,
                  "min_candidate_voltage_ac": min_candidate_ac}
        audit.append(record)
        if active_steps > 0 and candidate_violations > 0:
            return stress, audit
    raise RuntimeError(
        "No declared stress multiplier activated the safety projection. "
        "Increase --stress-homes-candidates or --stress-v-min; calibration audit="
        + json.dumps(audit))


def train_one(seed: int, method: str, args, topo, outdir: Path, stress: dict):
    variant = LEARNED_METHODS[method]
    env = make_env(topo, seed * 1_000_000, variant, "nominal", stress)
    cfg = SACConfig(
        seed=seed, batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        beta_r=args.beta_r if variant["risk"] else 0.0,
        beta_f=args.beta_f if variant["fair"] else 0.0,
        cvar_alpha=args.cvar_alpha,
        risk_warmup_episodes=args.risk_warmup,
    )
    trainer = HierarchicalSACTrainer(env, cfg)
    train_log = []
    for episode in range(args.train_episodes):
        profile = "binding_voltage" if episode % 2 else "nominal"
        scenario_seed = seed * 1_000_000 + episode
        trainer.env = make_env(
            topo, scenario_seed, variant, profile, stress,
            verify_candidate_ac=False,
        )
        shaped_reward, _ = trainer.collect_episode(deterministic=False)
        updates = []
        if len(trainer.buffer) >= cfg.batch_size:
            for _ in range(args.updates_per_episode):
                updates.append(trainer.train_step())
        metrics = dict(trainer.last_episode_metrics)
        train_log.append({
            "method": method, "training_seed": seed,
            "episode": episode, "scenario_seed": scenario_seed,
            "scenario_profile": profile, "shaped_reward": shaped_reward,
            "weighted_social_loss": metrics["weighted_social_loss"],
            "cvar_zeta": metrics["cvar_zeta"],
            "risk_objective_component": metrics["risk_objective_component"],
            "fairness_objective_component": metrics["fairness_objective_component"],
            "risk_component_fraction_of_daily_loss":
                metrics["risk_component_fraction_of_daily_loss"],
            "fairness_component_fraction_of_daily_loss":
                metrics["fairness_component_fraction_of_daily_loss"],
            "fairness_variance": metrics["normalized_discomfort_variance"],
            "projection_active_steps": metrics["projection_active_steps"],
            "n_updates": len(updates),
        })
    pd.DataFrame(train_log).to_csv(
        outdir / f"train_{method}_seed{seed}.csv", index=False)
    if args.save_checkpoints:
        trainer.save_checkpoint(str(outdir / f"checkpoint_{method}_seed{seed}.pt"))
    return trainer, train_log


def evaluate_learned(trainer, seed: int, method: str, args, topo,
                     stress: dict, profiles: list[str]) -> list[dict]:
    variant = LEARNED_METHODS[method]
    rows = []
    for profile in profiles:
        for day in range(args.eval_days):
            scenario_seed = args.eval_seed_base + seed * 10_000 + day
            trainer.env = make_env(topo, scenario_seed, variant, profile, stress)
            metrics, _ = trainer.evaluate_episode(deterministic=True)
            metrics.update({"method": method, "training_seed": seed,
                            "eval_day": day, "scenario_seed": scenario_seed,
                            "scenario_profile": profile,
                            "inferential_unit": "training_seed"})
            rows.append(metrics)
    return rows


def train_tabular(seed: int, args):
    scenario = build_tier_a_scenario(str(CONFIG), random_seed=seed * 1_000_000)
    cfg = QLearningConfig(grid_points=args.q_grid_points,
                          n_episodes=args.q_train_episodes,
                          epsilon_decay_episodes=min(args.q_train_episodes, 3000),
                          seed=seed, respect_ev_availability=True)
    agent = TabularQLearningAgent(scenario, cfg)
    agent.train()
    return agent.greedy_price_schedule()


def evaluate_baseline(seed: int, method: str, args, topo,
                      stress: dict, profiles: list[str]) -> list[dict]:
    tabular_prices = train_tabular(seed, args) if method == "Tabular_Q_learning" else None
    rows = []
    for profile in profiles:
        cfg = profile_config(profile, stress)
        for day in range(args.eval_days):
            scenario_seed = args.eval_seed_base + seed * 10_000 + day
            scenario = build_tier_a_scenario(str(CONFIG), random_seed=scenario_seed)
            prices = None
            if method == "No-DR":
                results = run_no_dr(scenario, respect_ev_availability=True)
            elif method == "TOU":
                results = run_tou(scenario, respect_ev_availability=True)
                prices = []
                for t in range(scenario.T):
                    multiplier = 1.6 if t in range(17, 22) else 1.05
                    eta = float(np.clip(scenario.wholesale_price[t] * multiplier,
                                        scenario.eta_bounds[0][t], scenario.eta_bounds[1][t]))
                    prices.append((eta, eta, eta))
            elif method == "Tabular_Q_learning":
                prices = tabular_prices
                results = run_price_schedule(
                    scenario, prices, respect_ev_availability=True)
            elif method == "MPC":
                results, prices = run_rolling_horizon_mpc(
                    scenario, horizon=args.mpc_horizon,
                    grid_points=args.mpc_grid_points,
                    beam_width=args.mpc_beam_width,
                    respect_ev_availability=True)
            else:
                raise ValueError(method)
            metrics = summarize_baseline_episode(
                scenario, topo, results, prices, profile,
                base_scale=cfg["base_scale"], homes_per_bus=cfg["homes_per_bus"],
                v_min=cfg["v_min"])
            metrics.update({"method": method, "training_seed": seed,
                            "eval_day": day, "scenario_seed": scenario_seed,
                            "scenario_profile": profile,
                            "inferential_unit": "training_seed"})
            rows.append(metrics)
    return rows


def per_seed_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (profile, method, seed), group in df.groupby(
            ["scenario_profile", "method", "training_seed"]):
        row = {"scenario_profile": profile, "method": method,
               "training_seed": seed, "n_eval_days": len(group),
               "inferential_unit": "training_seed"}
        for metric in SEED_MEAN_METRICS:
            row[metric] = float(group[metric].mean())
        row["CVaR90"] = cvar(group["weighted_social_loss"].to_numpy(), 0.90)
        row["CVaR95"] = cvar(group["weighted_social_loss"].to_numpy(), 0.95)
        records.append(row)
    return pd.DataFrame(records)


def validate_design(df: pd.DataFrame, methods: list[str], seeds: list[int],
                    profiles: list[str], eval_days: int) -> dict:
    """Fail closed on incomplete, duplicated, mismatched, or invalid output."""
    keys = ["method", "training_seed", "scenario_profile", "eval_day"]
    expected_rows = len(methods) * len(seeds) * len(profiles) * eval_days
    duplicated = int(df.duplicated(keys).sum())
    if len(df) != expected_rows or duplicated:
        raise RuntimeError(
            f"incomplete/duplicated evaluation grid: rows={len(df)}, "
            f"expected={expected_rows}, duplicates={duplicated}")
    expected_methods = set(methods)
    for (seed, profile, day), group in df.groupby(
            ["training_seed", "scenario_profile", "eval_day"]):
        if set(group.method) != expected_methods:
            raise RuntimeError(
                f"unmatched method block at seed={seed}, profile={profile}, day={day}")
        if group.scenario_seed.nunique() != 1:
            raise RuntimeError(
                f"methods do not share a scenario seed at seed={seed}, "
                f"profile={profile}, day={day}")
    required_numeric = [m for m in STAT_METRICS if m != "CVaR90"]
    bad = {column: int((~np.isfinite(df[column].to_numpy(float))).sum())
           for column in required_numeric}
    bad = {column: count for column, count in bad.items() if count}
    if bad:
        raise RuntimeError(f"non-finite publication metrics: {bad}")
    return {"expected_rows": expected_rows, "actual_rows": len(df),
            "duplicate_key_rows": duplicated, "matched_scenarios": True,
            "finite_required_metrics": True}


def build_statistics(df: pd.DataFrame, outdir: Path, min_seeds: int,
                     allow_fewer: bool) -> None:
    per_seed = per_seed_table(df)
    per_seed.to_csv(outdir / "per_seed_means.csv", index=False)
    rows = []
    for profile in sorted(per_seed["scenario_profile"].unique()):
        block = per_seed[per_seed.scenario_profile == profile]
        reference = block[block.method == "SAFE-PB"].set_index("training_seed")
        for comparison in sorted(set(block.method) - {"SAFE-PB"}):
            alternative = block[block.method == comparison].set_index("training_seed")
            common = reference.index.intersection(alternative.index)
            if len(common) < min_seeds and not allow_fewer:
                raise RuntimeError(
                    f"{profile}/{comparison}: only {len(common)} paired training seeds; "
                    f"publication protocol requires at least {min_seeds}")
            for metric in STAT_METRICS:
                x = reference.loc[common, metric].to_numpy(float)
                y = alternative.loc[common, metric].to_numpy(float)
                finite = np.isfinite(x) & np.isfinite(y)
                x, y = x[finite], y[finite]
                if len(x) < min_seeds and not allow_fewer:
                    raise RuntimeError(
                        f"{profile}/{comparison}/{metric}: only {len(x)} finite paired "
                        f"training seeds; publication protocol requires {min_seeds}")
                test = paired_test(x, y)
                ci = bootstrap_ci(x - y, seed=2026)
                rows.append({
                    "scenario_profile": profile, "reference": "SAFE-PB",
                    "comparison": comparison, "metric": metric,
                    "inferential_unit": "training_seed", "n_pairs": len(x),
                    "mean_paired_difference": float(np.mean(x - y)),
                    "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                    "test": test["test"], "p_value": test["p_value"],
                    "effect_size_dz": cohens_dz(x, y),
                })
    stats = pd.DataFrame(rows)
    stats["p_holm"] = np.nan
    for (_profile, _metric), indices in stats.groupby(
            ["scenario_profile", "metric"]).groups.items():
        valid = [i for i in indices if np.isfinite(stats.at[i, "p_value"])]
        if valid:
            adjusted = holm_correction([stats.at[i, "p_value"] for i in valid])
            for index, value in zip(valid, adjusted):
                stats.at[index, "p_holm"] = value
    stats.to_csv(outdir / "paired_statistics.csv", index=False)


def build_summary(df: pd.DataFrame, outdir: Path) -> None:
    """Summarize across independent training seeds, never pooled test days."""
    per_seed = per_seed_table(df)
    rows = []
    for (profile, method), group in per_seed.groupby(
            ["scenario_profile", "method"]):
        summary = {
            "scenario_profile": profile, "method": method,
            "n_training_seeds": int(len(group)),
            "n_evaluation_episodes": int(group["n_eval_days"].sum()),
            "inferential_unit": "training_seed",
        }
        for metric in SEED_MEAN_METRICS + ("CVaR90", "CVaR95"):
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            ci = bootstrap_ci(values, seed=2026)
            summary[f"{metric}_mean"] = (float(values.mean())
                                                  if len(values) else float("nan"))
            summary[f"{metric}_std"] = (float(values.std(ddof=1))
                                                 if len(values) > 1 else 0.0)
            summary[f"{metric}_median"] = (float(np.median(values))
                                                    if len(values) else float("nan"))
            summary[f"{metric}_ci_low"] = ci["ci_low"]
            summary[f"{metric}_ci_high"] = ci["ci_high"]
        rows.append(summary)
    pd.DataFrame(rows).to_csv(outdir / "publication_summary.csv", index=False)


def objective_diagnostics(train_logs: dict, args) -> dict:
    logs = pd.DataFrame(train_logs["SAFE-PB"])
    after_warmup = logs[logs.episode >= args.risk_warmup]
    result = {
        "beta_r": args.beta_r, "beta_f": args.beta_f,
        "risk_positive_episode_rate": float(
            (after_warmup.risk_objective_component > 0).mean()),
        "mean_risk_component_fraction": float(
            after_warmup.risk_component_fraction_of_daily_loss.mean()),
        "median_fairness_component_fraction": float(
            logs.fairness_component_fraction_of_daily_loss.median()),
        "minimum_required_fraction": args.objective_min_fraction,
    }
    result["risk_signal_verified"] = bool(
        result["risk_positive_episode_rate"] > 0 and
        result["mean_risk_component_fraction"] >= args.objective_min_fraction)
    result["fairness_signal_verified"] = bool(
        result["median_fairness_component_fraction"] >= args.objective_min_fraction)
    return result


def _worker_init(torch_threads: int) -> None:
    """Bound nested numerical threads in each independent process."""
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _run_job(method: str, seed: int, args, stress: dict,
             profiles: list[str], outdir_text: str) -> dict:
    """Execute one independent method/seed unit in a spawned process."""
    started = time.time()
    topo = build_feeder()
    outdir = Path(outdir_text)
    if method in LEARNED_METHODS:
        trainer, log = train_one(seed, method, args, topo, outdir, stress)
        rows = evaluate_learned(trainer, seed, method, args, topo, stress, profiles)
    else:
        log = []
        rows = evaluate_baseline(seed, method, args, topo, stress, profiles)
    return {
        "method": method, "seed": seed,
        "seconds": time.time() - started,
        "rows": rows,
        "train_log": log if method == "SAFE-PB" else [],
    }


def _safe_job_name(method: str, seed: int) -> str:
    return f"{method.replace('-', '_')}_seed{seed}"


def _cache_paths(outdir: Path, method: str, seed: int) -> tuple[Path, Path]:
    cache_dir = outdir / "job_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_job_name(method, seed)
    return cache_dir / f"{stem}.json", cache_dir / f"{stem}.meta.json"


def _atomic_write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=True))
    os.replace(temp, path)


def _save_job_cache(outdir: Path, result: dict, fingerprint: str) -> None:
    data_path, meta_path = _cache_paths(
        outdir, result["method"], int(result["seed"]))
    _atomic_write_json(data_path, result["rows"])
    _atomic_write_json(meta_path, {
        "status": "passed", "method": result["method"],
        "seed": int(result["seed"]), "seconds": float(result["seconds"]),
        "protocol_fingerprint": fingerprint,
    })


def _load_job_cache(outdir: Path, method: str, seed: int,
                    fingerprint: str) -> dict | None:
    data_path, meta_path = _cache_paths(outdir, method, seed)
    if not (data_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("protocol_fingerprint") != fingerprint:
        raise RuntimeError(
            f"cached job {method}/seed={seed} used different scientific "
            "settings; select a new --output-dir or pass --no-resume")
    if meta.get("status") != "passed":
        return None
    train_log = []
    if method == "SAFE-PB":
        train_path = outdir / f"train_{method}_seed{seed}.csv"
        if not train_path.exists():
            return None
        train_log = pd.read_csv(train_path).to_dict("records")
    return {
        "method": method, "seed": seed,
        "seconds": float(meta.get("seconds", 0.0)),
        "rows": json.loads(data_path.read_text()),
        "train_log": train_log, "resumed": True,
    }


def _protocol_fingerprint(args, stress: dict) -> str:
    ignored = {"workers", "torch_threads", "resume", "output_dir",
               "save_checkpoints"}
    scientific_args = {k: v for k, v in vars(args).items() if k not in ignored}
    payload = {
        "args": scientific_args, "stress": stress,
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "environment_sha256": sha256(ROOT / "safe_pb" / "environment.py"),
        "feeder_sha256": sha256(ROOT / "safe_pb" / "feeder_model.py"),
        "episode_metrics_sha256": sha256(ROOT / "safe_pb" / "episode_metrics.py"),
        "trainer_sha256": sha256(ROOT / "safe_pb" / "sac_trainer.py"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0:20",
                        help="independent training seeds; publication default is 0:20")
    parser.add_argument("--min-publication-seeds", type=int, default=20)
    parser.add_argument("--allow-fewer-seeds", action="store_true",
                        help="smoke tests only; manifest is marked non-publication")
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--updates-per-episode", type=int, default=8)
    parser.add_argument("--eval-days", type=int, default=20)
    parser.add_argument("--eval-seed-base", type=int, default=9_000_000)
    parser.add_argument("--stress-calibration-seed", type=int, default=8_000_000,
                        help="separate from all training and held-out evaluation seeds")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--beta-r", type=float, default=0.20)
    parser.add_argument("--beta-f", type=float, default=2000.0,
                        help="fairness variance weight calibrated to daily-loss units")
    parser.add_argument("--cvar-alpha", type=float, default=0.90)
    parser.add_argument("--risk-warmup", type=int, default=40)
    parser.add_argument("--objective-min-fraction", type=float, default=0.005)
    parser.add_argument("--scenario-profiles", default="nominal,binding_voltage")
    parser.add_argument("--v-min", type=float, default=0.95)
    parser.add_argument("--stress-v-min", type=float, default=0.95)
    parser.add_argument("--stress-base-scale", type=float, default=0.35)
    parser.add_argument("--stress-homes-candidates", default="2,4,8,12,16")
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--q-train-episodes", type=int, default=4000)
    parser.add_argument("--q-grid-points", type=int, default=7)
    parser.add_argument("--mpc-horizon", type=int, default=3)
    parser.add_argument("--mpc-grid-points", type=int, default=3)
    parser.add_argument("--mpc-beam-width", type=int, default=8)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="parallel method/seed processes; 0 chooses up to 4 automatically")
    parser.add_argument(
        "--torch-threads", type=int, default=1,
        help="PyTorch threads inside each worker")
    parser.add_argument(
        "--no-resume", action="store_false", dest="resume",
        help="rerun jobs instead of reusing matching completed job caches")
    parser.set_defaults(resume=True)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "publication_experiments"))
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    profiles = [x.strip() for x in args.scenario_profiles.split(",") if x.strip()]
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")
    if "SAFE-PB" not in methods:
        raise SystemExit("SAFE-PB must be included as the statistical reference")
    if len(methods) < 2:
        raise SystemExit("at least one comparison method must be included")
    if not args.allow_fewer_seeds and set(methods) != set(ALL_METHODS):
        raise SystemExit(
            "publication runs must include every declared SAFE-PB variant and "
            "baseline under the matched design; use --allow-fewer-seeds only "
            "for a reduced engineering run")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("training seeds must be unique")
    if len(seeds) < args.min_publication_seeds and not args.allow_fewer_seeds:
        raise SystemExit(
            f"publication run requires at least {args.min_publication_seeds} independent "
            f"training seeds; received {len(seeds)}")
    if set(profiles) != {"nominal", "binding_voltage"}:
        raise SystemExit("publication run requires nominal,binding_voltage profiles")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running", "publication_eligible": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args), "seeds": seeds, "methods": methods,
        "scenario_profiles": profiles, "inferential_unit": "training_seed",
        "config_sha256": sha256(CONFIG), "python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__,
        "numpy": np.__version__, "runs": [],
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    all_rows = []
    train_logs = {}
    try:
        topo = build_feeder()
        nominal_scenario = build_tier_a_scenario(str(CONFIG), random_seed=args.eval_seed_base)
        nominal_base_scale = float(nominal_scenario.network_config["base_scale"])
        stress, calibration = calibrate_binding_profile(topo, args, nominal_base_scale)
        manifest["stress_profile"] = stress
        manifest["stress_calibration_audit"] = calibration

        fingerprint = _protocol_fingerprint(args, stress)
        manifest["protocol_fingerprint"] = fingerprint
        jobs = [(method, seed) for method in methods for seed in seeds]
        workers = (min(4, len(jobs), os.cpu_count() or 1)
                   if args.workers == 0 else args.workers)
        if workers < 1:
            raise ValueError("--workers must be >= 1, or 0 for automatic")
        if args.torch_threads < 1:
            raise ValueError("--torch-threads must be >= 1")
        manifest["workers"] = workers
        manifest["torch_threads_per_worker"] = args.torch_threads

        pending = []
        for method, seed in jobs:
            cached = (_load_job_cache(outdir, method, seed, fingerprint)
                      if args.resume else None)
            if cached is None:
                pending.append((method, seed))
                continue
            all_rows.extend(cached["rows"])
            if method == "SAFE-PB":
                train_logs.setdefault(method, []).extend(cached["train_log"])
            manifest["runs"].append({
                "method": method, "seed": seed,
                "seconds": cached["seconds"], "status": "passed",
                "resumed": True,
            })

        completed = len(jobs) - len(pending)
        print(f"Experiment jobs: {len(jobs)} total, {completed} resumed, "
              f"{len(pending)} pending, {workers} workers", flush=True)
        if pending:
            context = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers, mp_context=context,
                    initializer=_worker_init,
                    initargs=(args.torch_threads,)) as pool:
                future_to_job = {
                    pool.submit(_run_job, method, seed, args, stress, profiles,
                                str(outdir.resolve())): (method, seed)
                    for method, seed in pending
                }
                for future in concurrent.futures.as_completed(future_to_job):
                    method, seed = future_to_job[future]
                    result = future.result()
                    _save_job_cache(outdir, result, fingerprint)
                    all_rows.extend(result["rows"])
                    if method == "SAFE-PB":
                        train_logs.setdefault(method, []).extend(result["train_log"])
                    manifest["runs"].append({
                        "method": method, "seed": seed,
                        "seconds": result["seconds"], "status": "passed",
                        "resumed": False,
                    })
                    completed += 1
                    print(f"[{completed}/{len(jobs)}] {method} seed={seed} "
                          f"completed in {result['seconds']:.1f}s", flush=True)
                    manifest["runs"].sort(key=lambda x: (
                        methods.index(x["method"]), seeds.index(x["seed"])))
                    (outdir / "run_manifest.json").write_text(
                        json.dumps(manifest, indent=2))
                    pd.DataFrame(manifest["runs"]).to_csv(
                        outdir / "progress.csv", index=False)

        frame = pd.DataFrame(all_rows).sort_values(
            ["method", "training_seed", "scenario_profile", "eval_day"]
        ).reset_index(drop=True)
        manifest["design_validation"] = validate_design(
            frame, methods, seeds, profiles, args.eval_days)
        frame.to_csv(outdir / "episode_results.csv", index=False)
        build_summary(frame, outdir)
        build_statistics(frame, outdir, args.min_publication_seeds,
                         args.allow_fewer_seeds)

        binding = frame[(frame.method == "SAFE-PB") &
                        (frame.scenario_profile == "binding_voltage")]
        activation_verified = bool(binding.projection_active_steps.sum() > 0 and
                                   binding.candidate_voltage_violation_minutes.sum() > 0)
        objective = objective_diagnostics(train_logs, args)
        (outdir / "objective_diagnostics.json").write_text(
            json.dumps(objective, indent=2))
        manifest["safety_activation_verified"] = activation_verified
        manifest["objective_diagnostics"] = objective
        if not activation_verified:
            raise RuntimeError("binding profile did not activate SAFE-PB projection")
        manifest["objective_validation_required"] = not args.allow_fewer_seeds
        if (not args.allow_fewer_seeds and
                not (objective["risk_signal_verified"] and
                     objective["fairness_signal_verified"])):
            raise RuntimeError(
                "CVaR/fairness objective contribution is below the declared minimum; "
                "inspect objective_diagnostics.json before changing weights")
        manifest["status"] = "passed"
        manifest["publication_eligible"] = bool(
            len(seeds) >= args.min_publication_seeds and not args.allow_fewer_seeds)
        manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        generate_publication_outputs(outdir)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        manifest["traceback"] = traceback.format_exc()
        raise
    finally:
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("PASS. Outputs written to", outdir)


if __name__ == "__main__":
    main()

"""
Short SAC/CTDE training smoke demo (Command 10, second implementation pass).

CORRECTIONS THIS ROUND:
  - Uses the new two-stage environment API (utility_announce_price then
    household_step via trainer.collect_episode, which was updated
    internally) so households genuinely observe the current price.
  - train_step() no longer takes update_utility -- the trainer's internal
    alternating_k schedule decides.
  - Runs THREE short training passes: default (beta_r=0, beta_f=0, as
    before), and two more with beta_r>0 and beta_f>0 respectively, so the
    CVaR and fairness code paths are actually exercised here, not just in
    pytest -- an earlier version's smoke report only ever ran with both
    disabled, silently never demonstrating (or catching the crash in) the
    beta_r>0 path.
  - Reports BOTH linear and independently AC-verified voltage from the
    evaluation rollout, and no longer embeds an absolute, machine-specific
    checkpoint path in the JSON (relative path only).

Still NOT a training-quality claim -- nothing here belongs in the
manuscript's results tables.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safe_pb.data_loader import build_tier_a_scenario
from safe_pb.feeder_model import build_feeder
from safe_pb.environment import SafePBEnv
from safe_pb.sac_trainer import HierarchicalSACTrainer, SACConfig

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "tier_a_wan2022.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_EPISODES = 20


def run_pass(scenario, topo, cfg: SACConfig, label: str):
    env = SafePBEnv(scenario, topo)
    trainer = HierarchicalSACTrainer(env, cfg)
    t0 = time.time()
    episode_rewards, loss_history = [], []
    for ep in range(N_EPISODES):
        reward, _ = trainer.collect_episode()
        episode_rewards.append(reward)
        if len(trainer.buffer) >= cfg.batch_size:
            losses = trainer.train_step()  # internal alternating_k schedule decides which actor
            loss_history.append({"episode": ep, **losses})
    elapsed = time.time() - t0
    print(f"  [{label}] {N_EPISODES} episodes in {elapsed:.1f}s, "
          f"final buffer size={len(trainer.buffer)}, "
          f"last losses={loss_history[-1] if loss_history else None}")

    ckpt_path_rel = os.path.join("results", f"sac_smoke_checkpoint_{label}.pt")
    ckpt_path_abs = os.path.join(RESULTS_DIR, f"sac_smoke_checkpoint_{label}.pt")
    trainer.save_checkpoint(ckpt_path_abs)
    step_count_before = trainer.train_step_count
    trainer.load_checkpoint(ckpt_path_abs)

    eval_reward, eval_infos = trainer.collect_episode(deterministic=True)
    min_v_linear = min(info["min_voltage_linear"] for info in eval_infos)
    min_v_ac = min(info["min_voltage_ac"] for info in eval_infos)
    ac_all_converged = all(info["ac_converged"] for info in eval_infos)
    proj_distances = [info["projection_distance"] for info in eval_infos]
    final_info = eval_infos[-1]
    terminal_soc = final_info.get("terminal_pev_soc", final_info["pev_soc"])
    terminal_shortfall = final_info.get(
        "terminal_pev_shortfall_kwh",
        [max(ev.ET - terminal_soc[j], 0.0) for j, ev in enumerate(scenario.pevs)],
    )

    return {
        "label": label, "n_episodes": N_EPISODES, "training_seconds": elapsed,
        "episode_rewards": episode_rewards, "loss_history": loss_history,
        "checkpoint_path_relative": ckpt_path_rel,  # no absolute, machine-specific path in the JSON
        "checkpoint_step_count_before_reload": step_count_before,
        "checkpoint_step_count_after_reload": trainer.train_step_count,
        "eval_rollout": {
            "reward": eval_reward,
            "base_scale": env.cfg.base_scale,
            "episode_steps": len(eval_infos),
            "min_voltage_linear": min_v_linear, "min_voltage_ac": min_v_ac,
            "ac_all_hours_converged": ac_all_converged,
            "ac_voltage_violation_steps": sum(info["ac_safety_violated"] for info in eval_infos),
            "mean_projection_distance_kw": sum(proj_distances) / len(proj_distances),
            "max_projection_distance_kw": max(proj_distances),
            "max_total_slack": max(info["total_slack"] for info in eval_infos),
            "terminal_pev_soc_kwh": list(map(float, terminal_soc)),
            "terminal_pev_shortfall_kwh": list(map(float, terminal_shortfall)),
            "pev_infeasibility_event_count": len(final_info["pev_infeasibility_events_so_far"]),
            "terminated_for_pev_infeasibility": bool(
                final_info["episode_terminated_for_pev_infeasibility"]
            ),
            "max_abs_pev_energy_balance_residual_kwh": max(
                float(max(abs(x) for x in info["pev_energy_balance_residual_kwh"]))
                for info in eval_infos
            ),
        },
    }


def main():
    scenario = build_tier_a_scenario(CONFIG_PATH)
    topo = build_feeder()

    print("STEP 1: default pass (beta_r=0, beta_f=0)")
    default_result = run_pass(scenario, topo, SACConfig(batch_size=32, seed=0), "default")

    print("STEP 2: beta_r>0 pass (exercises complete-day episodic CVaR shaping)")
    cvar_result = run_pass(scenario, topo, SACConfig(batch_size=32, seed=0, beta_r=1.0), "beta_r_positive")

    print("STEP 3: beta_f>0 pass (exercises normalized-discomfort fairness shaping)")
    fair_result = run_pass(scenario, topo, SACConfig(batch_size=32, seed=0, beta_f=1.0), "beta_f_positive")

    report = {
        "passes": [default_result, cvar_result, fair_result],
        "scope_note": ("This is a mechanics smoke demo (20 episodes x 3 configurations), not a "
                        "training-quality claim -- 20 episodes is nowhere near enough for a "
                        "convergence or ablation claim, and no number here belongs in the "
                        "manuscript's results tables. That said, beta_r>0 and beta_f>0 DO exercise "
                        "the real objectives, not proxies: beta_r applies a Rockafellar-Uryasev "
                        "complete-day CVaR penalty computed from the actual executed trajectory's "
                        "weighted social loss, and beta_f applies the real variance of Eq. 8's "
                        "normalized daily household discomfort (using each household's true "
                        "device-bound D_max). See safe_pb/sac_trainer.py's module docstring and "
                        "tests/test_sac_training.py's test_beta_r_positive_uses_complete_day_cvar_state "
                        "/ test_beta_f_positive_uses_normalized_daily_discomfort. Physical "
                        "feasibility is assessed from the independent AC fields and explicit "
                        "slack/violation counters."),
    }
    with open(os.path.join(RESULTS_DIR, "sac_smoke_demo_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nWritten to results/sac_smoke_demo_report.json")
    return report


if __name__ == "__main__":
    main()

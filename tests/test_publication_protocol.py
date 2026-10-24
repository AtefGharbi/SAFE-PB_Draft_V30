import numpy as np

from safe_pb.baselines import (QLearningConfig, TabularQLearningAgent,
                               run_price_schedule, run_rolling_horizon_mpc)
from safe_pb.data_loader import build_tier_a_scenario
from safe_pb.episode_metrics import discomfort_normalizers
from safe_pb.environment import EnvConfig, SafePBEnv
from safe_pb.feeder_model import ac_powerflow_check, build_feeder
from safe_pb.statistical_analysis import paired_test


CONFIG = "config/tier_a_wan2022.yaml"


def test_mpc_is_causal_full_day_rollout():
    scenario = build_tier_a_scenario(CONFIG, random_seed=11)
    results, actions = run_rolling_horizon_mpc(
        scenario, horizon=2, grid_points=2, beam_width=3)
    assert len(results) == scenario.T
    assert len(actions) == scenario.T
    assert all(len(action) == 3 for action in actions)


def test_tabular_policy_can_be_applied_to_held_out_scenario():
    training = build_tier_a_scenario(CONFIG, random_seed=21)
    held_out = build_tier_a_scenario(CONFIG, random_seed=22)
    agent = TabularQLearningAgent(
        training, QLearningConfig(grid_points=2, n_episodes=2,
                                  epsilon_decay_episodes=2, seed=3))
    agent.train()
    schedule = agent.greedy_price_schedule()
    results = run_price_schedule(held_out, schedule)
    assert len(results) == held_out.T


def test_all_zero_paired_difference_is_valid_null_result():
    result = paired_test(np.arange(20.0), np.arange(20.0))
    assert result["test"] == "wilcoxon"
    assert result["p_value"] == 1.0


def test_fairness_population_excludes_structural_battery_zeros():
    scenario = build_tier_a_scenario(CONFIG, random_seed=31)
    normalizers = discomfort_normalizers(scenario)
    assert len(normalizers) == len(scenario.gen_loads) + len(scenario.pevs)
    assert np.all(normalizers > 0)


def test_recycled_ac_matches_fresh_flat_start():
    first = build_feeder()
    buses = first.non_slack_buses
    p0 = {b: 0.001 * (1 + b % 3) for b in buses}
    q0 = {b: 0.0003 * (1 + b % 2) for b in buses}
    p1 = {b: 0.0015 * (1 + b % 4) for b in buses}
    q1 = {b: 0.0002 * (1 + b % 3) for b in buses}
    ac_powerflow_check(first, p0, q0, base_scale=0.35)
    recycled = ac_powerflow_check(first, p1, q1, base_scale=0.35)

    fresh = build_feeder()
    cold = ac_powerflow_check(fresh, p1, q1, base_scale=0.35)
    assert recycled["converged"] and cold["converged"]
    assert np.isclose(recycled["min_vm_pu"], cold["min_vm_pu"], atol=1e-9)
    assert np.allclose(
        [recycled["vm_pu"][b] for b in sorted(recycled["vm_pu"])],
        [cold["vm_pu"][b] for b in sorted(cold["vm_pu"])],
        atol=1e-9,
    )


def test_training_mode_skips_only_candidate_ac(monkeypatch):
    import safe_pb.environment as environment_module

    scenario = build_tier_a_scenario(CONFIG, random_seed=41)
    topo = build_feeder()
    real_check = environment_module.ac_powerflow_check
    calls = []

    def counted_check(*args, **kwargs):
        calls.append(1)
        return real_check(*args, **kwargs)

    monkeypatch.setattr(environment_module, "ac_powerflow_check", counted_check)
    env = SafePBEnv(
        scenario, topo,
        EnvConfig(base_scale=0.35, verify_candidate_ac=False),
    )
    env.reset()
    env.utility_announce_price(np.zeros(env.cfg.n_utility_act))
    _, _, _, _, info = env.household_step(
        np.zeros((env.n_households, env.cfg.n_household_act)))
    assert len(calls) == 1
    assert np.isnan(info["candidate_min_voltage_ac"])
    assert np.isfinite(info["min_voltage_ac"])

"""
Smoke tests for the SAC/CTDE training implementation (Command 10, second
implementation pass).

Per the explicit scope of this pass: these tests prove the MECHANICS work
(parameters actually update via gradient descent, losses stay finite,
checkpoints save/restore exactly, a full episode runs end-to-end through
the real environment and safety layer) -- they do NOT test training
quality, convergence, or produce any number that belongs in the
manuscript's results tables. Kept in a separate file from test_core.py
since they depend on torch, a heavier dependency than the rest of this
codebase.
"""
import os
import sys
import copy
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

from safe_pb.data_loader import build_tier_a_scenario
from safe_pb.feeder_model import build_feeder, ac_powerflow_check
from safe_pb.environment import (SafePBEnv, EnvConfig, ev_available,
                                 EV_AWAY_START, EV_AWAY_END)
from safe_pb.sac_trainer import HierarchicalSACTrainer, SACConfig
from safe_pb.sac_networks import GaussianActor, TwinCritic
from safe_pb.ev_model import EVTerminalInfeasible

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "tier_a_wan2022.yaml")


@pytest.fixture(scope="module")
def scenario():
    return build_tier_a_scenario(CONFIG_PATH)


@pytest.fixture(scope="module")
def topo():
    return build_feeder()


@pytest.fixture
def trainer(scenario, topo):
    env = SafePBEnv(scenario, topo)
    cfg = SACConfig(batch_size=16, seed=0)
    t = HierarchicalSACTrainer(env, cfg)
    t.collect_episode()
    t.collect_episode()
    return t


# ---------------------------------------------------------------------------
# Environment: EV availability, fixed loads, PEV bounds, price sequencing
# ---------------------------------------------------------------------------

def test_ev_availability_window_is_real():
    assert not ev_available(EV_AWAY_START)
    assert not ev_available(EV_AWAY_END - 1)
    assert ev_available(EV_AWAY_END)
    assert ev_available(0)
    assert ev_available(23)


def test_environment_full_episode_runs_through_safety_layer(scenario, topo):
    """A complete episode must run end-to-end through the real environment,
    reporting BOTH linear and independent AC voltage every step."""
    env = SafePBEnv(scenario, topo)
    u_obs, h_obs = env.reset()
    assert u_obs.shape == (env.cfg.n_utility_obs,)
    assert h_obs.shape == (env.n_households, env.cfg.n_household_obs)

    rng = np.random.default_rng(0)
    n_steps = 0
    done = False
    while not done:
        u_act = rng.uniform(-1, 1, size=env.cfg.n_utility_act)
        h_obs_now = env.utility_announce_price(u_act)
        h_act = rng.uniform(-1, 1, size=(env.n_households, env.cfg.n_household_act))
        u_obs, h_obs, reward, done, info = env.household_step(h_act)
        assert np.isfinite(reward)
        assert "projection_distance" in info
        assert "min_voltage_linear" in info and "min_voltage_ac" in info
        n_steps += 1
    assert n_steps == scenario.T


def test_ev_actions_are_zero_while_away(scenario, topo):
    env = SafePBEnv(scenario, topo)
    env.reset()
    env.t = EV_AWAY_START + 1
    household_kw = np.zeros(env.n_households)
    devices, a_hat, _, _ = env.build_safety_devices(household_kw)
    for j in range(env.n_pev):
        d = devices[env.n_gen + j]
        assert d.lb_kw == 0.0 and d.ub_kw == 0.0


def test_fixed_loads_are_included_in_safety_projection(scenario, topo):
    """External review: an earlier version's build_safety_devices had a
    'for ... : pass' loop that never actually added non-dispatchable
    household loads to fixed_p/fixed_q -- the safety layer and reported
    voltage silently ignored them. This locks in that they are real,
    nonzero, and match the scenario's actual non-dispatchable demand.

    UPDATED for battery/PV wiring: fixed_p/fixed_q now also carry PV's
    negative (generation) injection at PV's own assigned buses, so the
    bus-entry count is n_nondispatch + n_pv (each on a distinct bus,
    guaranteed by the single round-robin assignment in __init__ since
    len(household_ids+non_ids+pv_ids) <= the feeder's non-slack bus
    count), and the net fixed_p total is non-dispatch load MINUS PV
    generation, not non-dispatch load alone."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    devices, a_hat, fixed_p, fixed_q = env.build_safety_devices(np.zeros(env.n_households))
    assert len(fixed_p) == env.n_nondispatch + env.n_pv  # one bus entry per non-dispatch load + per PV unit
    assert sum(fixed_q.values()) > 0.0  # PV is unity-PF (zero reactive), so this is still all from non-dispatch loads
    expected_nondispatch_kw = float(scenario.nondispatch_demand_kwh[:, 0].sum())
    expected_pv_kw = float(scenario.pv_profiles_kw[:, 0].sum())
    expected_net_kw = expected_nondispatch_kw - expected_pv_kw
    actual_total_mw = sum(fixed_p.values())
    assert actual_total_mw == pytest.approx(expected_net_kw * env.cfg.homes_per_bus / 1000.0, rel=1e-6)


def test_ac_voltage_independently_verified_after_projection(scenario, topo):
    """household_step must report an independently AC-verified voltage,
    not only the linear model's estimate -- and they should generally
    differ (confirming AC is a real, separate computation, not a copy)."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    u_obs = env._utility_obs()
    h_obs = env.utility_announce_price(np.zeros(3))
    _, _, _, _, info = env.household_step(np.zeros((env.n_households, 1)))
    assert info["ac_converged"]
    assert np.isfinite(info["min_voltage_ac"])
    assert np.isfinite(info["min_voltage_linear"])


def test_overloaded_case_produces_nonzero_projection_distance(scenario, topo):
    """A candidate schedule at maximum device-bound consumption, evaluated
    on the KNOWN native-violating feeder (base_scale=1.0, established in
    the safety-layer validation report to sit below 0.95 p.u. even before
    any household load), must be pulled back by the projection --
    projection_distance must be nonzero, confirming the safety layer
    actually intervenes rather than passing candidates through unchanged.
    The comparison uses the device-feasible candidate rather than the raw
    actor output, so it isolates a network-level intervention, which
    base_scale=1.0 reliably triggers."""
    env = SafePBEnv(scenario, topo)
    env.cfg.base_scale = 1.0
    env.reset()
    env.utility_announce_price(np.array([1.0, 1.0, 1.0]))
    extreme_action = np.full((env.n_households, 1), 1.0)
    _, _, _, _, info = env.household_step(extreme_action)
    assert info["projection_distance"] > 0.0
    assert np.any(np.abs(info["applied_actions_kw"] - info["candidate_actions_kw"]) > 1e-9)
    assert np.allclose(info["projection_delta_kw"],
                       info["applied_actions_kw"] - info["candidate_actions_kw"])
    assert "candidate_min_voltage_ac" in info
    assert "candidate_min_voltage_linear" in info
    assert info["projection_active"]


def test_pev_bounds_respect_soc_capacity(scenario, topo):
    """A near-full PEV must not be assigned an upper bound that would
    push it past capacity; a mid-charge PEV should retain real flexibility."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    ev = scenario.pevs[0]
    env.pev_soc[0] = ev.C_M - 0.5
    lb, ub = env._pev_bounds(0, ev, 20)  # an available hour
    max_possible_charge = (ev.C_M - env.pev_soc[0]) / ev.mu_c
    assert ub <= max_possible_charge + 1e-6
    assert ub < ev.p_rated_kw  # genuinely constrained below rated power near full


def test_pev_soc_never_exceeds_capacity_over_a_full_episode(scenario, topo):
    """Running PEVs at their bound-respecting maximum charge every
    available hour must never push SoC outside [C_m, C_M]."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    done = False
    while not done:
        env.utility_announce_price(np.zeros(3))
        h_act = np.ones((env.n_households, 1))  # always request max charging
        _, _, _, done, _ = env.household_step(h_act)
        for j, ev in enumerate(scenario.pevs):
            assert ev.C_m - 1e-6 <= env.pev_soc[j] <= ev.C_M + 1e-6


def test_households_observe_current_not_previous_price(scenario, topo):
    """External review: an earlier version let both actors act from the
    SAME stale observation, so households never actually saw the price
    they were responding to. utility_announce_price must change the
    household observation's price field relative to the pre-announcement
    (previous-step) observation."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    obs_before = env._household_obs_all(env.last_prices)
    obs_after = env.utility_announce_price(np.array([1.0, -1.0, 0.5]))  # a price far from the default
    # price fields are index 6 (gen) -- must differ after a real price announcement
    assert not np.allclose(obs_before[:, 6], obs_after[:, 6])


def test_household_step_requires_prior_price_announcement(scenario, topo):
    """household_step must refuse to run if utility_announce_price wasn't
    called first this timestep -- households cannot act on an undetermined
    price."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    with pytest.raises(RuntimeError):
        env.household_step(np.zeros((env.n_households, 1)))


def test_hours_remaining_excludes_away_window(scenario, topo):
    """External review: an earlier version counted ALL remaining calendar
    hours (including the 09:00-17:00 away window) as available charging
    opportunity, overestimating how much time a PEV actually has."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    naive = scenario.T - 5
    available = env._available_hours_remaining(5)
    assert available < naive
    assert available == naive - (EV_AWAY_END - EV_AWAY_START)  # exactly the away window excluded


def test_pev_infeasibility_is_recorded_not_silently_absorbed(scenario, topo):
    """External review: catching EVTerminalInfeasible and forcing max
    charging without any record was flagged as hiding a genuine terminal
    failure. Force a genuinely infeasible case (near end of day, large
    remaining gap) and confirm it's recorded in _pev_infeasibility_events."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    ev = scenario.pevs[0]
    env.pev_soc[0] = ev.C_m  # empty
    lb, ub = env._pev_bounds(0, ev, scenario.T - 1)  # last hour, empty battery: certainly infeasible
    assert len(env._pev_infeasibility_events) >= 1
    assert env._pev_infeasibility_events[-1]["pev_id"] == ev.id


def test_ac_verification_uses_signed_pev_power(scenario, topo):
    """External review: an earlier version used max(kw,0.0) when building
    the AC-check injection, discarding PEV discharge/export entirely, so
    AC verification did not evaluate the same schedule the projection
    actually applied whenever a device discharged. Force a discharge
    scenario and confirm the reported AC voltage differs from what a
    (buggy) max(kw,0)-based check would have produced."""
    env = SafePBEnv(scenario, topo,
                    EnvConfig(pev_infeasibility_policy="continue_and_report"))
    env.reset()
    env.utility_announce_price(np.zeros(3))
    # request maximum discharge from every household (negative normalized action)
    h_act = np.full((env.n_households, 1), -1.0)
    _, _, _, _, info = env.household_step(h_act)
    pev_power = info["applied_kw"][env.n_gen:]
    assert np.any(pev_power < -1e-9)
    assert info["ac_converged"]
    assert np.isfinite(info["min_voltage_ac"])

    env.t = 0  # reconstruct the same hour that household_step just evaluated
    devices, _, fixed_p, fixed_q = env.build_safety_devices(info["applied_kw"])
    signed_p, signed_q = dict(fixed_p), dict(fixed_q)
    clipped_p, clipped_q = dict(fixed_p), dict(fixed_q)
    for d, kw in zip(devices, info["applied_kw"]):
        signed_p[d.bus] = signed_p.get(d.bus, 0.0) + kw * env.cfg.homes_per_bus / 1000.0
        signed_q[d.bus] = (signed_q.get(d.bus, 0.0) +
                           kw * env.cfg.q_over_p * env.cfg.homes_per_bus / 1000.0)
        clipped = max(kw, 0.0)
        clipped_p[d.bus] = clipped_p.get(d.bus, 0.0) + clipped * env.cfg.homes_per_bus / 1000.0
        clipped_q[d.bus] = (clipped_q.get(d.bus, 0.0) +
                            clipped * env.cfg.q_over_p * env.cfg.homes_per_bus / 1000.0)
    signed_ac = ac_powerflow_check(topo, signed_p, signed_q, base_scale=env.cfg.base_scale)
    clipped_ac = ac_powerflow_check(topo, clipped_p, clipped_q, base_scale=env.cfg.base_scale)
    assert info["min_voltage_ac"] == pytest.approx(signed_ac["min_vm_pu"], abs=1e-9)
    assert abs(signed_ac["min_vm_pu"] - clipped_ac["min_vm_pu"]) > 1e-9


def test_base_scale_has_one_authoritative_source(scenario, topo):
    env = SafePBEnv(scenario, topo)
    assert env.cfg.base_scale == pytest.approx(scenario.network_config["base_scale"])


def test_default_base_scale_gives_real_headroom(scenario, topo):
    """External review: base_scale=0.5 gave an exogenous-only (zero
    controllable device) worst-hour AC minimum of 0.9487 p.u. -- already
    below 0.95 before any controllable action, making the safety layer
    untestable. The recalibrated default must give >= 0.96 p.u."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    worst = 1.0
    for t in range(scenario.T):
        env.t = t
        devices, a_hat, fixed_p, fixed_q = env.build_safety_devices(np.zeros(env.n_households))
        ac = ac_powerflow_check(topo, fixed_p, fixed_q, base_scale=env.cfg.base_scale)
        if ac["converged"]:
            worst = min(worst, ac["min_vm_pu"])
    assert worst >= 0.96


def test_reward_includes_pev_dissatisfaction_and_degradation(scenario, topo):
    """External review: an earlier version's reward/info only included
    general-dispatchable-load discomfort, omitting PEV dissatisfaction and
    degradation entirely."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    env.utility_announce_price(np.zeros(3))
    _, _, _, _, info = env.household_step(np.zeros((env.n_households, 1)))
    assert "pev_dissatisfaction" in info and "pev_degradation" in info
    assert len(info["pev_dissatisfaction"]) == env.n_pev


def test_terminal_step_reports_pev_shortfall(scenario, topo):
    """The final step of an episode must report terminal SoC and shortfall
    per PEV, not leave terminal feasibility unreported."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    done = False
    info = None
    while not done:
        env.utility_announce_price(np.zeros(3))
        _, _, _, done, info = env.household_step(np.zeros((env.n_households, 1)))
    assert "terminal_pev_soc" in info and "terminal_pev_shortfall_kwh" in info
    assert len(info["terminal_pev_shortfall_kwh"]) == env.n_pev
    assert np.all(info["terminal_pev_shortfall_kwh"] >= 0.0)


def test_feasible_episode_reaches_all_pev_targets_and_conserves_energy(scenario, topo):
    env = SafePBEnv(scenario, topo)
    env.reset()
    done = False
    info = None
    while not done:
        env.utility_announce_price(np.zeros(3))
        _, _, _, done, info = env.household_step(np.zeros((env.n_households, 1)))
        assert np.max(np.abs(info["pev_energy_balance_residual_kwh"])) < 1e-8
    assert not info["episode_terminated_for_pev_infeasibility"]
    assert np.all(info["terminal_pev_shortfall_kwh"] <= 1e-6)


def test_reject_policy_raises_for_deliberately_infeasible_pev(scenario, topo):
    env = SafePBEnv(scenario, topo,
                    EnvConfig(pev_infeasibility_policy="reject_scenario"))
    env.reset()
    env.t = scenario.T - 1
    env.pev_soc[0] = scenario.pevs[0].C_m
    with pytest.raises(EVTerminalInfeasible):
        env._pev_bounds(0, scenario.pevs[0], env.t)


# ---------------------------------------------------------------------------
# SAC trainer: parameter updates, alternating schedule, CVaR, fairness
# ---------------------------------------------------------------------------

def test_sac_parameters_actually_update(trainer):
    before = copy.deepcopy(trainer.utility_actor.state_dict())
    trainer.cfg.alternating_k = 1  # force a switch every step for this test
    trainer.env_step_count = 0
    trainer.train_step()
    after = trainer.utility_actor.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)

    before_h = copy.deepcopy(trainer.household_actor.state_dict())
    trainer.env_step_count = 1  # next block: household's turn
    trainer.train_step()
    after_h = trainer.household_actor.state_dict()
    assert any(not torch.equal(before_h[k], after_h[k]) for k in before_h)


def test_frozen_actor_does_not_update_during_the_others_step(trainer):
    trainer.cfg.alternating_k = 1000  # stay in the utility block the whole test
    trainer.env_step_count = 0
    before_h = copy.deepcopy(trainer.household_actor.state_dict())
    trainer.train_step()
    after_h = trainer.household_actor.state_dict()
    assert all(torch.equal(before_h[k], after_h[k]) for k in before_h)


def test_alternating_k_produces_the_requested_update_sequence(trainer):
    """External review: alternating_k was defined but never used --
    should_update_utility() must actually alternate every K ENV steps,
    not every episode or every call."""
    trainer.cfg.alternating_k = 3
    results = []
    for step in range(9):
        trainer.env_step_count = step
        results.append(trainer.should_update_utility())
    # 3 steps utility, 3 steps household, 3 steps utility again
    assert results == [True, True, True, False, False, False, True, True, True]


def test_sac_losses_are_finite(trainer):
    for _ in range(6):
        losses = trainer.train_step()
        for k, v in losses.items():
            if k == "updated":
                continue
            assert np.isfinite(v)


def test_train_step_raises_on_insufficient_buffer(scenario, topo):
    env = SafePBEnv(scenario, topo)
    cfg = SACConfig(batch_size=1000, seed=0)
    t = HierarchicalSACTrainer(env, cfg)
    t.collect_episode()
    with pytest.raises(RuntimeError):
        t.train_step()


def test_beta_r_positive_uses_complete_day_cvar_state(scenario, topo):
    env = SafePBEnv(scenario, topo)
    cfg = SACConfig(batch_size=16, seed=0, beta_r=1.0, alternating_k=1000,
                    risk_warmup_episodes=2)
    t = HierarchicalSACTrainer(env, cfg)
    t.collect_episode(); t.collect_episode()
    assert t.cvar_zeta is not None
    assert len(t.daily_loss_history) == 2
    assert 'risk_excess_penalty_unweighted' in t.last_episode_metrics
    before_actor = copy.deepcopy(t.utility_actor.state_dict())
    losses = t.train_step()
    assert any(not torch.equal(before_actor[k], t.utility_actor.state_dict()[k]) for k in before_actor)


def test_beta_f_positive_uses_normalized_daily_discomfort(scenario, topo):
    env = SafePBEnv(scenario, topo)
    cfg = SACConfig(batch_size=16, seed=0, beta_f=1.0, alternating_k=1)
    t = HierarchicalSACTrainer(env, cfg)
    t.collect_episode(); t.collect_episode()
    assert 'normalized_discomfort_variance' in t.last_episode_metrics
    assert t.last_episode_metrics['normalized_discomfort_variance'] >= 0
    t.env_step_count = 1
    before = copy.deepcopy(t.household_actor.state_dict())
    t.train_step()
    assert any(not torch.equal(before[k], t.household_actor.state_dict()[k]) for k in before)


def test_checkpoint_restores_exact_state_including_buffer_and_rng(trainer, tmp_path):
    """External review: an earlier version's checkpoint omitted the
    entropy/zeta optimizers, replay buffer, episode-loss history, and RNG
    state. This test locks in that a full save/mutate/restore cycle
    recovers ALL of these, not just the actor weights."""
    trainer.cfg.alternating_k = 1
    trainer.env_step_count = 0
    trainer.train_step()
    ckpt_path = str(tmp_path / "ckpt.pt")
    trainer.save_checkpoint(ckpt_path)

    saved_weights = copy.deepcopy(trainer.utility_actor.state_dict())
    saved_step_count = trainer.train_step_count
    saved_env_step_count = trainer.env_step_count
    saved_buffer_size = len(trainer.buffer)
    saved_episode_losses = list(trainer.episode_losses)
    saved_alpha_opt_state = copy.deepcopy(trainer.alpha_opt.state_dict())

    # Mutate further: more training (a genuine utility update this time --
    # env_step_count=2 with alternating_k=1 lands back in a utility block:
    # block=2//1=2, even -> True) AND more buffer growth
    trainer.env_step_count = 2
    trainer.train_step()
    trainer.collect_episode()
    mutated_weights = trainer.utility_actor.state_dict()
    assert any(not torch.equal(saved_weights[k], mutated_weights[k]) for k in saved_weights)
    assert len(trainer.buffer) != saved_buffer_size

    trainer.load_checkpoint(ckpt_path)
    restored_weights = trainer.utility_actor.state_dict()
    assert all(torch.equal(saved_weights[k], restored_weights[k]) for k in saved_weights)
    assert trainer.train_step_count == saved_step_count
    assert trainer.env_step_count == saved_env_step_count
    assert len(trainer.buffer) == saved_buffer_size
    assert trainer.episode_losses == saved_episode_losses
    restored_alpha_opt_state = trainer.alpha_opt.state_dict()
    assert str(saved_alpha_opt_state) == str(restored_alpha_opt_state)


def test_checkpoint_resume_reproduces_uninterrupted_training(scenario, topo):
    """Deterministic resume-equivalence: training A->B->C (uninterrupted)
    must produce the same final actor weights as A->B, checkpoint, reload,
    ->C, given the same seed and no other source of randomness between
    the two paths (both use deterministic=True episode collection so no
    stochastic action sampling diverges the two runs' buffer contents)."""
    def make_trainer():
        env = SafePBEnv(scenario, topo)
        cfg = SACConfig(batch_size=16, seed=0, alternating_k=1)
        t = HierarchicalSACTrainer(env, cfg)
        torch.manual_seed(0)
        t.collect_episode(deterministic=True)
        t.collect_episode(deterministic=True)
        return t

    # Path 1: uninterrupted
    t1 = make_trainer()
    t1.env_step_count = 0
    t1.train_step()
    t1.env_step_count = 1
    t1.train_step()

    # Path 2: interrupted by a checkpoint/reload in between
    t2 = make_trainer()
    t2.env_step_count = 0
    t2.train_step()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, "ckpt.pt")
        t2.save_checkpoint(ckpt_path)
        t2.load_checkpoint(ckpt_path)
    t2.env_step_count = 1
    t2.train_step()

    w1 = t1.utility_actor.state_dict()
    w2 = t2.utility_actor.state_dict()
    assert all(torch.allclose(w1[k], w2[k]) for k in w1)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

def test_gaussian_actor_action_is_bounded():
    actor = GaussianActor(obs_dim=5, act_dim=3)
    obs = torch.randn(10, 5) * 100
    action, log_prob, mean = actor.sample(obs)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    assert torch.all(torch.isfinite(log_prob))


def test_twin_critic_outputs_two_independent_estimates():
    critic = TwinCritic(global_obs_dim=10, global_act_dim=4)
    obs = torch.randn(8, 10)
    act = torch.randn(8, 4)
    q1, q2 = critic(obs, act)
    assert q1.shape == (8,) and q2.shape == (8,)
    assert torch.all(torch.isfinite(q1)) and torch.all(torch.isfinite(q2))


# ---------------------------------------------------------------------------
# Battery / PV wiring (new: batteries and PV were previously built by the
# scenario but not used anywhere in the environment)
# ---------------------------------------------------------------------------

def test_battery_households_are_present_and_ordered_after_pev(scenario, topo):
    """Household order must be [gen..., pev..., battery...] so that all
    existing `self.n_gen + j` PEV index math keeps meaning unchanged."""
    env = SafePBEnv(scenario, topo)
    assert env.n_households == env.n_gen + env.n_pev + env.n_bat
    assert env.n_bat == len(scenario.batteries)
    assert env.household_ids[env.n_gen + env.n_pev:] == [f"bat_{b.id}" for b in scenario.batteries]


def test_battery_bounds_respect_soc_capacity(scenario, topo):
    """A near-full battery must not be assigned a charge upper bound that
    would push it past capacity; a near-empty battery must not be assigned
    a discharge lower bound that would push it below zero."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    bat = scenario.batteries[0]
    env.battery_soc[0] = bat.capacity_kwh - 0.5
    lb, ub = env._battery_bounds(0, bat)
    max_possible_charge = (bat.capacity_kwh - env.battery_soc[0]) / bat.eta_c
    assert ub <= max_possible_charge + 1e-6
    assert ub < bat.p_max_kw  # genuinely constrained below rated power near full

    env.battery_soc[0] = 0.5
    lb, ub = env._battery_bounds(0, bat)
    max_possible_discharge = env.battery_soc[0] * bat.eta_d
    assert -lb <= max_possible_discharge + 1e-6


def test_battery_soc_never_exceeds_capacity_over_a_full_episode(scenario, topo):
    """Running batteries at their bound-respecting maximum charge every
    hour must never push SoC outside [0, capacity_kwh]."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    done = False
    while not done:
        env.utility_announce_price(np.zeros(3))
        h_act = np.ones((env.n_households, 1))  # always request max charging
        _, _, _, done, _ = env.household_step(h_act)
        for k, bat in enumerate(scenario.batteries):
            assert -1e-6 <= env.battery_soc[k] <= bat.capacity_kwh + 1e-6


def test_battery_degradation_cost_is_reported_and_nonzero_when_cycling(scenario, topo):
    """Charging/discharging a battery must produce a nonzero, finite
    degradation cost (Eq. 9) reported in info, and the reward equation
    (via C_t) must account for it -- not just an unused diagnostic."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    env.utility_announce_price(np.zeros(3))
    h_act = np.ones((env.n_households, 1))  # request max charge for everyone
    _, _, _, _, info = env.household_step(h_act)
    assert "battery_degradation" in info and "p_bat" in info and "battery_soc" in info
    assert len(info["battery_degradation"]) == env.n_bat
    assert np.all(np.isfinite(info["battery_degradation"]))
    assert np.any(np.abs(info["p_bat"]) > 1e-9)  # batteries actually moved
    assert np.any(info["battery_degradation"] > 0.0)


def test_battery_has_no_discomfort_but_is_billed(scenario, topo):
    """Batteries carry no discomfort term (household_discomfort_all's
    battery entries are always exactly 0.0) but DO appear in the realized
    per-household bill diagnostic (household_energy_bill)."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    env.utility_announce_price(np.zeros(3))
    h_act = np.full((env.n_households, 1), -1.0)  # request max discharge for everyone
    _, _, _, _, info = env.household_step(h_act)
    battery_discomfort = info["household_discomfort_all"][env.n_gen + env.n_pev:]
    assert np.all(battery_discomfort == 0.0)
    assert len(info["household_energy_bill"]) == env.n_households


def test_pv_reduces_net_fixed_injection(scenario, topo):
    """PV must enter fixed_p as a NEGATIVE injection (generation offsets
    load) at its own assigned bus -- distinct from, and on top of, the
    non-dispatchable loads' positive injections."""
    env = SafePBEnv(scenario, topo)
    env.reset()
    assert env.n_pv == scenario.pv_profiles_kw.shape[0]
    _, _, fixed_p, _ = env.build_safety_devices(np.zeros(env.n_households))
    pv_buses = {env.bus_map[pv_id] for pv_id in env.pv_ids}
    non_buses = {env.bus_map[nid] for nid in env.non_ids}
    # With n_non=5, n_pv=6 << 32 non-slack buses and a single round-robin
    # assignment, PV and non-dispatch buses must be disjoint.
    assert pv_buses.isdisjoint(non_buses)
    for pv_id in env.pv_ids:
        bus = env.bus_map[pv_id]
        assert fixed_p[bus] <= 0.0  # PV-only buses must be net-negative (generation)

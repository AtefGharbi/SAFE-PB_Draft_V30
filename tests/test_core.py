import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safe_pb.data_loader import build_tier_a_scenario
from safe_pb.household_models import GeneralDispatchableLoad, NonDispatchableLoad
from safe_pb.ev_model import PEVModel, EVTerminalInfeasible
from safe_pb.battery_model import BatteryModel
from safe_pb import metrics
from safe_pb.feeder_model import build_feeder, linearized_voltages, ac_powerflow_check, assign_households_to_buses
from safe_pb.safety_projection import Device, project

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "tier_a_wan2022.yaml")


@pytest.fixture(scope="module")
def scenario():
    return build_tier_a_scenario(CONFIG_PATH)


@pytest.fixture(scope="module")
def topo():
    return build_feeder()


def test_scenario_loads_and_shapes(scenario):
    assert scenario.T == 24
    assert len(scenario.gen_loads) == 6
    assert len(scenario.pevs) == 4
    assert scenario.nondispatch_demand_kwh.shape[0] == 5
    assert scenario.gen_demand_kwh.shape == (6, 24)


def test_wholesale_price_matches_table_ii(scenario):
    # spot-check a few values transcribed from Wan et al. Table II
    assert scenario.wholesale_price[0] == pytest.approx(1.9)
    assert scenario.wholesale_price[15] == pytest.approx(4.4)
    assert scenario.wholesale_price[-1] == pytest.approx(2.6)


def test_non_dispatchable_consumption_equals_demand():
    """Eq. (10): p_{n,t} = e_{n,t} always, no price effect."""
    ld = NonDispatchableLoad(id="test")
    e = np.array([10.0, 20.0, 5.0])
    p = ld.consumption(e)
    assert np.allclose(p, e)


def test_general_dispatchable_dr_bounds_respected():
    """Eq. (3): DR_min <= e - p <= DR_max must hold after clipping."""
    ld = GeneralDispatchableLoad(id="g1", alpha=5.0, beta=5.0, dr_min_frac=0.1, dr_max_frac=0.5)
    e = np.array([10.0])
    # extreme price far outside bounds should still respect DR limits after clipping
    p, was_clipped = ld.consumption(e, eta_price=np.array([100.0]),
                                     wholesale_price=np.array([2.0]), xi_elasticity=np.array([-0.5]))
    gap = e - p
    assert gap[0] >= ld.dr_min_frac * e[0] - 1e-9
    assert gap[0] <= ld.dr_max_frac * e[0] + 1e-9
    assert was_clipped  # this price is extreme enough that clipping must trigger


def test_general_dispatchable_price_equals_wholesale_still_hits_dr_min():
    """When eta == lambda, Eq. (1) collapses to a raw p = e (no elasticity
    effect) -- BUT Eq. (3)'s DR_min = 0.1*e is a mandatory minimum
    reduction that Table I applies regardless of price. So the source
    model actually forces >=10% shed even at zero price markup; the raw
    p=e point is infeasible under (3) and gets clipped to e*(1-DR_min_frac).
    This is a genuine property of the source paper's own equations,
    surfaced by this validation pass -- not a bug in our reproduction."""
    ld = GeneralDispatchableLoad(id="g1", alpha=5.0, beta=5.0, dr_min_frac=0.1, dr_max_frac=0.5)
    e = np.array([10.0])
    p, clipped = ld.consumption(e, eta_price=np.array([2.0]), wholesale_price=np.array([2.0]),
                                 xi_elasticity=np.array([-0.5]))
    assert p[0] == pytest.approx(9.0)   # e * (1 - dr_min_frac)
    assert clipped


def test_pev_terminal_energy_diagnostic_and_capacity_bounds():
    """Eq. (7): SoC must stay within [C_m, C_M] at every step (hard-clipped);
    Eq. (6): terminal shortfall is reported, not silently hidden."""
    ev = PEVModel(id="pevtest", alpha=3.5, beta=4, C_m=5, C_M=98, mu_c=0.9, mu_d=0.9,
                   p_rated_kw=9, E0=10, ET=98)
    T = 24
    e_demand = np.zeros(T)
    e_demand[18:24] = 20.0  # ask for a lot of charging in the evening
    eta = np.full(T, 3.0)
    lam = np.full(T, 3.0)
    xi = np.full(T, -0.5)
    out = ev.simulate(e_demand, eta, lam, xi)
    assert np.all(out["soc_kwh"] >= ev.C_m - 1e-6)
    assert np.all(out["soc_kwh"] <= ev.C_M + 1e-6)
    assert "terminal_shortfall_kwh" in out  # must be reported, not assumed zero


def test_scenario_pev_targets_are_physically_feasible(scenario):
    """External review caught that all 4 PEVs' terminal-energy targets were
    unreachable under an earlier 8h charging-window assumption (e.g. PEV1
    needed 88 kWh but only 64.8 kWh was deliverable in that window). This
    regression test locks in the fix: every PEV's (ET-E0) target must be
    achievable within 24h at rated power * charging efficiency."""
    for ev in scenario.pevs:
        need = ev.ET - ev.E0
        max_deliverable = ev.p_rated_kw * ev.mu_c * scenario.T
        assert max_deliverable >= need, (
            f"{ev.id}: needs {need} kWh but only {max_deliverable} kWh "
            f"deliverable in {scenario.T}h -- infeasible scenario")


def test_pev_viability_floor_detects_infeasibility_immediately():
    """External review's specific point: an earlier version only detected
    infeasibility when hours_remaining<=0, silently returning a
    capped-at-rated-power floor otherwise even when that's insufficient.
    88 kWh needed with only 2 hours remaining at 9 kW rated power (max
    deliverable 16.2 kWh) must raise immediately, not wait until the
    hours run out."""
    ev = PEVModel(id="pevtest8", alpha=3.5, beta=4, C_m=5, C_M=98, mu_c=0.9, mu_d=0.9,
                   p_rated_kw=9, E0=10, ET=98)
    with pytest.raises(EVTerminalInfeasible):
        ev.viability_floor_kw(current_soc_kwh=10, hours_remaining=2)
    # Sanity: the same EV with enough hours remaining must NOT raise
    ev.viability_floor_kw(current_soc_kwh=10, hours_remaining=20)


def test_pev_viability_floor_forces_charging_when_time_running_out():
    """External review asked for enforcement, not just diagnosis, of the
    terminal EV target. This is the per-step viability constraint: if
    skipping this step would make the target unreachable even at full
    rated power for all remaining steps, the floor must be > 0."""
    ev = PEVModel(id="pevtest3", alpha=3.5, beta=4, C_m=5, C_M=98, mu_c=0.9, mu_d=0.9,
                   p_rated_kw=9, E0=10, ET=98)
    # Plenty of time left: floor should be 0 (no need to force charging yet)
    assert ev.viability_floor_kw(current_soc_kwh=10, hours_remaining=20) == 0.0
    # 2 hours left, moderate remaining need: 1 future step can deliver up to
    # p_rated*mu_c=8.1 kWh, so this step must cover the rest (a genuine
    # mid-range floor, not clipped to rated power).
    floor = ev.viability_floor_kw(current_soc_kwh=85, hours_remaining=2)
    assert 0 < floor < ev.p_rated_kw
    # Already at target: floor should be 0 regardless of time left
    assert ev.viability_floor_kw(current_soc_kwh=98, hours_remaining=5) == 0.0


def test_pev_viability_floor_raises_when_genuinely_infeasible():
    """External review's point: hours_remaining==0 with a real energy gap
    must be signaled explicitly, not returned as an indistinguishable 0.0
    (which reads identically to 'target already met')."""
    from safe_pb.ev_model import EVTerminalInfeasible
    ev = PEVModel(id="pevtest4", alpha=3.5, beta=4, C_m=5, C_M=98, mu_c=0.9, mu_d=0.9,
                   p_rated_kw=9, E0=10, ET=98)
    with pytest.raises(EVTerminalInfeasible):
        ev.viability_floor_kw(current_soc_kwh=50, hours_remaining=0)
    # but hours_remaining==0 with target ALREADY met must NOT raise
    assert ev.viability_floor_kw(current_soc_kwh=98, hours_remaining=0) == 0.0


def test_pev_trajectory_viability_enforcement_reaches_target():
    """External review's point 4: the viability floor must be integrated
    into an actual sequential trajectory, not left as a disconnected
    helper. With enforcement on, a PEV facing a price trajectory that
    would otherwise undercharge it must still reach its terminal target
    (or explicitly raise infeasibility if that's truly impossible)."""
    ev = PEVModel(id="pevtest5", alpha=3.5, beta=4, C_m=5, C_M=98, mu_c=0.9, mu_d=0.9,
                   p_rated_kw=9, E0=10, ET=98)
    T = 24
    need = ev.ET - ev.E0
    e_demand = np.full(T, (need / ev.mu_c) / T)  # feasible target, spread evenly (matches data_loader)
    # Adversarial price: always high, which via Eq.(4)'s elasticity term
    # suppresses charging every hour -- exactly the TOU/Q-learning pattern
    # that causes undercharging in the non-enforced simulate().
    eta = np.full(T, 6.0)
    lam = np.full(T, 2.0)
    xi = np.full(T, -0.7)

    unenforced = ev.simulate(e_demand, eta, lam, xi)
    assert unenforced["terminal_shortfall_kwh"] > 1.0  # confirms the adversarial case actually undercharges

    enforced = ev.simulate_with_viability_enforcement(e_demand, eta, lam, xi)
    assert enforced["terminal_shortfall_kwh"] == pytest.approx(0.0, abs=1e-6)
    assert np.any(enforced["viability_forced"])  # the floor actually had to intervene


def test_pev_trajectory_viability_enforcement_raises_if_truly_infeasible():
    """If the target is genuinely unreachable even at full rated power for
    the whole horizon, the trajectory simulation must raise, not return a
    silently-short result."""
    from safe_pb.ev_model import EVTerminalInfeasible
    ev = PEVModel(id="pevtest6", alpha=3.5, beta=4, C_m=5, C_M=98, mu_c=0.9, mu_d=0.9,
                   p_rated_kw=9, E0=10, ET=98)
    T = 2  # far too few hours for an 88 kWh gap at 9kW rated power
    e_demand = np.zeros(T)
    eta = np.full(T, 2.0)
    lam = np.full(T, 2.0)
    xi = np.full(T, -0.5)
    with pytest.raises(EVTerminalInfeasible):
        ev.simulate_with_viability_enforcement(e_demand, eta, lam, xi)


def test_projection_returns_reactive_power_by_device(topo):
    """External review: independent_q's optimized reactive decision was
    computed internally but never returned, making it unavailable for any
    downstream AC check. q_by_device_kvar must be populated for every
    device regardless of inverter mode."""
    from safe_pb.safety_projection import Device, project
    monitored = topo.non_slack_buses[:4]
    d_fixed = Device(name="fixed", bus=monitored[0], lb_kw=0, ub_kw=20, inverter_mode="fixed_pf", q_over_p=0.4)
    d_unity = Device(name="unity", bus=monitored[1], lb_kw=0, ub_kw=20, inverter_mode="unity_pf")
    d_indep = Device(name="indep", bus=monitored[2], lb_kw=0, ub_kw=20, inverter_mode="independent_q", s_max_kva=25.0)
    a_hat = np.array([10.0, 10.0, 10.0])
    res = project(topo, monitored, [d_fixed, d_unity, d_indep], a_hat, fixed_bus_p_mw={}, base_scale=0.5)
    assert set(res.q_by_device_kvar.keys()) == {"fixed", "unity", "indep"}
    assert res.q_by_device_kvar["unity"] == 0.0
    assert res.q_by_device_kvar["fixed"] == pytest.approx(0.4 * res.a_star[0])
    assert len(res.q_indep_star_kvar) == 3


def test_iterative_ac_repair_fallback_uses_original_bounds_and_reactive_power(topo):
    """External review: the fallback previously used bounds already
    tightened by prior iterations (not the device's true physical bounds)
    and omitted reactive power from its AC check entirely. This confirms
    the fallback action respects the ORIGINAL device bounds."""
    from safe_pb.safety_projection import Device, iterative_ac_repair
    devices = [Device(name="d0", bus=topo.non_slack_buses[0], lb_kw=-15, ub_kw=15,
                       weight=1.0, inverter_mode="fixed_pf", q_over_p=0.3)]
    result = iterative_ac_repair(topo, topo.non_slack_buses, devices, np.array([200.0]),
                                  fixed_bus_p_mw={}, base_scale=0.3, max_iterations=2, trust_region_kw=2.0)
    # candidate is absurdly out of bounds; if the fallback used tightened
    # bounds from the (failed) repair iterations it could differ from the
    # device's true [-15, 15] range -- it must not.
    assert -15.0 <= result["final_action"][0] <= 15.0


def test_iterative_ac_repair_genuinely_repairs_a_real_violation(topo):
    """External review's specific point: the existing 'repair succeeds'
    test may pass because the initial candidate is already AC-feasible,
    which doesn't demonstrate the repair mechanism actually working. This
    test starts from a candidate that DOES violate AC voltage, confirms a
    feasible correction exists, and checks the repair (a) actually moves
    the action substantially (not a no-op) and (b) reaches AC feasibility."""
    from safe_pb.safety_projection import Device, iterative_ac_repair
    depths = sorted(((len(p), b) for b, p in topo.path_to_root.items()), reverse=True)
    bus = depths[3][1]  # a moderately deep bus with real but not extreme sensitivity
    devices = [Device(name="d0", bus=bus, lb_kw=0, ub_kw=400, weight=1.0, q_over_p=0.35)]
    a_hat = np.array([250.0])

    # Confirm the starting point genuinely violates before repairing
    from safe_pb.feeder_model import ac_powerflow_check
    p0 = {bus: 250.0 / 1000.0}
    q0 = {bus: 250.0 * 0.35 / 1000.0}
    ac0 = ac_powerflow_check(topo, p0, q0, base_scale=0.5)
    assert ac0["min_vm_pu"] < 0.95, "test setup error: starting point should violate AC voltage"

    result = iterative_ac_repair(topo, topo.non_slack_buses, devices, a_hat,
                                  fixed_bus_p_mw={}, base_scale=0.5, max_iterations=5, trust_region_kw=350.0)
    assert result["outcome"] == "ac_verified_feasible"
    assert not result["fallback_activated"]
    displacement = abs(result["final_action"][0] - a_hat[0])
    assert displacement > 50.0  # the action must have moved substantially, not been a no-op
    # at least one iteration's local correction must show real solver success and displacement
    assert any(h.get("local_correction", {}).get("solver_success") and
               h["local_correction"]["displacement_kw"] > 1.0 for h in result["history"])


def test_ev_terminal_infeasibility_is_not_silently_masked():
    """External review: a broad `except Exception: floor = 0.0` in the
    demo's device construction could convert a genuine terminal-
    infeasibility signal into an unflagged default. viability_floor_kw
    itself must still raise EVTerminalInfeasible (unchanged behavior);
    this locks in that a *specific* exception type is what's raised, so a
    caller can deliberately catch only that one, not swallow everything."""
    from safe_pb.ev_model import EVTerminalInfeasible
    ev = PEVModel(id="pevtest7", alpha=3.5, beta=4, C_m=5, C_M=98, mu_c=0.9, mu_d=0.9,
                   p_rated_kw=9, E0=10, ET=98)
    try:
        ev.viability_floor_kw(current_soc_kwh=20, hours_remaining=0)
        assert False, "expected EVTerminalInfeasible to be raised"
    except EVTerminalInfeasible:
        pass  # correct: a caller catching ONLY this type gets it
    except Exception as e:
        assert False, f"raised wrong exception type: {type(e)}"


def test_iterative_ac_repair_handles_overvoltage():
    """External review's specific point: overvoltage handling was
    implemented (Sec. 2's target_kind='over' logic) but never tested in
    the main scenario, which is undervoltage-only. This is a synthetic
    reverse-power/PV case: a large export (negative kW) at a deep bus
    with light native loading pushes voltage ABOVE v_max, and the repair
    must recognize the 'over' direction, move the action, and reach
    feasibility on the upper bound."""
    from safe_pb.feeder_model import build_feeder, ac_powerflow_check
    from safe_pb.safety_projection import Device, iterative_ac_repair
    topo = build_feeder()
    depths = sorted(((len(p), b) for b, p in topo.path_to_root.items()), reverse=True)
    bus = depths[0][1]  # deepest bus: strongest voltage-rise sensitivity to export
    devices = [Device(name="pv0", bus=bus, lb_kw=-1500, ub_kw=0, weight=1.0, q_over_p=0.35)]
    a_hat = np.array([-900.0])  # large export candidate

    p0 = {bus: -900.0 / 1000.0}
    q0 = {bus: -900.0 * 0.35 / 1000.0}
    ac0 = ac_powerflow_check(topo, p0, q0, base_scale=0.1)
    assert ac0["max_vm_pu"] > 1.05, "test setup error: starting point should violate the upper voltage limit"

    result = iterative_ac_repair(topo, topo.non_slack_buses, devices, a_hat,
                                  fixed_bus_p_mw={}, base_scale=0.1, max_iterations=5, trust_region_kw=1000.0)
    assert result["outcome"] == "ac_verified_feasible"
    assert not result["fallback_activated"]
    # Independent re-verification: don't just trust the internally-reported
    # outcome string -- rerun AC power flow ourselves on final_action.
    from safe_pb.safety_projection import AC_FEASIBILITY_TOL
    final_p = {bus: result["final_action"][0] / 1000.0}
    final_q = {bus: result["final_action"][0] * 0.35 / 1000.0}
    final_ac = ac_powerflow_check(topo, final_p, final_q, base_scale=0.1)
    assert final_ac["converged"]
    assert final_ac["max_vm_pu"] <= 1.05 + AC_FEASIBILITY_TOL
    # the FIRST iteration must have correctly identified an overvoltage target
    assert result["history"][0]["target_kind"] == "over"
    displacement = abs(result["final_action"][0] - a_hat[0])
    assert displacement > 100.0  # must have moved substantially, not been a no-op


def test_pev_rated_power_bound():
    ev = PEVModel(id="pevtest2", alpha=3.5, beta=4, C_m=5, C_M=98, mu_c=0.9, mu_d=0.9,
                   p_rated_kw=9, E0=50, ET=50)
    T = 4
    e_demand = np.array([200.0, 200.0, -200.0, -200.0])  # absurd demand to force clipping
    eta = np.array([2.0, 2.0, 2.0, 2.0])
    lam = np.array([2.0, 2.0, 2.0, 2.0])
    xi = np.array([-0.9, -0.9, -0.9, -0.9])
    out = ev.simulate(e_demand, eta, lam, xi)
    assert np.all(np.abs(out["p_kw"]) <= ev.p_rated_kw + 1e-9)


def test_battery_soc_never_exceeds_capacity():
    bat = BatteryModel(id="b1", capacity_kwh=13.5, p_max_kw=5.0, eta_c=0.95, eta_d=0.95,
                        degradation_usd_per_kwh_throughput=0.02, soc0_kwh=13.0)
    soc = bat.soc0_kwh
    for _ in range(10):
        p, soc, deg, clipped = bat.step(soc, p_kw=5.0)  # keep trying to overcharge
        assert soc <= bat.capacity_kwh + 1e-9
        assert deg >= 0


def test_pricing_bounds_from_config(scenario):
    lo, hi = scenario.eta_bounds
    assert np.all(hi >= lo)
    assert np.all(lo == pytest.approx(1.5 * scenario.wholesale_price.min()))
    assert np.all(hi == pytest.approx(1.5 * scenario.wholesale_price.max()))


def test_cvar_ge_mean_for_losses():
    rng = np.random.default_rng(0)
    losses = rng.normal(10, 3, size=2000)
    mean_loss = losses.mean()
    c = metrics.cvar(losses, alpha=0.95)
    assert c >= mean_loss  # CVaR of the worst tail must be >= the mean


def test_jain_index_bounds():
    assert metrics.jain_index(np.array([1, 1, 1, 1])) == pytest.approx(1.0)  # perfectly fair
    j = metrics.jain_index(np.array([1, 0, 0, 0]))
    assert 0.24 < j < 0.26  # 1/n for one nonzero out of n=4


def test_gini_zero_for_equal_distribution():
    g = metrics.gini_coefficient(np.array([5.0, 5.0, 5.0, 5.0]))
    assert g == pytest.approx(0.0, abs=1e-9)


def test_jain_and_gini_reject_negative_input():
    """External review caught that these had no guard against signed
    inputs (e.g. net bill savings, which can be negative) -- both indices
    are only well-defined for non-negative allocations."""
    with pytest.raises(ValueError):
        metrics.jain_index(np.array([1.0, -2.0, 3.0]))
    with pytest.raises(ValueError):
        metrics.gini_coefficient(np.array([1.0, -2.0, 3.0]))


def test_feeder_topology_is_radial_tree(topo):
    # every non-root bus has exactly one parent -> |edges| == n_buses - 1
    assert len(topo.parent) == 33
    non_root_with_parent = sum(1 for b, p in topo.parent.items() if p is not None)
    assert non_root_with_parent == 32


def test_linearized_voltage_matches_ac_direction(topo):
    """Not bit-exact (that's the whole point of Command 6's 3-way
    feasibility distinction) but must agree in DIRECTION: adding load
    should reduce voltage in both models."""
    # Small, feeder-realistic increment (comparable to the original case33bw
    # per-bus loads of 0.045-0.2 MW); a multi-MW injection on 5 adjacent
    # buses is unrealistic for this feeder and makes Newton-Raphson diverge
    # (verified separately) -- that divergence is a modeling-scale issue,
    # not a bug, so we keep this test in the realistic operating region.
    buses = topo.non_slack_buses[:5]
    base_lin = linearized_voltages(topo, {}, {})
    loaded_lin = linearized_voltages(topo, {b: 0.05 for b in buses}, {b: 0.02 for b in buses})
    for b in buses:
        assert loaded_lin[b] < base_lin[b]

    base_ac = ac_powerflow_check(topo, {}, {})
    loaded_ac = ac_powerflow_check(topo, {b: 0.05 for b in buses}, {b: 0.02 for b in buses})
    assert base_ac["converged"] and loaded_ac["converged"]
    for b in buses:
        assert loaded_ac["vm_pu"][b] < base_ac["vm_pu"][b]


def test_safety_projection_respects_device_bounds(topo):
    monitored = topo.non_slack_buses[:6]
    devices = [Device(name=f"d{i}", bus=b, lb_kw=-5, ub_kw=5, weight=1.0)
               for i, b in enumerate(monitored)]
    a_hat = np.array([10.0, -10.0, 3.0, -3.0, 0.0, 4.9])  # first two exceed device bounds
    res = project(topo, monitored, devices, a_hat, fixed_bus_p_mw={}, homes_per_bus=1.0)
    assert np.all(res.a_star >= -5.0001)
    assert np.all(res.a_star <= 5.0001)
    assert res.solver_success


def test_safety_projection_enforces_voltage_floor(topo):
    """Force a voltage violation by loading heavily, then confirm the
    projection moves the action to relieve it (or reports slack usage --
    both are valid, honest outcomes; silently ignoring the violation is not)."""
    # Use buses deep in the radial tree (far from the substation along a
    # long lateral), where even moderate load causes real voltage drop --
    # buses right next to the root barely move (verified separately: a
    # bus adjacent to the slack needs several MW to breach +-5%, which is
    # not a realistic single-time-step household action).
    depths = sorted(((len(p), b) for b, p in topo.path_to_root.items()), reverse=True)
    monitored = [b for _, b in depths[:6]]
    devices = [Device(name=f"d{i}", bus=b, lb_kw=0, ub_kw=400, weight=1.0)
               for i, b in enumerate(monitored)]
    a_hat = np.array([300.0] * 6)  # deliberately huge candidate load, verified to breach v_min
    res = project(topo, monitored, devices, a_hat, fixed_bus_p_mw={}, v_min=0.95, v_max=1.05,
                   homes_per_bus=1.0)
    assert res.solver_success
    # either the projection pulled load down (a_star << a_hat) or flagged base infeasibility
    assert (res.projection_distance > 1.0) or res.joint_projection_uses_slack


def test_cohens_dz_paired_effect_size():
    """External review caught that using pooled independent-sample Cohen's
    d on paired data (same seeds across methods) is the wrong effect size;
    cohens_dz is the correct paired version."""
    from safe_pb import statistical_analysis as stats_mod
    x = np.array([10.0, 12.0, 11.0, 13.0, 9.0])
    y = np.array([9.0, 10.0, 10.5, 11.0, 8.0])   # consistently smaller, paired
    dz = stats_mod.cohens_dz(x, y)
    assert dz > 0  # x consistently exceeds y
    assert np.isfinite(dz)
    # cohens_dz and cohens_d should generally differ for the same data
    d_indep = stats_mod.cohens_d(x, y)
    assert dz != pytest.approx(d_indep)


def test_bus_assignment_covers_all_households(topo):
    hh_ids = [f"h{i}" for i in range(15)]
    mapping = assign_households_to_buses(topo, hh_ids, homes_per_bus=40, seed=1)
    assert set(mapping.keys()) == set(hh_ids)
    assert all(b in topo.non_slack_buses for b in mapping.values())


def test_projection_reports_before_after_and_residual(topo):
    """The projection must report linear voltage before/after and a
    non-negative violation measure, not just a pass/fail."""
    from safe_pb.safety_projection import Device, project
    monitored = topo.non_slack_buses  # ALL buses, not a subset
    devices = [Device(name="d0", bus=monitored[5], lb_kw=0, ub_kw=300, weight=1.0, q_over_p=0.35)]
    a_hat = np.array([250.0])
    res = project(topo, monitored, devices, a_hat, fixed_bus_p_mw={}, base_scale=0.5)
    assert set(res.linear_v_before.keys()) == set(monitored)
    assert set(res.linear_v_after.keys()) == set(monitored)
    assert np.isfinite(res.violation_before) and res.violation_before >= 0.0
    assert np.isfinite(res.violation_after) and res.violation_after >= 0.0
    assert np.isfinite(res.margin_before)  # signed, can be negative


def test_fixed_load_feasibility_checks_zero_controllable_case(topo):
    """base_infeasible (soft mode) only tested whether the penalty QP chose
    to use slack, conflating that with whether FIXED load alone (no
    controllable devices at all) is AC-feasible. This is the standalone,
    correctly-scoped check."""
    from safe_pb.safety_projection import fixed_load_feasibility
    result = fixed_load_feasibility(topo, {}, {}, v_min=0.95, v_max=1.05, base_scale=1.0)
    assert result["converged"]
    assert result["feasible"] is False
    assert result["min_vm_pu"] < 0.95
    result_calibrated = fixed_load_feasibility(topo, {}, {}, v_min=0.95, v_max=1.05, base_scale=0.5)
    assert result_calibrated["feasible"] is True


def test_hard_feasibility_mode_has_no_slack(topo):
    """allow_slack=False must produce zero slack by construction (no slack
    variables exist at all), distinguishing it from the soft penalty mode
    whose slack usage does NOT prove infeasibility."""
    from safe_pb.safety_projection import Device, project
    monitored = topo.non_slack_buses[:6]
    devices = [Device(name=f"d{i}", bus=b, lb_kw=0, ub_kw=50, weight=1.0) for i, b in enumerate(monitored)]
    a_hat = np.array([10.0] * 6)
    res = project(topo, monitored, devices, a_hat, fixed_bus_p_mw={}, allow_slack=False, base_scale=0.5)
    assert res.total_slack == 0.0
    assert not res.allow_slack


def test_min_slack_feasibility_reports_truly_feasible_case(topo):
    """The Phase-I minimize-total-slack problem is the correctly-scoped
    infeasibility test. Sanity check: a trivially satisfiable case (huge
    device authority, loose bounds) must report feasible with ~zero slack."""
    from safe_pb.safety_projection import Device, min_slack_feasibility
    monitored = topo.non_slack_buses[:4]
    devices = [Device(name=f"d{i}", bus=b, lb_kw=-1000, ub_kw=1000, weight=1.0) for i, b in enumerate(monitored)]
    result = min_slack_feasibility(topo, monitored, devices, fixed_bus_p_mw={}, base_scale=0.5)
    assert result["feasible"] is True
    assert result["min_total_slack"] < 1e-3


def test_max_min_voltage_authority_experiment(topo):
    """The 'available control authority' experiment must return a real
    action and a linear voltage prediction, independent of any candidate
    action -- used to distinguish objective-preference from physical
    inability."""
    from safe_pb.safety_projection import Device, max_min_voltage
    monitored = topo.non_slack_buses[:6]
    devices = [Device(name=f"d{i}", bus=b, lb_kw=-20, ub_kw=20, weight=1.0) for i, b in enumerate(monitored)]
    result = max_min_voltage(topo, monitored, devices, fixed_bus_p_mw={}, base_scale=0.5)
    assert result["solver_success"]
    assert len(result["a_star"]) == 6
    assert np.isfinite(result["best_achievable_min_v_linear"])


def test_inverter_modes_change_reactive_contribution(topo):
    """External review: reactive-power behavior must be configurable and
    stated, not a single hard-coded assumption. unity_pf must contribute
    zero Q; fixed_pf must scale with signed P; independent_q must accept
    its own bound without raising."""
    from safe_pb.safety_projection import Device, project
    monitored = topo.non_slack_buses[:3]
    a_hat = np.array([10.0])

    d_unity = Device(name="d", bus=monitored[0], lb_kw=-20, ub_kw=20, inverter_mode="unity_pf")
    res_unity = project(topo, monitored, [d_unity], a_hat, fixed_bus_p_mw={}, base_scale=0.5)
    assert res_unity.solver_success

    d_fixed = Device(name="d", bus=monitored[0], lb_kw=-20, ub_kw=20, inverter_mode="fixed_pf", q_over_p=0.35)
    res_fixed = project(topo, monitored, [d_fixed], a_hat, fixed_bus_p_mw={}, base_scale=0.5)
    assert res_fixed.solver_success

    d_indep = Device(name="d", bus=monitored[0], lb_kw=-20, ub_kw=20, inverter_mode="independent_q", s_max_kva=25.0)
    res_indep = project(topo, monitored, [d_indep], a_hat, fixed_bus_p_mw={}, base_scale=0.5)
    assert res_indep.solver_success
    # independent_q with a real s_max should allow a nonzero q bound
    assert d_indep.q_bound_kvar() > 0


def test_device_rejects_unknown_inverter_mode():
    from safe_pb.safety_projection import Device
    with pytest.raises(ValueError):
        Device(name="d", bus=1, lb_kw=0, ub_kw=10, inverter_mode="not_a_real_mode")


def test_independent_q_requires_s_max():
    from safe_pb.safety_projection import Device
    with pytest.raises(ValueError):
        Device(name="d", bus=1, lb_kw=0, ub_kw=10, inverter_mode="independent_q")  # missing s_max_kva


def test_iterative_ac_repair_succeeds_when_feasible(topo):
    """A mild candidate action on a lightly-loaded bus should be repaired
    (or already be AC-feasible) within a few iterations."""
    from safe_pb.safety_projection import Device, iterative_ac_repair
    devices = [Device(name="d0", bus=topo.non_slack_buses[0], lb_kw=0, ub_kw=100, weight=1.0)]
    a_hat = np.array([80.0])
    result = iterative_ac_repair(topo, topo.non_slack_buses, devices, a_hat,
                                  fixed_bus_p_mw={}, base_scale=0.5, max_iterations=5, trust_region_kw=20)
    assert result["outcome"] == "ac_verified_feasible"
    assert not result["fallback_activated"]
    assert len(result["history"]) >= 1


def test_iterative_ac_repair_activates_fallback_when_infeasible(topo):
    """When the device set genuinely cannot reach AC feasibility, the
    repair loop must exhaust its iterations and report an outcome that
    reflects whether the fallback itself is actually AC-feasible --
    'fallback_activated' only if it is, 'unresolved_infeasibility' if even
    the fallback still violates the limit (an earlier version collapsed
    both into 'fallback_activated' regardless, which reads as if a safe
    state was always reached)."""
    from safe_pb.safety_projection import Device, iterative_ac_repair
    depths = sorted(((len(p), b) for b, p in topo.path_to_root.items()), reverse=True)
    far_bus = depths[0][1]  # the deepest, most electrically remote bus
    devices = [Device(name="d0", bus=topo.non_slack_buses[0], lb_kw=0, ub_kw=20, weight=1.0)]
    fixed_p = {far_bus: 3.0}  # deliberately large fixed load far from the only controllable device
    result = iterative_ac_repair(topo, topo.non_slack_buses, devices, np.array([10.0]),
                                  fixed_bus_p_mw=fixed_p, base_scale=1.0, max_iterations=3, trust_region_kw=5.0)
    assert result["outcome"] in ("fallback_activated", "unresolved_infeasibility")
    assert result["fallback_activated"]  # the fallback was tried either way
    assert result["iterations_used"] <= 3
    assert "fallback_ac_feasible" in result
    # this specific extreme case should make even the fallback infeasible
    assert result["outcome"] == "unresolved_infeasibility"
    assert result["fallback_ac_feasible"] is False


def test_max_min_voltage_ac_smoke(topo):
    """Cheap smoke test (tiny DE budget) confirming max_min_voltage_ac runs
    end-to-end, logs AC evaluations, and reports a well-formed conclusion.
    Full-budget runs (used in the actual demo script) are much slower and
    not repeated here to keep the test suite fast."""
    from safe_pb.safety_projection import Device, max_min_voltage_ac
    devices = [Device(name="d0", bus=topo.non_slack_buses[0], lb_kw=-10, ub_kw=10, weight=1.0)]
    result = max_min_voltage_ac(topo, devices, fixed_bus_p_mw={}, base_scale=0.5,
                                 de_maxiter=2, de_popsize=4, n_multistart=1)
    assert result["n_ac_evaluations"] > 0
    assert "best_overall" in result
    assert result["conclusion"].startswith(("an AC-feasible", "no AC-feasible"))

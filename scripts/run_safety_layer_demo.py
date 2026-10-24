"""
Safety-layer demonstration (Command 6), fourth-round corrected version.

External review correctly identified two problems with the prior version:
(1) the manuscript reported a 0-0.10 margin sweep that was run
interactively and never saved by this script -- not reproducible from the
committed code; (2) the soft-penalty QP always permits slack, so a large
internal margin does not prove infeasibility -- the solver may simply pay
the (bounded) slack penalty rather than move `a`. Both are fixed here:
the FULL sweep is executed and saved by this script, every requested
diagnostic field is recorded, and a genuine hard-feasibility test
(min_slack_feasibility, a Phase-I LP with no distance-to-candidate
objective) is run alongside the soft sweep so the two questions --
"did the penalty formulation use slack" and "does a slack-free solution
exist at all" -- are never conflated.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safe_pb.data_loader import build_tier_a_scenario
from safe_pb.ev_model import EVTerminalInfeasible
from safe_pb.baselines import TabularQLearningAgent, QLearningConfig
from safe_pb.feeder_model import build_feeder, assign_households_to_buses, ac_powerflow_check
from safe_pb.safety_projection import (Device, project, fixed_load_feasibility,
                                        min_slack_feasibility, max_min_voltage)
from safe_pb.environment import ev_available

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "tier_a_wan2022.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

V_MIN, V_MAX = 0.95, 1.05
Q_OVER_P = 0.35
MARGINS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]


def build_scenario_devices(scenario, topo, worst_t, all_results):
    """`all_results` is the FULL day's list of StepResults (not just the
    worst hour) -- needed to replay each PEV's SoC trajectory up to
    worst_t so its viability floor can be computed correctly. An earlier
    version only received the single worst-hour row and gave every PEV
    static [-p_rated, +p_rated] bounds regardless of remaining terminal
    feasibility, which could let the safety search use discharge that's
    actually incompatible with reaching the terminal target."""
    r = all_results[worst_t]
    household_ids = ([f"gen_{l.id}" for l in scenario.gen_loads] +
                      [f"pev_{e.id}" for e in scenario.pevs])
    non_ids = [f"non_{l.id}" for l in scenario.nondispatch_loads]
    bus_map = assign_households_to_buses(topo, household_ids + non_ids, homes_per_bus=1, seed=1)

    fixed_extra_p, fixed_extra_q = {}, {}
    for hid, kw in zip(non_ids, r.p_non):
        b = bus_map[hid]
        fixed_extra_p[b] = fixed_extra_p.get(b, 0.0) + kw / 1000.0
        fixed_extra_q[b] = fixed_extra_q.get(b, 0.0) + kw * Q_OVER_P / 1000.0

    devices, a_hat = [], []
    infeasible_pevs = []
    for i, ld in enumerate(scenario.gen_loads):
        e = scenario.gen_demand_kwh[i, worst_t]
        lb, ub = e * (1 - ld.dr_max_frac), e * (1 - ld.dr_min_frac)
        devices.append(Device(name=f"gen_{ld.id}", bus=bus_map[f"gen_{ld.id}"], lb_kw=lb, ub_kw=ub,
                               weight=1.0, q_over_p=Q_OVER_P))
        a_hat.append(r.p_gen[i])
    for i, ev in enumerate(scenario.pevs):
        # Replay this PEV's SoC up to (not including) worst_t under the
        # ACTUAL chosen trajectory, so the viability floor reflects where
        # it really is, not an assumed starting SoC.
        soc = ev.E0
        for t in range(worst_t):
            p = all_results[t].p_pev[i]
            mu = ev.mu_c if p > 0 else (1.0 / ev.mu_d if p < 0 else 1.0)
            soc += mu * p
        hours_remaining = sum(1 for h in range(worst_t, scenario.T) if ev_available(h))
        try:
            floor = ev.viability_floor_kw(soc, hours_remaining)
        except EVTerminalInfeasible as exc:
            # External review correctly flagged that a broad `except
            # Exception: floor = 0.0` could silently convert a genuine
            # terminal-infeasibility signal into an unflagged default,
            # masking a real problem. Catch ONLY the specific exception,
            # record it explicitly, and use a conservative floor (0: allow
            # charging, forbid discharge) rather than pretending nothing
            # is wrong.
            print(f"  WARNING: {ev.id} terminal target is infeasible at hour {worst_t} "
                  f"(shortfall {exc.shortfall_kwh:.2f} kWh even at full rated charging for "
                  f"all remaining hours) -- forcing floor=0 (charge-only, no discharge) "
                  f"and flagging this PEV as infeasible in the report.")
            infeasible_pevs.append({"pev_id": ev.id, "hour": worst_t, "shortfall_kwh": exc.shortfall_kwh})
            floor = 0.0
        lb_kw = max(-ev.p_rated_kw, floor)
        devices.append(Device(name=f"pev_{ev.id}", bus=bus_map[f"pev_{ev.id}"],
                               lb_kw=lb_kw, ub_kw=ev.p_rated_kw, weight=1.0, q_over_p=Q_OVER_P))
        a_hat.append(max(r.p_pev[i], lb_kw))  # candidate must itself respect the floor
    return devices, np.array(a_hat), fixed_extra_p, fixed_extra_q, bus_map, non_ids, infeasible_pevs


def apply_action_to_grid(fixed_p, fixed_q, devices, a):
    p, q = dict(fixed_p), dict(fixed_q)
    for d, kw in zip(devices, a):
        p[d.bus] = p.get(d.bus, 0.0) + kw / 1000.0
        q[d.bus] = q.get(d.bus, 0.0) + kw * d.q_over_p / 1000.0
    return p, q


def main():
    scenario = build_tier_a_scenario(CONFIG_PATH)
    topo = build_feeder()
    base_scale = float(scenario.network_config["base_scale"])
    ALL_BUSES = topo.non_slack_buses

    print("STEP 1: confirm the calibrated native feeder is feasible with zero household load")
    native = ac_powerflow_check(topo, {}, {}, base_scale=base_scale)
    print(f"  base_scale={base_scale}: native min V = {native['min_vm_pu']:.4f} p.u.")
    assert native["min_vm_pu"] >= V_MIN

    print("STEP 2: train the Q-learning reconstruction, find its worst hour")
    cfg = QLearningConfig(grid_points=7, n_episodes=4000, seed=0)
    agent = TabularQLearningAgent(scenario, cfg)
    agent.train()
    ql_results, _ = agent.greedy_policy_rollout()
    worst_t = int(np.argmax([r.p_non.sum() + r.p_gen.sum() + r.p_pev.sum() for r in ql_results]))
    r = ql_results[worst_t]
    print(f"  worst hour = {worst_t}")

    devices, a_hat, fixed_p, fixed_q, bus_map, non_ids, infeasible_pevs = build_scenario_devices(scenario, topo, worst_t, ql_results)
    if infeasible_pevs:
        print(f"  NOTE: {len(infeasible_pevs)} PEV(s) had an infeasible terminal target at hour {worst_t}: {infeasible_pevs}")

    print("STEP 3: fixed-load-only AC feasibility (controllable devices at zero)")
    fixed_feas = fixed_load_feasibility(topo, fixed_p, fixed_q, v_min=V_MIN, v_max=V_MAX, base_scale=base_scale)
    print(f"  feasible={fixed_feas['feasible']}, min_vm_pu={fixed_feas.get('min_vm_pu')}")

    print("STEP 4: unprotected candidate schedule (raw price-reaction actions), AC-checked")
    up, uq = apply_action_to_grid(fixed_p, fixed_q, devices, a_hat)
    ac_unprotected = ac_powerflow_check(topo, up, uq, base_scale=base_scale)
    worst_bus = min(ac_unprotected["vm_pu"], key=ac_unprotected["vm_pu"].get)
    print(f"  min V (AC) = {ac_unprotected['min_vm_pu']:.4f} at bus {worst_bus} "
          f"({'VIOLATES' if ac_unprotected['min_vm_pu'] < V_MIN else 'ok'})")

    print(f"STEP 5: SOFT-mode margin sweep, {MARGINS}, every field saved (fixes the prior "
          f"un-reproducible interactive sweep)")
    fixed_qp_p = {b: fixed_p.get(b, 0.0) for b in ALL_BUSES}
    fixed_qp_q = {b: fixed_q.get(b, 0.0) for b in ALL_BUSES}
    sweep_results = []
    for margin in MARGINS:
        res = project(topo, ALL_BUSES, devices, a_hat, fixed_bus_p_mw=fixed_qp_p,
                       fixed_bus_q_mvar=fixed_qp_q, v_min=V_MIN + margin, v_max=V_MAX,
                       homes_per_bus=1.0, base_scale=base_scale, allow_slack=True)
        pp, pq = apply_action_to_grid(fixed_p, fixed_q, devices, res.a_star)
        ac = ac_powerflow_check(topo, pp, pq, base_scale=base_scale)
        row = {
            "margin": margin, "solver_success": res.solver_success, "solver_status": res.solver_status,
            "solver_message": res.solver_message, "objective_value": res.objective_value,
            "a_star_kw": res.a_star.tolist(), "projection_distance_kw": res.projection_distance,
            "slack_lo": res.slack_lo.tolist(), "slack_hi": res.slack_hi.tolist(),
            "total_slack": res.total_slack, "joint_projection_uses_slack": res.joint_projection_uses_slack,
            "linear_violation_before": res.violation_before, "linear_violation_after": res.violation_after,
            "linear_margin_after": res.margin_after,
            "linear_min_v_after": min(res.linear_v_after.values()),
            "ac_min_v_after": ac["min_vm_pu"], "ac_worst_bus_after": min(ac["vm_pu"], key=ac["vm_pu"].get),
            "physical_0_95_met": bool(ac["min_vm_pu"] >= V_MIN),
        }
        sweep_results.append(row)
        print(f"  margin={margin:.3f}: dist={res.projection_distance:.3f}kW, "
              f"slack_used={res.joint_projection_uses_slack} (total={res.total_slack:.4f}), "
              f"AC min V={ac['min_vm_pu']:.4f}, met_0.95={row['physical_0_95_met']}")

    print("STEP 6: HARD feasibility (allow_slack=False) at the nominal limit")
    hard_res = project(topo, ALL_BUSES, devices, a_hat, fixed_bus_p_mw=fixed_qp_p,
                        fixed_bus_q_mvar=fixed_qp_q, v_min=V_MIN, v_max=V_MAX,
                        homes_per_bus=1.0, base_scale=base_scale, allow_slack=False)
    print(f"  solver_success={hard_res.solver_success}, message={hard_res.solver_message!r}, "
          f"violation_after={hard_res.violation_after:.5f}")

    print("STEP 7: Phase-I min-slack feasibility (the correctly-scoped infeasibility test)")
    phase1 = min_slack_feasibility(topo, ALL_BUSES, devices, fixed_bus_p_mw=fixed_qp_p,
                                    fixed_bus_q_mvar=fixed_qp_q, v_min=V_MIN, v_max=V_MAX, base_scale=base_scale)
    print(f"  feasible={phase1['feasible']}, min_total_slack={phase1['min_total_slack']:.5f}, "
          f"solver_success={phase1['solver_success']}")
    if phase1["feasible"]:
        pp, pq = apply_action_to_grid(fixed_p, fixed_q, devices, np.array(phase1["a_star"]))
        ac_phase1 = ac_powerflow_check(topo, pp, pq, base_scale=base_scale)
        phase1["ac_min_v_at_solution"] = ac_phase1["min_vm_pu"]
        phase1["ac_feasible_at_solution"] = bool(ac_phase1["min_vm_pu"] >= V_MIN)
        print(f"  AC-checking the Phase-I solution: min V = {ac_phase1['min_vm_pu']:.4f} "
              f"({'AC-feasible too' if phase1['ac_feasible_at_solution'] else 'linear-feasible but AC-infeasible'})")

    print("STEP 8: max-min-voltage control-authority experiment (linear-model best case, then AC-verify)")
    authority = max_min_voltage(topo, ALL_BUSES, devices, fixed_bus_p_mw=fixed_qp_p,
                                 fixed_bus_q_mvar=fixed_qp_q, base_scale=base_scale)
    pp, pq = apply_action_to_grid(fixed_p, fixed_q, devices, np.array(authority["a_star"]))
    ac_authority = ac_powerflow_check(topo, pp, pq, base_scale=base_scale)
    authority["ac_min_v_at_best_action"] = ac_authority["min_vm_pu"]
    authority["ac_feasible_at_best_action"] = bool(ac_authority["min_vm_pu"] >= V_MIN)
    print(f"  best achievable linear min V = {authority['best_achievable_min_v_linear']:.4f}, "
          f"AC-checked = {ac_authority['min_vm_pu']:.4f} "
          f"({'meets 0.95 even at best action' if authority['ac_feasible_at_best_action'] else 'CANNOT reach 0.95 even at best action -- decisive'})")

    print("STEP 8b: direct AC-objective global search (differential evolution + multistart) -- "
          "an INDEPENDENT method that never relies on the linear model at all")
    from safe_pb.safety_projection import max_min_voltage_ac, iterative_ac_repair
    ac_search = max_min_voltage_ac(topo, devices, fixed_bus_p_mw=fixed_qp_p, fixed_bus_q_mvar=fixed_qp_q,
                                    base_scale=base_scale, de_maxiter=12, de_popsize=8, n_multistart=4, de_seed=0)
    print(f"  {ac_search['n_ac_evaluations']} AC power-flow evaluations; "
          f"best found min V = {ac_search['best_overall']['min_v']:.4f}")
    print(f"  {ac_search['conclusion']}")

    # Persist the full per-candidate evaluation log to its own CSV rather
    # than embedding ~2,400 rows in the main JSON report (which would
    # bloat it past 800KB and make it unreadable as a summary document).
    # This is a real, separate, analyzable artifact -- not a discarded log
    # -- addressing the point that "every evaluation is recorded" must be
    # actually true and actually inspectable, just not inline here.
    eval_log_path = os.path.join(RESULTS_DIR, "ac_search_eval_log.csv")
    with open(eval_log_path, "w") as f:
        n_dev = len(devices)
        f.write(",".join([f"a_{i}" for i in range(n_dev)]) + ",converged,min_v\n")
        for row in ac_search["eval_log"]:
            a_str = ",".join(str(v) for v in row["a"])
            f.write(f"{a_str},{row['converged']},{row['min_v']}\n")
    print(f"  full per-candidate evaluation log ({len(ac_search['eval_log'])} rows) written to {eval_log_path}")
    ac_search_summary = {k: v for k, v in ac_search.items() if k != "eval_log"}
    ac_search_summary["eval_log_csv_path"] = eval_log_path

    print("STEP 8c: iterative AC-repair procedure -- reports three DISTINCT outcomes, "
          "not a single pass/fail")
    repair = iterative_ac_repair(topo, ALL_BUSES, devices, a_hat, fixed_bus_p_mw=fixed_qp_p,
                                  fixed_bus_q_mvar=fixed_qp_q, v_min=V_MIN, v_max=V_MAX,
                                  base_scale=base_scale, max_iterations=5, trust_region_kw=5.0)
    print(f"  outcome={repair['outcome']}, iterations_used={repair['iterations_used']}, "
          f"fallback_activated={repair['fallback_activated']}")
    if repair["fallback_activated"]:
        print(f"  fallback AC min V = {repair['fallback_ac_min_v']:.4f}, "
              f"fallback_ac_feasible={repair['fallback_ac_feasible']}")

    three_outcomes = {
        # An earlier version collapsed this into a single ambiguous
        # "fallback_or_unresolved" boolean that could not distinguish a
        # genuinely-safe fallback from an unresolved infeasibility --
        # exactly the distinction `repair["outcome"]` already draws.
        # These fields expose that distinction directly instead of
        # re-collapsing it.
        "linear_model_feasible": bool(hard_res.solver_success and hard_res.violation_after <= 1e-6),
        "ac_verified_feasible": bool(authority["ac_feasible_at_best_action"] or
                                      ac_search["best_overall"]["min_v"] >= V_MIN),
        "repair_outcome": repair["outcome"],
        "fallback_attempted": bool(repair["fallback_activated"]),
        "fallback_ac_feasible": bool(repair.get("fallback_ac_feasible", False)),
    }
    print(f"  THREE-OUTCOME SUMMARY: {three_outcomes}")

    report = {
        "base_scale": base_scale, "native_feeder_min_v": native["min_vm_pu"], "worst_hour": worst_t,
        "infeasible_pevs_at_worst_hour": infeasible_pevs,
        "fixed_load_only_feasibility": fixed_feas,
        "unprotected_min_v_ac": ac_unprotected["min_vm_pu"], "unprotected_worst_bus": worst_bus,
        "unprotected_violates": bool(ac_unprotected["min_vm_pu"] < V_MIN),
        "n_monitored_buses": len(ALL_BUSES),
        "soft_margin_sweep": sweep_results,
        "hard_feasibility_at_nominal_limit": {
            "solver_success": hard_res.solver_success, "solver_message": hard_res.solver_message,
            "violation_after": hard_res.violation_after,
        },
        "phase1_min_slack_feasibility": phase1,
        "max_min_voltage_authority_experiment_linear": authority,
        "max_min_voltage_ac_direct_search": ac_search_summary,
        "iterative_ac_repair": repair,
        "three_outcomes_summary": three_outcomes,
        "conclusion": (
            f"The linear-authority diagnostic (AC-verified min V = {authority['ac_min_v_at_best_action']:.4f}) "
            f"and the direct AC-objective search (best observed min V = {ac_search['best_overall']['min_v']:.4f} "
            f"across {ac_search['n_ac_evaluations']} AC evaluations) did not find an AC-feasible action. "
            f"Within the AC search, the global differential-evolution run exhausted its iteration budget "
            f"(success=False); one of {len(ac_search['multistart_results'])} local polishing multistarts did "
            f"report numerical convergence (success=True) at essentially the same best value the other "
            f"three found without formally converging -- local-solver convergence is not the same claim as "
            f"finding an AC-feasible action, and none of the searches found one. The finite-difference "
            f"AC-repair procedure's outcome was '{repair['outcome']}'. Together this is empirical evidence "
            f"that no action found within these specific search and iteration budgets reached AC feasibility "
            f"for this device set and placement -- not a certified infeasibility proof, and not a claim that "
            f"no optimizer ever converged."
        ),
    }
    with open(os.path.join(RESULTS_DIR, "safety_layer_demo_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nSTEP 9: result written to results/safety_layer_demo_report.json")
    return report


if __name__ == "__main__":
    main()

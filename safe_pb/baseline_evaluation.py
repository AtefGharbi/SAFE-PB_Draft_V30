"""Common held-out evaluation for non-learning and classical baselines.

All comparators use the same generated scenario, feeder mapping, AC check,
metric definitions, and scenario labels as SAFE-PB. No baseline result is
copied from a separate validation run.
"""
from __future__ import annotations

import numpy as np

from .episode_metrics import summarize_episode
from .feeder_model import (ac_powerflow_check, assign_households_to_buses,
                           linearized_voltages)


def summarize_baseline_episode(scenario, topo, step_results, prices,
                               scenario_label: str, base_scale: float,
                               homes_per_bus: float, v_min: float = 0.95,
                               v_max: float = 1.05) -> dict:
    """Convert baseline trajectories into the common episode schema."""
    if len(step_results) != scenario.T:
        raise ValueError("baseline rollout must contain one result per time slot")
    if prices is not None and len(prices) != scenario.T:
        raise ValueError("baseline price schedule length does not match the scenario")

    household_ids = ([f"gen_{x.id}" for x in scenario.gen_loads] +
                     [f"pev_{x.id}" for x in scenario.pevs] +
                     [f"bat_{x.id}" for x in scenario.batteries])
    non_ids = [f"non_{x.id}" for x in scenario.nondispatch_loads]
    pv_ids = [f"pv_{i}" for i in range(len(scenario.pv_profiles_kw))]
    bus_map = assign_households_to_buses(
        topo, household_ids + non_ids + pv_ids, homes_per_bus, seed=1)

    infos = []
    pev_soc = np.array([ev.E0 for ev in scenario.pevs], dtype=float)
    # V23: the authority certificate depends only on device bounds and the
    # exogenous injections, so it is evaluated with the SAFE-PB environment's
    # own device model at the baseline's realised PEV state of charge.
    from .environment import SafePBEnv, EnvConfig
    from .safety_projection import authority_certificate
    cert_env = SafePBEnv(scenario, topo, EnvConfig(
        base_scale=base_scale, homes_per_bus=homes_per_bus, v_min=v_min, v_max=v_max,
        pev_infeasibility_policy="continue_and_report"))
    for t, result in enumerate(step_results):
        cert_env.t = t
        cert_env.pev_soc = pev_soc.copy()
        _devs, _, _fp, _fq = cert_env.build_safety_devices(
            np.zeros(cert_env.n_households))
        cert = authority_certificate(topo, _devs, _fp, _fq, homes_per_bus=homes_per_bus,
                                     base_scale=base_scale, v_min=v_min)
        device_kw = np.concatenate([
            np.asarray(result.p_gen, float), np.asarray(result.p_pev, float),
            np.zeros(len(scenario.batteries), dtype=float)])
        bus_p, bus_q = {}, {}
        for hid, kw in zip(household_ids, device_kw):
            bus = bus_map[hid]
            bus_p[bus] = bus_p.get(bus, 0.0) + kw * homes_per_bus / 1000.0
            bus_q[bus] = bus_q.get(bus, 0.0) + kw * 0.35 * homes_per_bus / 1000.0
        for i, hid in enumerate(non_ids):
            kw = float(result.p_non[i])
            bus = bus_map[hid]
            bus_p[bus] = bus_p.get(bus, 0.0) + kw * homes_per_bus / 1000.0
            bus_q[bus] = bus_q.get(bus, 0.0) + kw * 0.35 * homes_per_bus / 1000.0
        for i, hid in enumerate(pv_ids):
            bus = bus_map[hid]
            bus_p[bus] = (bus_p.get(bus, 0.0) -
                          float(scenario.pv_profiles_kw[i, t]) * homes_per_bus / 1000.0)

        linear = linearized_voltages(topo, bus_p, bus_q, base_scale=base_scale)
        ac = ac_powerflow_check(topo, bus_p, bus_q, base_scale=base_scale)
        min_ac = float(ac.get("min_vm_pu", float("nan"))) if ac.get("converged") else float("nan")
        max_ac = float(ac.get("max_vm_pu", float("nan"))) if ac.get("converged") else float("nan")
        ac_bad = bool(ac.get("converged") and (min_ac < v_min or max_ac > v_max))

        p_gen = np.asarray(result.p_gen, float)
        p_pev = np.asarray(result.p_pev, float)
        discomfort = np.array([
            0.5 * load.alpha * (scenario.gen_demand_kwh[i, t] - p_gen[i]) ** 2 +
            load.beta * (scenario.gen_demand_kwh[i, t] - p_gen[i])
            for i, load in enumerate(scenario.gen_loads)], dtype=float)
        pev_discomfort = np.array([
            0.5 * ev.alpha * (scenario.pev_demand_kwh[i, t] - p_pev[i]) ** 2 +
            ev.beta * (scenario.pev_demand_kwh[i, t] - p_pev[i])
            for i, ev in enumerate(scenario.pevs)], dtype=float)
        for i, ev in enumerate(scenario.pevs):
            p = p_pev[i]
            efficiency = ev.mu_c if p > 0 else (1.0 / ev.mu_d if p < 0 else 1.0)
            pev_soc[i] = np.clip(pev_soc[i] + efficiency * p, ev.C_m, ev.C_M)

        if prices is None:
            eta_gen = eta_pev = float(scenario.wholesale_price[t])
        else:
            _eta_non, eta_gen, eta_pev = map(float, prices[t])
        pev_bill = np.where(p_pev >= 0.0, eta_pev * p_pev,
                            scenario.wholesale_price[t] * p_pev)
        infos.append({
            "U_t": float(result.U_t), "C_t": float(result.C_t),
            "p_non": np.asarray(result.p_non, float), "p_gen": p_gen,
            "p_pev": p_pev, "p_bat": np.zeros(len(scenario.batteries)),
            "household_discomfort_all": np.concatenate([
                discomfort, pev_discomfort, np.zeros(len(scenario.batteries))]),
            "fairness_discomfort": np.concatenate([discomfort, pev_discomfort]),
            "household_energy_bill": np.concatenate([
                eta_gen * p_gen, pev_bill, np.zeros(len(scenario.batteries))]),
            "candidate_actions_kw": device_kw.copy(),
            "applied_actions_kw": device_kw.copy(),
            "projection_delta_kw": np.zeros_like(device_kw),
            "projection_distance": 0.0, "projection_active": False,
            "joint_projection_uses_slack": False, "total_slack": 0.0,
            "candidate_min_voltage_linear": float(min(linear.values())),
            "candidate_min_voltage_ac": min_ac,
            "candidate_ac_converged": bool(ac.get("converged", False)),
            "candidate_ac_safety_violated": ac_bad,
            "min_voltage_linear": float(min(linear.values())),
            "min_voltage_ac": min_ac,
            "ac_converged": bool(ac.get("converged", False)),
            "ac_safety_violated": ac_bad,
            "authority_feasible": cert["authority_feasible"],
            "episode_terminated_for_pev_infeasibility": False,
            "scenario_label": scenario_label, "base_scale": base_scale,
            "homes_per_bus": homes_per_bus,
        })
    terminal_shortfall = np.maximum(
        np.array([ev.ET for ev in scenario.pevs]) - pev_soc, 0.0)
    infos[-1]["terminal_pev_shortfall_kwh"] = terminal_shortfall
    infos[-1]["episode_terminated_for_pev_infeasibility"] = bool(
        np.any(terminal_shortfall > 1e-6))
    return summarize_episode(infos, scenario)

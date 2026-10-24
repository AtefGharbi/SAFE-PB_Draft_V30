"""Find and verify a controlled before/after safety-projection case.

The search accepts only cases where fixed exogenous load is AC-feasible,
the unprotected controllable candidate violates the voltage band, the
projection moves the action, and the projected schedule is AC-feasible.
If no such case exists for the declared search grid, the script fails rather
than presenting an already-infeasible feeder as evidence of protection.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safe_pb.data_loader import build_tier_a_scenario
from safe_pb.environment import SafePBEnv, EnvConfig
from safe_pb.feeder_model import build_feeder, ac_powerflow_check

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT, "config", "tier_a_wan2022.yaml")
OUTPUT_PATH = os.path.join(ROOT, "results", "safety_effectiveness_demo.json")


def ac_for_actions(topo, env, devices, action_kw, fixed_p, fixed_q):
    p, q = dict(fixed_p), dict(fixed_q)
    for device, kw in zip(devices, action_kw):
        scale = env.cfg.homes_per_bus / 1000.0
        p[device.bus] = p.get(device.bus, 0.0) + float(kw) * scale
        q[device.bus] = q.get(device.bus, 0.0) + float(kw) * device.q_over_p * scale
    return ac_powerflow_check(topo, p, q, base_scale=env.cfg.base_scale)


def main():
    scenario = build_tier_a_scenario(CONFIG_PATH)
    topo = build_feeder()
    configured_scale = float(scenario.network_config["base_scale"])
    found = None
    attempts = []

    for homes in np.arange(1.0, 4.01, 0.05):
        for margin in (0.0, 0.002, 0.005, 0.01):
            for hour in range(0, 9):
                cfg = EnvConfig(base_scale=configured_scale,
                                homes_per_bus=float(homes),
                                v_min=0.95 + margin,
                                pev_infeasibility_policy="continue_and_report")
                env = SafePBEnv(scenario, topo, cfg)
                env.t = hour
                proposed = np.full(env.n_households, 1e6)
                devices, candidate, fixed_p, fixed_q = env.build_safety_devices(proposed)
                fixed_ac = ac_powerflow_check(topo, fixed_p, fixed_q,
                                              base_scale=configured_scale)
                unprotected = ac_for_actions(topo, env, devices, candidate, fixed_p, fixed_q)
                if not (fixed_ac["converged"] and fixed_ac["min_vm_pu"] >= 0.95 and
                        unprotected["converged"] and unprotected["min_vm_pu"] < 0.95):
                    continue

                env.utility_announce_price(np.zeros(3))
                _, _, _, _, info = env.household_step(
                    np.ones((env.n_households, 1))
                )
                attempt = {
                    "base_scale": configured_scale,
                    "homes_per_bus_stress_multiplier": float(homes),
                    "hour": hour,
                    "internal_linear_margin_pu": margin,
                    "fixed_only_min_voltage_ac_pu": float(fixed_ac["min_vm_pu"]),
                    "unprotected_min_voltage_ac_pu": float(unprotected["min_vm_pu"]),
                    "projected_min_voltage_linear_pu": float(info["min_voltage_linear"]),
                    "projected_min_voltage_ac_pu": float(info["min_voltage_ac"]),
                    "projection_distance_kw": float(info["projection_distance"]),
                    "total_slack": float(info["total_slack"]),
                    "projected_ac_safe": bool(not info["ac_safety_violated"]),
                }
                attempts.append(attempt)
                if (attempt["projection_distance_kw"] > 1e-9 and
                        attempt["projected_ac_safe"]):
                    found = attempt
                    break
            if found:
                break
        if found:
            break

    report = {
        "generated_by": "scripts/run_safety_effectiveness_demo.py",
        "acceptance_criteria": {
            "fixed_exogenous_ac_feasible": True,
            "unprotected_candidate_ac_violating": True,
            "nonzero_projection_movement": True,
            "projected_schedule_ac_feasible": True,
        },
        "status": "passed" if found else "no_restorative_case_found",
        "accepted_case": found,
        "qualifying_attempts": attempts,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    if found is None:
        raise RuntimeError("no controlled AC-restorative projection case found")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()

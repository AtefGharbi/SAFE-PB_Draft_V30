"""Reproduce the feeder calibration from the single YAML configuration."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safe_pb.data_loader import build_tier_a_scenario
from safe_pb.environment import SafePBEnv, EnvConfig
from safe_pb.feeder_model import build_feeder, ac_powerflow_check

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT, "config", "tier_a_wan2022.yaml")
OUTPUT_PATH = os.path.join(ROOT, "results", "network_calibration.json")


def main():
    scenario = build_tier_a_scenario(CONFIG_PATH)
    topo = build_feeder()
    network = scenario.network_config
    target = float(network["exogenous_headroom_target_pu"])
    configured = float(network["base_scale"])
    rows = []
    for scale in network["calibration_candidates"]:
        env = SafePBEnv(scenario, topo, EnvConfig(base_scale=float(scale)))
        worst = 1.0
        worst_hour = None
        for hour in range(scenario.T):
            env.t = hour
            _, _, fixed_p, fixed_q = env.build_safety_devices(
                [0.0] * env.n_households
            )
            ac = ac_powerflow_check(topo, fixed_p, fixed_q, base_scale=float(scale))
            if ac["converged"] and ac["min_vm_pu"] < worst:
                worst = float(ac["min_vm_pu"])
                worst_hour = hour
        rows.append({"base_scale": float(scale), "worst_hour": worst_hour,
                     "exogenous_only_worst_min_voltage_ac_pu": worst,
                     "meets_headroom_target": bool(worst >= target)})

    eligible = [r["base_scale"] for r in rows if r["meets_headroom_target"]]
    selected = max(eligible) if eligible else None
    report = {
        "generated_by": "scripts/run_network_calibration.py",
        "selection_rule": "largest configured candidate with exogenous-only worst-hour AC minimum >= target",
        "headroom_target_pu": target,
        "configured_base_scale": configured,
        "selected_base_scale": selected,
        "selection_matches_configuration": selected == configured,
        "sweep": rows,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    if selected != configured:
        raise RuntimeError(
            f"configured base_scale={configured} does not match calibrated value {selected}"
        )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()

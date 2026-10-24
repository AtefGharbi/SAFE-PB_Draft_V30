"""V23 safety re-validation on the original 168-case controlled search grid.

Reproduces the V21/V22 search entry criteria exactly (feasible exogenous load,
AC-violating maximum-consumption candidate) and evaluates, for every case:
  soft          legacy penalty QP (V21/V22)
  lexicographic min-slack LP then min-distance QP, no AC correction
  ac_corrected  lexicographic + AC-measured bias correction (V23 default)
plus the authority certificate (is ANY device-feasible action AC-feasible?).
Outputs results/safety_revalidation_cases.csv and results/safety_revalidation_summary.json.
"""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from safe_pb.data_loader import build_tier_a_scenario
from safe_pb.environment import SafePBEnv, EnvConfig
from safe_pb.feeder_model import build_feeder, ac_powerflow_check
from safe_pb.safety_projection import (project, lexicographic_project, ac_corrected_project,
                                       authority_certificate, AC_FEASIBILITY_TOL)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results")


def inject(env, devices, a, fp, fq):
    return env._network_injections(devices, a, fp, fq)


def main():
    sc = build_tier_a_scenario(os.path.join(ROOT, "config", "tier_a_wan2022.yaml"))
    topo = build_feeder()
    scale = float(sc.network_config["base_scale"])
    buses = topo.non_slack_buses
    rows = []
    for homes in np.arange(1.0, 4.01, 0.05):
        for margin in (0.0, 0.002, 0.005, 0.01):
            for hour in range(0, 9):
                cfg = EnvConfig(base_scale=scale, homes_per_bus=float(homes), v_min=0.95 + margin,
                                pev_infeasibility_policy="continue_and_report")
                env = SafePBEnv(sc, topo, cfg); env.t = hour
                devices, cand, fp, fq = env.build_safety_devices(np.full(env.n_households, 1e6))
                fixed_ac = ac_powerflow_check(topo, fp, fq, base_scale=scale)
                cand_ac = ac_powerflow_check(topo, *inject(env, devices, cand, fp, fq), base_scale=scale)
                if not (fixed_ac["converged"] and fixed_ac["min_vm_pu"] >= 0.95 and
                        cand_ac["converged"] and cand_ac["min_vm_pu"] < 0.95):
                    continue
                r = dict(homes_per_bus=round(float(homes), 2), internal_margin_pu=margin, hour=hour,
                         candidate_min_v_ac=cand_ac["min_vm_pu"])
                cert = authority_certificate(topo, devices, fp, fq, homes, scale, 0.95)
                r["authority_feasible"] = cert["authority_feasible"]
                r["authority_best_min_v_ac"] = cert["best_min_vm_pu"]
                t0 = time.perf_counter()
                soft = project(topo, buses, devices, cand, fp, 0.95 + margin, 1.05, homes, scale, fq)
                r["soft_time_s"] = time.perf_counter() - t0
                lex = lexicographic_project(topo, buses, devices, cand, fp, fq, 0.95, 1.05, homes, scale)
                t0 = time.perf_counter()
                acc = ac_corrected_project(topo, buses, devices, cand, fp, fq, 0.95, 1.05, homes, scale)
                r["ac_corrected_time_s"] = time.perf_counter() - t0
                for name, a in (("soft", soft.a_star), ("lexicographic", lex["a_star"]), ("ac_corrected", acc["a_star"])):
                    ac = ac_powerflow_check(topo, *inject(env, devices, a, fp, fq), base_scale=scale)
                    r[f"{name}_min_v_ac"] = ac["min_vm_pu"]
                    r[f"{name}_ac_feasible"] = bool(ac["min_vm_pu"] >= 0.95 - AC_FEASIBILITY_TOL)
                    r[f"{name}_distance_kw"] = float(np.linalg.norm(a - cand))
                r["soft_total_slack"] = soft.total_slack
                r["lexicographic_min_slack"] = lex["min_total_slack"]
                r["ac_corrections_used"] = acc["corrections_used"]
                rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "safety_revalidation_cases.csv"), index=False)
    fa = df.authority_feasible
    summary = {"n_cases": len(df), "n_authority_feasible": int(fa.sum()),
               "n_authority_infeasible": int((~fa).sum()), "methods": {}}
    for m in ("soft", "lexicographic", "ac_corrected"):
        ok = df[f"{m}_ac_feasible"]
        summary["methods"][m] = {
            "ac_feasible_all": int(ok.sum()),
            "ac_feasible_among_authority_feasible": int((ok & fa).sum()),
            "restoration_rate_authority_feasible": float((ok & fa).sum() / max(fa.sum(), 1)),
            "median_distance_kw": float(df[f"{m}_distance_kw"].median()),
            "median_min_v_ac": float(df[f"{m}_min_v_ac"].median())}
    summary["soft_cases_using_slack"] = int((df.soft_total_slack > 1e-6).sum())
    summary["soft_cases_using_slack_while_linear_slack_free"] = int(
        ((df.soft_total_slack > 1e-6) & (df.lexicographic_min_slack <= 1e-6)).sum())
    summary["ac_corrections_distribution"] = {int(k): int(v) for k, v in df.ac_corrections_used.value_counts().items()}
    summary["mean_time_s"] = {"soft": float(df.soft_time_s.mean()), "ac_corrected": float(df.ac_corrected_time_s.mean())}
    with open(os.path.join(OUT, "safety_revalidation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

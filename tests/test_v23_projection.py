"""V23 tests for the lexicographic / AC-corrected projection and authority certificate."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from safe_pb.data_loader import build_tier_a_scenario
from safe_pb.environment import SafePBEnv, EnvConfig
from safe_pb.feeder_model import build_feeder, ac_powerflow_check
from safe_pb.safety_projection import (project, lexicographic_project, ac_corrected_project,
                                       authority_certificate)

CFG = os.path.join(os.path.dirname(__file__), "..", "config", "tier_a_wan2022.yaml")


@pytest.fixture(scope="module")
def stressed_case():
    """A case from the original 168-case grid: feasible exogenous load,
    AC-violating candidate, authority-feasible."""
    sc = build_tier_a_scenario(CFG); topo = build_feeder()
    env = SafePBEnv(sc, topo, EnvConfig(homes_per_bus=3.0, pev_infeasibility_policy="continue_and_report"))
    env.t = 7
    devices, cand, fp, fq = env.build_safety_devices(np.full(env.n_households, 1e6))
    return env, topo, devices, cand, fp, fq


def _ac(env, topo, devices, a, fp, fq):
    return ac_powerflow_check(topo, *env._network_injections(devices, a, fp, fq), base_scale=env.cfg.base_scale)


def test_case_is_a_valid_stress_case(stressed_case):
    env, topo, devices, cand, fp, fq = stressed_case
    assert ac_powerflow_check(topo, fp, fq, base_scale=env.cfg.base_scale)["min_vm_pu"] >= 0.95
    assert _ac(env, topo, devices, cand, fp, fq)["min_vm_pu"] < 0.95
    assert authority_certificate(topo, devices, fp, fq, 3.0, env.cfg.base_scale)["authority_feasible"]


def test_soft_projection_prefers_slack_when_linear_set_nonempty(stressed_case):
    """Documents the V21/V22 defect: soft QP pays slack although min-slack LP is 0."""
    env, topo, devices, cand, fp, fq = stressed_case
    b = topo.non_slack_buses
    soft = project(topo, b, devices, cand, fp, homes_per_bus=3.0, base_scale=env.cfg.base_scale, fixed_bus_q_mvar=fq)
    lex = lexicographic_project(topo, b, devices, cand, fp, fq, homes_per_bus=3.0, base_scale=env.cfg.base_scale)
    assert lex["min_total_slack"] <= 1e-6
    assert soft.total_slack > 1e-4


def test_lexicographic_meets_linear_bound(stressed_case):
    env, topo, devices, cand, fp, fq = stressed_case
    lex = lexicographic_project(topo, topo.non_slack_buses, devices, cand, fp, fq,
                                homes_per_bus=3.0, base_scale=env.cfg.base_scale)
    assert min(lex["linear_v_after"].values()) >= 0.95 - 1e-6
    lb = np.array([d.lb_kw for d in devices]); ub = np.array([d.ub_kw for d in devices])
    assert np.all(lex["a_star"] >= lb - 1e-7) and np.all(lex["a_star"] <= ub + 1e-7)


def test_ac_corrected_restores_ac_feasibility(stressed_case):
    env, topo, devices, cand, fp, fq = stressed_case
    res = ac_corrected_project(topo, topo.non_slack_buses, devices, cand, fp, fq,
                               homes_per_bus=3.0, base_scale=env.cfg.base_scale)
    assert res["ac_feasible"]
    assert _ac(env, topo, devices, res["a_star"], fp, fq)["min_vm_pu"] >= 0.95 - 1e-6


def test_no_movement_when_candidate_already_feasible():
    sc = build_tier_a_scenario(CFG); topo = build_feeder()
    env = SafePBEnv(sc, topo, EnvConfig(homes_per_bus=1, pev_infeasibility_policy="continue_and_report"))
    env.t = 3
    devices, cand, fp, fq = env.build_safety_devices(np.zeros(env.n_households))
    res = ac_corrected_project(topo, topo.non_slack_buses, devices, cand, fp, fq, base_scale=env.cfg.base_scale)
    assert res["projection_distance"] < 1e-5 and res["corrections_used"] == 0


def test_authority_certificate_detects_exogenous_infeasibility():
    sc = build_tier_a_scenario(CFG); topo = build_feeder()
    env = SafePBEnv(sc, topo, EnvConfig(homes_per_bus=4, pev_infeasibility_policy="continue_and_report"))
    env.t = 19
    devices, cand, fp, fq = env.build_safety_devices(np.zeros(env.n_households))
    assert not authority_certificate(topo, devices, fp, fq, 4, env.cfg.base_scale)["authority_feasible"]


def test_env_step_reports_v23_fields():
    sc = build_tier_a_scenario(CFG); topo = build_feeder()
    env = SafePBEnv(sc, topo, EnvConfig(homes_per_bus=3, pev_infeasibility_policy="continue_and_report"))
    env.reset(); env.t = 7
    env.utility_announce_price(np.zeros(3))
    *_, info = env.household_step(np.ones((env.n_households, 1)))
    assert info["projection_mode"] == "ac_corrected"
    assert info["authority_feasible"] is True
    assert not info["ac_safety_violated"]

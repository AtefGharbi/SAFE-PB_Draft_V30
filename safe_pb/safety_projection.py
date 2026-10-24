"""
Feasibility-projection quadratic program (manuscript Sec. 5.2 / Command 6).

SOFT mode (allow_slack=True, the default in `project()`):
    a* = argmin_{a,s}  0.5*(a-a_hat)^T W (a-a_hat) + SLACK_PENALTY * sum(s^2)
    s.t.  lb <= a <= ub
          v_min - s_lo <= v_lin(a) <= v_max + s_hi,   s_lo, s_hi >= 0
This is a PENALTY formulation: slack is always permitted, so a large
internal margin does not force a hard correction -- the solver may simply
pay the (bounded) slack penalty instead of moving `a` further, especially
once moving `a` stops helping the actual bottleneck bus. An earlier
version of this codebase drew a feasibility conclusion ("the joint
feasible set was not reachable") from soft-mode margin sweeps alone,
which conflates "the penalty formulation preferred slack" with "no
feasible point exists" -- caught in external review. `project()` now also
supports `allow_slack=False` (a genuine HARD feasibility problem, no
slack variables at all) and `min_slack_feasibility()` (a proper Phase-I
LP: minimize total slack subject to device bounds and the voltage
constraint) for a defensible infeasibility claim.

Distinguishes three feasibility levels, as required by Command 6:
  1. feasible under the linearized training model (see min_slack_feasibility)
  2. feasible within QP solver tolerance (SOLVER_TOLERANCE below)
  3. feasible under independent AC power flow (feeder_model.ac_powerflow_check)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize, Bounds

from .feeder_model import FeederTopology, linearized_voltages, ac_powerflow_check

SOLVER_TOLERANCE = 1e-6
SLACK_PENALTY = 1e4
DEFAULT_Q_OVER_P = 0.35
AC_FEASIBILITY_TOL = 1e-6   # numerical tolerance for AC feasibility checks near an
# active constraint boundary -- see iterative_ac_repair's usage for why this matters


def voltage_sensitivity(topo: FeederTopology, buses: list, base_scale: float = 1.0):
    """Sp[i,j] = d v(buses[i])/d P(buses[j]) [pu/MW]; Sq analogous for Q."""
    base = linearized_voltages(topo, {}, {}, base_scale=base_scale)
    n = len(buses)
    Sp = np.zeros((n, n))
    Sq = np.zeros((n, n))
    for j, bp in enumerate(buses):
        pert_p = linearized_voltages(topo, {bp: 1.0}, {}, base_scale=base_scale)
        pert_q = linearized_voltages(topo, {}, {bp: 1.0}, base_scale=base_scale)
        for i, b in enumerate(buses):
            Sp[i, j] = base[b] - pert_p[b]
            Sq[i, j] = base[b] - pert_q[b]
    base_v = np.array([base[b] for b in buses])
    return Sp, Sq, base_v


@dataclass
class Device:
    name: str
    bus: int
    lb_kw: float
    ub_kw: float
    weight: float = 1.0
    q_over_p: float = DEFAULT_Q_OVER_P
    # Reactive-power model, now CONFIGURABLE (external review: a single
    # hard-coded assumption was silently applied; different inverter
    # control modes behave differently and the manuscript must state which
    # is used, not assume one):
    #   "unity_pf"      -> Q = 0 always, regardless of P.
    #   "fixed_pf"      -> Q = q_over_p * P (same direction as P; a
    #                      discharging device also reverses Q). This was
    #                      the only behavior in earlier versions.
    #   "independent_q" -> Q is a separate decision variable, bounded by
    #                      a simplified DECOUPLED apparent-power limit
    #                      |Q| <= sqrt(max(s_max_kva^2 - ub_kw^2, 0)) --
    #                      a conservative approximation of P^2+Q^2<=S_max^2
    #                      that avoids a true second-order-cone coupling
    #                      between P and Q in the same solve (stated
    #                      explicitly as a simplification, not hidden).
    inverter_mode: str = "fixed_pf"
    s_max_kva: float | None = None   # required for "independent_q"

    def __post_init__(self):
        if self.inverter_mode not in ("unity_pf", "fixed_pf", "independent_q"):
            raise ValueError(f"unknown inverter_mode: {self.inverter_mode}")
        if self.inverter_mode == "independent_q" and self.s_max_kva is None:
            raise ValueError(f"{self.name}: independent_q mode requires s_max_kva")

    def q_bound_kvar(self) -> float:
        """Decoupled reactive-power bound for independent_q mode."""
        if self.inverter_mode != "independent_q":
            return 0.0
        p_extreme = max(abs(self.lb_kw), abs(self.ub_kw))
        return float(np.sqrt(max(self.s_max_kva ** 2 - p_extreme ** 2, 0.0)))


def _build_mapping(monitored_buses, devices, homes_per_bus):
    """Mp maps each device's active power (kW->MW) to its bus. Mq maps the
    device's contribution to Q for unity_pf/fixed_pf modes only (0 or
    q_over_p*P respectively); independent_q devices contribute through a
    SEPARATE decision variable handled by the caller, not through Mq."""
    bus_index = {b: i for i, b in enumerate(monitored_buses)}
    n = len(devices)
    Mp = np.zeros((len(monitored_buses), n))
    Mq = np.zeros((len(monitored_buses), n))
    for j, d in enumerate(devices):
        if d.bus in bus_index:
            Mp[bus_index[d.bus], j] = homes_per_bus / 1000.0
            if d.inverter_mode == "fixed_pf":
                Mq[bus_index[d.bus], j] = homes_per_bus * d.q_over_p / 1000.0
            elif d.inverter_mode == "unity_pf":
                Mq[bus_index[d.bus], j] = 0.0
            # independent_q: left at 0 here; handled via extra q-variables
    return Mp, Mq


@dataclass
class ProjectionResult:
    a_hat: np.ndarray
    a_star: np.ndarray
    objective_value: float
    projection_distance: float
    solver_success: bool
    solver_status: str
    solver_message: str
    allow_slack: bool
    slack_lo: np.ndarray            # per monitored bus
    slack_hi: np.ndarray            # per monitored bus
    total_slack: float
    joint_projection_uses_slack: bool   # renamed from base_infeasible: this
        # ONLY means the soft-penalty solve chose to use slack, NOT a proof
        # that no slack-free solution exists -- see min_slack_feasibility
        # for that question.
    linear_v_before: dict
    linear_v_after: dict
    violation_before: float   # max(0, worst bound breach) at a_hat -- NEVER negative
    violation_after: float    # max(0, worst bound breach) at a_star -- NEVER negative
    margin_before: float      # signed: v_min - v (or v - v_max), can be negative (= slack margin)
    margin_after: float
    q_indep_star_kvar: np.ndarray    # independent_q devices' optimized Q, in device order (0 for non-independent_q devices)
    q_by_device_kvar: dict           # {device.name: reactive power actually used, ALL devices, any mode}


def project(topo: FeederTopology, monitored_buses: list, devices: list[Device],
            a_hat_kw: np.ndarray, fixed_bus_p_mw: dict, v_min: float = 0.95, v_max: float = 1.05,
            homes_per_bus: float = 1.0, base_scale: float = 1.0,
            fixed_bus_q_mvar: dict | None = None, allow_slack: bool = True) -> ProjectionResult:
    """Solve the safety QP for one time slot.

    `allow_slack=True` (default): the SOFT penalty formulation described in
    the module docstring. Reports how much slack was actually used, but a
    large margin met partly by slack does NOT establish infeasibility.

    `allow_slack=False`: a genuine HARD feasibility problem -- no slack
    variables at all.

    Devices with `inverter_mode="independent_q"` get their own reactive-
    power decision variable, bounded by `Device.q_bound_kvar()` (a
    decoupled apparent-power approximation -- see the Device docstring).
    """
    n = len(devices)
    fixed_bus_q_mvar = fixed_bus_q_mvar or {}
    Sp, Sq, base_v = voltage_sensitivity(topo, monitored_buses, base_scale=base_scale)
    Mp, Mq = _build_mapping(monitored_buses, devices, homes_per_bus)
    Pfix = np.array([fixed_bus_p_mw.get(b, 0.0) for b in monitored_buses])
    Qfix = np.array([fixed_bus_q_mvar.get(b, 0.0) for b in monitored_buses])
    W = np.diag([d.weight for d in devices])
    a_hat = np.asarray(a_hat_kw, dtype=float)
    nb = len(monitored_buses)

    indep_idx = [j for j, d in enumerate(devices) if d.inverter_mode == "independent_q"]
    n_indep = len(indep_idx)
    bus_index = {b: i for i, b in enumerate(monitored_buses)}
    Mq_indep = np.zeros((nb, n_indep))  # maps independent-q vars to bus Q (MVAR/kVAR)
    for k, j in enumerate(indep_idx):
        d = devices[j]
        if d.bus in bus_index:
            Mq_indep[bus_index[d.bus], k] = homes_per_bus / 1000.0

    def split(x):
        a = x[:n]
        q_indep = x[n:n + n_indep]
        rest = x[n + n_indep:]
        return a, q_indep, rest

    def voltage_vec_full(a, q_indep):
        return base_v - Sp @ (Pfix + Mp @ a) - Sq @ (Qfix + Mq @ a + Mq_indep @ q_indep)

    a_only_before = a_hat
    q_indep_before = np.array([0.0] * n_indep)  # candidate assumed 0 extra Q unless specified
    v_before = voltage_vec_full(a_only_before, q_indep_before)
    margin_before = float(np.max(np.maximum(v_min - v_before, v_before - v_max)))
    violation_before = max(margin_before, 0.0)

    n_slack = 2 * nb if allow_slack else 0

    def objective(x):
        a, q_indep, rest = split(x)
        val = 0.5 * (a - a_hat) @ W @ (a - a_hat) + 0.5 * np.sum(q_indep ** 2) * 1e-6
        if allow_slack:
            val += SLACK_PENALTY * np.sum(rest ** 2)
        return val

    def objective_grad(x):
        a, q_indep, rest = split(x)
        g_a = W @ (a - a_hat)
        g_q = q_indep * 1e-6
        if allow_slack:
            g_s = 2 * SLACK_PENALTY * rest
            return np.concatenate([g_a, g_q, g_s])
        return np.concatenate([g_a, g_q])

    if allow_slack:
        def c_lo(x):
            a, q_indep, rest = split(x)
            s_lo = rest[:nb]
            return voltage_vec_full(a, q_indep) - v_min + s_lo

        def c_hi(x):
            a, q_indep, rest = split(x)
            s_hi = rest[nb:]
            return v_max + s_hi - voltage_vec_full(a, q_indep)
    else:
        def c_lo(x):
            a, q_indep, _ = split(x)
            return voltage_vec_full(a, q_indep) - v_min

        def c_hi(x):
            a, q_indep, _ = split(x)
            return v_max - voltage_vec_full(a, q_indep)

    cons = [{"type": "ineq", "fun": c_lo}, {"type": "ineq", "fun": c_hi}]

    q_bounds = [d.q_bound_kvar() for d in devices if d.inverter_mode == "independent_q"]
    lb = ([d.lb_kw for d in devices] + [-qb for qb in q_bounds] +
          ([0.0] * n_slack if allow_slack else []))
    ub = ([d.ub_kw for d in devices] + [qb for qb in q_bounds] +
          ([np.inf] * n_slack if allow_slack else []))
    bounds = Bounds(np.array(lb), np.array(ub))

    x0 = np.concatenate([
        np.clip(a_hat, [d.lb_kw for d in devices], [d.ub_kw for d in devices]),
        np.zeros(n_indep),
        np.zeros(n_slack),
    ])

    res = minimize(objective, x0, jac=objective_grad, method="SLSQP",
                    bounds=bounds, constraints=cons,
                    options={"ftol": SOLVER_TOLERANCE, "maxiter": 300})

    a_star, q_indep_star, rest_star = split(res.x)
    slack_lo = rest_star[:nb] if allow_slack else np.zeros(nb)
    slack_hi = rest_star[nb:] if allow_slack else np.zeros(nb)
    total_slack = float(np.sum(slack_lo) + np.sum(slack_hi))

    v_after = voltage_vec_full(a_star, q_indep_star)
    margin_after = float(np.max(np.maximum(v_min - v_after, v_after - v_max)))
    violation_after = max(margin_after, 0.0)

    # Full per-device Q actually used (not just the independent_q slice) --
    # external review correctly caught that q_indep_star alone was computed
    # internally but never returned, silently losing the optimized reactive
    # decision for any downstream AC check.
    q_indep_full = np.zeros(n)
    for k, j in enumerate(indep_idx):
        q_indep_full[j] = q_indep_star[k]
    q_by_device = {}
    for j, d in enumerate(devices):
        if d.inverter_mode == "unity_pf":
            q_by_device[d.name] = 0.0
        elif d.inverter_mode == "fixed_pf":
            q_by_device[d.name] = float(d.q_over_p * a_star[j])
        else:  # independent_q
            q_by_device[d.name] = float(q_indep_full[j])

    return ProjectionResult(
        a_hat=a_hat, a_star=a_star,
        objective_value=float(res.fun),
        projection_distance=float(np.linalg.norm(a_star - a_hat)),
        solver_success=bool(res.success),
        solver_status=str(getattr(res, "status", "n/a")),
        solver_message=str(res.message),
        allow_slack=allow_slack,
        slack_lo=slack_lo, slack_hi=slack_hi, total_slack=total_slack,
        joint_projection_uses_slack=total_slack > 1e-6,
        linear_v_before={monitored_buses[i]: float(v_before[i]) for i in range(nb)},
        linear_v_after={monitored_buses[i]: float(v_after[i]) for i in range(nb)},
        violation_before=violation_before, violation_after=violation_after,
        margin_before=margin_before, margin_after=margin_after,
        q_indep_star_kvar=q_indep_full, q_by_device_kvar=q_by_device,
    )


def min_slack_feasibility(topo: FeederTopology, monitored_buses: list, devices: list[Device],
                           fixed_bus_p_mw: dict, fixed_bus_q_mvar: dict | None = None,
                           v_min: float = 0.95, v_max: float = 1.05,
                           homes_per_bus: float = 1.0, base_scale: float = 1.0) -> dict:
    """Phase-I feasibility problem: minimize TOTAL slack (not a weighted
    distance-to-candidate objective) subject to device bounds and the
    voltage constraint. If the minimum achievable total slack is > 0 (up
    to solver tolerance), the linearized joint feasible set is genuinely
    empty for this device set/placement -- this is the correct basis for
    an infeasibility claim, not a soft-penalty margin sweep."""
    n = len(devices)
    fixed_bus_q_mvar = fixed_bus_q_mvar or {}
    Sp, Sq, base_v = voltage_sensitivity(topo, monitored_buses, base_scale=base_scale)
    Mp, Mq = _build_mapping(monitored_buses, devices, homes_per_bus)
    Pfix = np.array([fixed_bus_p_mw.get(b, 0.0) for b in monitored_buses])
    Qfix = np.array([fixed_bus_q_mvar.get(b, 0.0) for b in monitored_buses])
    nb = len(monitored_buses)

    def voltage_vec(x):
        a = x[:n]
        return base_v - Sp @ (Pfix + Mp @ a) - Sq @ (Qfix + Mq @ a)

    def objective(x):
        return float(np.sum(x[n:]))  # pure slack minimization, LINEAR objective

    def objective_grad(x):
        g = np.zeros(len(x))
        g[n:] = 1.0
        return g

    def c_lo(x):
        v = voltage_vec(x)
        s_lo = x[n:n + nb]
        return v - v_min + s_lo

    def c_hi(x):
        v = voltage_vec(x)
        s_hi = x[n + nb:]
        return v_max + s_hi - v

    cons = [{"type": "ineq", "fun": c_lo}, {"type": "ineq", "fun": c_hi}]
    lb = np.concatenate([[d.lb_kw for d in devices], np.zeros(2 * nb)])
    ub = np.concatenate([[d.ub_kw for d in devices], np.full(2 * nb, np.inf)])
    bounds = Bounds(lb, ub)
    # Start from the midpoint of device bounds (not a_hat, since Phase-I
    # asks a different question: is there ANY feasible action, regardless
    # of proximity to any particular candidate).
    x0 = np.concatenate([
        np.array([(d.lb_kw + d.ub_kw) / 2 for d in devices]),
        np.full(2 * nb, 1.0),
    ])

    res = minimize(objective, x0, jac=objective_grad, method="SLSQP",
                    bounds=bounds, constraints=cons,
                    options={"ftol": 1e-9, "maxiter": 500})

    min_total_slack = float(res.fun)
    feasible = res.success and min_total_slack <= 1e-6
    return {
        "feasible": feasible, "solver_success": bool(res.success),
        "solver_message": str(res.message), "min_total_slack": min_total_slack,
        "a_star": res.x[:n].tolist(),
    }


def max_min_voltage(topo: FeederTopology, monitored_buses: list, devices: list[Device],
                     fixed_bus_p_mw: dict, fixed_bus_q_mvar: dict | None = None,
                     base_scale: float = 1.0) -> dict:
    """'Available control authority' experiment (external review's request):
    ignore any candidate action entirely and ask the best possible
    question -- maximize the MINIMUM linearized bus voltage over all
    monitored buses, subject only to device bounds. This is the best this
    device set could ever achieve, distinguishing "the objective preferred
    not to move `a`" from "moving `a` cannot help." The resulting action
    is then meant to be independently checked with full AC power flow by
    the caller."""
    n = len(devices)
    fixed_bus_q_mvar = fixed_bus_q_mvar or {}
    Sp, Sq, base_v = voltage_sensitivity(topo, monitored_buses, base_scale=base_scale)
    Mp, Mq = _build_mapping(monitored_buses, devices, homes_per_bus=1.0)
    Pfix = np.array([fixed_bus_p_mw.get(b, 0.0) for b in monitored_buses])
    Qfix = np.array([fixed_bus_q_mvar.get(b, 0.0) for b in monitored_buses])

    def voltage_vec(a):
        return base_v - Sp @ (Pfix + Mp @ a) - Sq @ (Qfix + Mq @ a)

    # maximize min(v) == minimize -min(v) == minimize t s.t. -v <= t for all buses
    def objective(x):
        return x[n]  # t

    def objective_grad(x):
        g = np.zeros(len(x))
        g[n] = 1.0
        return g

    def cons_fun(x):
        a, t = x[:n], x[n]
        v = voltage_vec(a)
        return t - (-v)  # t >= -v  <=>  -t <= v

    cons = [{"type": "ineq", "fun": cons_fun}]
    lb = np.array([d.lb_kw for d in devices] + [-10.0])
    ub = np.array([d.ub_kw for d in devices] + [10.0])
    x0 = np.array([(d.lb_kw + d.ub_kw) / 2 for d in devices] + [0.0])

    res = minimize(objective, x0, jac=objective_grad, method="SLSQP",
                    bounds=Bounds(lb, ub), constraints=cons,
                    options={"ftol": 1e-9, "maxiter": 500})
    a_star = res.x[:n]
    v_achieved = voltage_vec(a_star)
    return {
        "solver_success": bool(res.success), "solver_message": str(res.message),
        "a_star": a_star.tolist(), "best_achievable_min_v_linear": float(np.min(v_achieved)),
        "linear_v_by_bus": {monitored_buses[i]: float(v_achieved[i]) for i in range(len(monitored_buses))},
    }


def max_min_voltage_ac(topo: FeederTopology, devices: list[Device], fixed_bus_p_mw: dict,
                        fixed_bus_q_mvar: dict | None = None, base_scale: float = 1.0,
                        n_multistart: int = 8, de_seed: int = 0, de_maxiter: int = 40,
                        de_popsize: int = 15) -> dict:
    """Direct AC-power-flow-based global search for the action maximizing
    the minimum AC bus voltage, using SciPy's differential_evolution (a
    population-based global optimizer) followed by local polishing
    multistarts. This directly optimizes the AC objective -- unlike
    `max_min_voltage`, which optimizes the LINEAR model and only AC-checks
    the single resulting action, this function's objective function IS an
    AC power flow call, so every candidate the optimizer considers is
    evaluated at AC fidelity.

    IMPORTANT SCOPE STATEMENT: differential evolution with a finite
    iteration budget is a heuristic global search, not a certified global
    optimizer. If no AC-feasible action is found, the correct claim is
    "no AC-feasible action was found within this search" -- NOT a proof
    that none exists.

    SCOPE LIMITATION, enforced not just documented: this function's
    objective only models `fixed_pf` and `unity_pf` reactive-power modes
    (matching `_apply_action_to_grid`); `independent_q` devices are NOT
    supported here (their Q is a separate decision variable that only
    `project()` optimizes). Passing an independent_q device raises rather
    than silently treating its Q as zero.
    """
    if any(d.inverter_mode == "independent_q" for d in devices):
        raise ValueError("max_min_voltage_ac does not support independent_q devices "
                          "(their reactive power is not modeled by this function's AC "
                          "objective) -- use only fixed_pf/unity_pf devices here, or "
                          "extend this function before use with independent_q.")
    n = len(devices)
    lb = np.array([d.lb_kw for d in devices])
    ub = np.array([d.ub_kw for d in devices])
    fixed_bus_q_mvar = fixed_bus_q_mvar or {}

    eval_log = []

    def neg_min_voltage_ac(a):
        """Objective for the optimizer: -min(AC voltage), so minimizing
        this maximizes the minimum voltage. Non-convergent power flow is
        penalized (not silently treated as zero or ignored) so the
        optimizer is pushed away from operating points AC power flow
        itself cannot solve."""
        p, q = dict(fixed_bus_p_mw), dict(fixed_bus_q_mvar)
        for d, kw in zip(devices, a):
            p[d.bus] = p.get(d.bus, 0.0) + kw / 1000.0
            q_contrib = (0.0 if d.inverter_mode == "unity_pf" else
                         kw * d.q_over_p / 1000.0 if d.inverter_mode == "fixed_pf" else 0.0)
            q[d.bus] = q.get(d.bus, 0.0) + q_contrib
        ac = ac_powerflow_check(topo, p, q, base_scale=base_scale)
        if not ac["converged"]:
            eval_log.append({"a": list(a), "converged": False, "min_v": None})
            return 10.0  # penalize non-convergence heavily but finitely
        eval_log.append({"a": list(a), "converged": True, "min_v": ac["min_vm_pu"]})
        return -ac["min_vm_pu"]

    from scipy.optimize import differential_evolution
    de_result = differential_evolution(
        neg_min_voltage_ac, bounds=list(zip(lb, ub)), seed=de_seed,
        maxiter=de_maxiter, popsize=de_popsize, tol=1e-6, polish=True,
        updating="deferred", workers=1,  # workers=1 for reproducibility
    )
    de_best_a = de_result.x
    de_best_min_v = -de_result.fun

    # Reproducible multistart local polish around the DE result and a few
    # deterministic seeded points, for a second independent check.
    rng = np.random.default_rng(de_seed + 1000)
    multistart_results = []
    for i in range(n_multistart):
        if i == 0:
            x0 = de_best_a.copy()
        else:
            x0 = rng.uniform(lb, ub)
        res = minimize(lambda a: neg_min_voltage_ac(a), x0, method="Nelder-Mead",
                        bounds=list(zip(lb, ub)),
                        options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 300})
        multistart_results.append({"start_index": i, "success": bool(res.success),
                                    "final_min_v": -res.fun, "a_star": res.x.tolist()})

    best_overall = max([{"a_star": de_best_a.tolist(), "min_v": de_best_min_v, "source": "differential_evolution"}] +
                        [{"a_star": m["a_star"], "min_v": m["final_min_v"], "source": f"multistart_{m['start_index']}"}
                         for m in multistart_results],
                        key=lambda r: r["min_v"])

    v_min_target = 0.95
    feasible_found = best_overall["min_v"] >= v_min_target
    return {
        "n_ac_evaluations": len(eval_log),
        "eval_log": eval_log,  # full per-candidate record: {a, converged, min_v} -- previously
        # only its length was returned, so "every candidate is recorded" was not actually true;
        # callers with size concerns should slice/save this themselves (e.g. to CSV).
        "de_result": {"success": bool(de_result.success), "message": str(de_result.message),
                      "best_min_v": de_best_min_v, "a_star": de_best_a.tolist(), "seed": de_seed,
                      "n_iterations": int(de_result.nit)},
        "multistart_results": multistart_results,
        "best_overall": best_overall,
        "conclusion": ("an AC-feasible action (min V >= 0.95) was found within this search"
                       if feasible_found else
                       "no AC-feasible action was found within this search -- NOT a proof "
                       "that none exists, since DE with a finite budget is a heuristic, not "
                       "a certified global optimizer"),
    }


def _apply_action_to_grid(devices, action, fixed_p, fixed_q):
    """Shared helper: builds bus P/Q dicts from a device action, respecting
    each device's inverter mode. Used consistently by the repair loop and
    its fallback so the two paths cannot silently diverge (the fallback
    previously omitted Q entirely -- this helper makes that impossible by
    construction, since there's only one code path now).

    SCOPE LIMITATION, enforced not just documented: only `fixed_pf` and
    `unity_pf` are modeled here. `independent_q` devices raise rather than
    silently contributing zero Q -- their reactive power is a separate
    decision variable that only `project()` currently optimizes; anything
    routing through this helper (the AC-repair loop, its fallback, and the
    demo script) does not yet support them."""
    if any(d.inverter_mode == "independent_q" for d in devices):
        raise ValueError("_apply_action_to_grid does not support independent_q devices; "
                          "the AC-repair loop, its fallback, and max_min_voltage_ac are "
                          "currently scoped to fixed_pf/unity_pf devices only.")
    p, q = dict(fixed_p), dict(fixed_q)
    for d, kw in zip(devices, action):
        p[d.bus] = p.get(d.bus, 0.0) + kw / 1000.0
        if d.inverter_mode == "fixed_pf":
            q[d.bus] = q.get(d.bus, 0.0) + kw * d.q_over_p / 1000.0
        # unity_pf contributes 0
    return p, q


def _ac_finite_difference_jacobian(topo, devices, current_action, fixed_p, fixed_q,
                                    base_scale, target_bus, delta_kw=0.5):
    """Genuine local sensitivity: perturb each device and rerun FULL AC
    power flow (not the linear model) to measure the resulting change in
    voltage at `target_bus`. Uses a BOUND-RESPECTING one-sided difference
    (forward if the device has room to increase, else backward, else the
    device is pinned within less than delta_kw of both bounds and
    contributes zero sensitivity) -- an earlier version always used
    `pert[j] += delta_kw` regardless of the device's upper bound, which
    could evaluate AC power flow at a physically invalid, out-of-bounds
    action."""
    p0, q0 = _apply_action_to_grid(devices, current_action, fixed_p, fixed_q)
    base_ac = ac_powerflow_check(topo, p0, q0, base_scale=base_scale)
    if not base_ac["converged"] or target_bus not in base_ac["vm_pu"]:
        return None, None
    v0 = base_ac["vm_pu"][target_bus]
    jac = np.zeros(len(devices))
    for j, d in enumerate(devices):
        if current_action[j] + delta_kw <= d.ub_kw + 1e-9:
            step = delta_kw
        elif current_action[j] - delta_kw >= d.lb_kw - 1e-9:
            step = -delta_kw
        else:
            jac[j] = 0.0  # pinned within less than delta_kw of both bounds
            continue
        pert = current_action.copy()
        pert[j] += step
        p1, q1 = _apply_action_to_grid(devices, pert, fixed_p, fixed_q)
        ac1 = ac_powerflow_check(topo, p1, q1, base_scale=base_scale)
        if not ac1["converged"] or target_bus not in ac1["vm_pu"]:
            jac[j] = 0.0
            continue
        jac[j] = (ac1["vm_pu"][target_bus] - v0) / step
    return jac, v0


def _local_jacobian_correction(devices, current_action, jac, v0, v_min, v_max, trust_region_kw):
    """One local SQP-style step targeting the currently-worst bus, now
    enforcing BOTH v >= v_min and v <= v_max (an earlier version only had
    the lower-bound constraint, so it could not repair overvoltage even
    though the final AC check tests both limits). Returns a dict with the
    corrected action AND the solver's own success/message/displacement --
    an earlier version silently returned the unchanged action on solver
    failure with no record of why, so a history showing an unchanged
    action was indistinguishable from "already optimal" vs. "solver
    failed silently"."""
    lb = np.maximum([d.lb_kw for d in devices], current_action - trust_region_kw)
    ub = np.minimum([d.ub_kw for d in devices], current_action + trust_region_kw)

    def obj(x):
        return float(0.5 * np.sum((x - current_action) ** 2))

    def obj_grad(x):
        return x - current_action

    def c_lo(x):  # v >= v_min
        return v0 + jac @ (x - current_action) - v_min

    def c_hi(x):  # v <= v_max
        return v_max - (v0 + jac @ (x - current_action))

    res = minimize(obj, current_action, jac=obj_grad, method="SLSQP",
                    bounds=Bounds(lb, ub),
                    constraints=[{"type": "ineq", "fun": c_lo}, {"type": "ineq", "fun": c_hi}],
                    options={"maxiter": 100, "ftol": 1e-9})
    action = res.x if res.success else current_action
    return {
        "action": action,
        "solver_success": bool(res.success),
        "solver_message": str(res.message),
        "displacement_kw": float(np.linalg.norm(action - current_action)),
    }


def iterative_ac_repair(topo: FeederTopology, monitored_buses: list, devices: list[Device],
                         a_hat_kw: np.ndarray, fixed_bus_p_mw: dict, v_min: float = 0.95,
                         v_max: float = 1.05, homes_per_bus: float = 1.0, base_scale: float = 1.0,
                         fixed_bus_q_mvar: dict | None = None, max_iterations: int = 5,
                         trust_region_kw: float = 5.0, fallback_action: np.ndarray | None = None) -> dict:
    """Iterative AC-repair procedure (Command 6 extension) -- CORRECTED
    version using a genuine finite-difference AC Jacobian re-computed at
    every iteration's actual operating point (see
    `_ac_finite_difference_jacobian`), not a fixed, operating-point-
    independent linear matrix. Handles BOTH undervoltage and overvoltage
    (an earlier version's local correction only had a v>=v_min constraint).
    Reports three DISTINCT outcomes:
      - "ac_verified_feasible": AC power flow confirms the final action
      - "fallback_activated": iteration limit reached; a stated fallback,
        built from the ORIGINAL (untightened) device bounds with P and Q
        both applied consistently, is evaluated and reported honestly --
        including as "unresolved_infeasibility" if even the fallback
        violates the AC limit (an earlier version called this a "safe
        fallback" regardless of whether it actually was safe).

    KNOWN LIMITATION, stated rather than hidden: each iteration targets
    only the single MOST-violated bus (in whichever direction). If a
    network has multiple simultaneously-violated buses that pull the
    correction in different directions, this can converge slowly or
    oscillate between them rather than resolving all of them jointly. A
    multi-bus-simultaneous correction (e.g. penalizing the sum or max of
    all violations in one QP) would be the natural next extension.
    """
    fixed_bus_q_mvar = fixed_bus_q_mvar or {}
    original_devices = list(devices)  # preserved for the fallback -- an earlier
    # version's fallback used bounds already tightened by prior iterations,
    # not the device's true physical bounds.
    # Clip the initial candidate to device bounds -- an out-of-range a_hat
    # was previously carried through the whole loop unclipped, which could
    # also invert the trust-region bounds in the local correction step
    # (lb=max(d.lb, x-tr) > ub=min(d.ub, x+tr) when x is far outside [d.lb,d.ub]).
    current_action = np.clip(np.asarray(a_hat_kw, dtype=float).copy(),
                              [d.lb_kw for d in devices], [d.ub_kw for d in devices])
    history = []

    for it in range(max_iterations):
        p, q = _apply_action_to_grid(devices, current_action, fixed_bus_p_mw, fixed_bus_q_mvar)
        ac = ac_powerflow_check(topo, p, q, base_scale=base_scale)
        # Small explicit numerical tolerance: a correctly-converged
        # constrained optimum sits AT the active constraint boundary,
        # where strict floating-point comparison is fragile (observed:
        # genuine repairs landing at 0.949999999998, a few ULPs under 0.95
        # from roundoff alone, not a real remaining violation).
        ac_feasible = (ac["converged"] and ac["min_vm_pu"] >= v_min - AC_FEASIBILITY_TOL
                       and ac["max_vm_pu"] <= v_max + AC_FEASIBILITY_TOL)

        # Target whichever bus is MOST violated, in EITHER direction -- an
        # earlier version only ever looked at undervoltage (min_vm_pu),
        # so the repair could not address an overvoltage violation even
        # though the final AC check tests both v_min and v_max.
        target_bus, target_kind = None, None
        if ac["converged"]:
            under = {b: v for b, v in ac["vm_pu"].items() if v < v_min}
            over = {b: v for b, v in ac["vm_pu"].items() if v > v_max}
            worst_under = (v_min - min(under.values())) if under else -1.0
            worst_over = (max(over.values()) - v_max) if over else -1.0
            if worst_under >= worst_over and under:
                target_bus, target_kind = min(under, key=under.get), "under"
            elif over:
                target_bus, target_kind = max(over, key=over.get), "over"

        entry = {
            "iteration": it, "ac_converged": ac["converged"], "ac_min_v": ac.get("min_vm_pu"),
            "ac_max_v": ac.get("max_vm_pu"), "target_bus": target_bus, "target_kind": target_kind,
            "ac_feasible": ac_feasible, "action": current_action.tolist(),
        }
        if ac_feasible:
            history.append(entry)
            return {
                "outcome": "ac_verified_feasible", "iterations_used": it + 1,
                "final_action": current_action.tolist(), "history": history,
                "fallback_activated": False,
            }
        if target_bus is None:
            entry["stopped_reason"] = "AC power flow did not converge; no target bus to repair toward"
            history.append(entry)
            break
        jac, v0 = _ac_finite_difference_jacobian(topo, devices, current_action, fixed_bus_p_mw,
                                                   fixed_bus_q_mvar, base_scale, target_bus)
        if jac is None:
            entry["stopped_reason"] = "Jacobian evaluation's own AC power flow call did not converge"
            history.append(entry)
            break
        correction = _local_jacobian_correction(devices, current_action, jac, v0, v_min, v_max, trust_region_kw)
        entry["local_correction"] = {
            "solver_success": correction["solver_success"],
            "solver_message": correction["solver_message"],
            "displacement_kw": correction["displacement_kw"],
        }
        history.append(entry)
        current_action = correction["action"]

    # Iteration limit reached (or AC divergence) without feasibility: build
    # the fallback from ORIGINAL device bounds, with P and Q both applied.
    if fallback_action is None:
        fallback_action = np.array([d.lb_kw for d in original_devices])
    p, q = _apply_action_to_grid(original_devices, np.asarray(fallback_action), fixed_bus_p_mw, fixed_bus_q_mvar)
    ac_fallback = ac_powerflow_check(topo, p, q, base_scale=base_scale)
    fallback_feasible = bool(ac_fallback["converged"] and ac_fallback.get("min_vm_pu", 0) >= v_min - AC_FEASIBILITY_TOL
                              and ac_fallback.get("max_vm_pu", 0) <= v_max + AC_FEASIBILITY_TOL)
    return {
        # An earlier version called this "fallback_activated" regardless of
        # whether the fallback itself was AC-feasible, which reads as if a
        # safe state was always reached. If the fallback ALSO violates the
        # AC limit, the honest outcome is "unresolved_infeasibility."
        "outcome": "fallback_activated" if fallback_feasible else "unresolved_infeasibility",
        "iterations_used": len(history), "final_action": np.asarray(fallback_action).tolist(),
        "history": history, "fallback_activated": True,
        "fallback_ac_min_v": ac_fallback.get("min_vm_pu"), "fallback_ac_max_v": ac_fallback.get("max_vm_pu"),
        "fallback_ac_feasible": fallback_feasible,
    }


def fixed_load_feasibility(topo: FeederTopology, fixed_bus_p_mw: dict, fixed_bus_q_mvar: dict,
                            v_min: float = 0.95, v_max: float = 1.05,
                            base_scale: float = 1.0) -> dict:
    """Tests whether the network is AC-feasible from FIXED load alone (all
    controllable devices at zero). A separate, well-defined question from
    whether the joint feasible set (with controllable devices active) is
    empty -- see `min_slack_feasibility` for that question."""
    ac = ac_powerflow_check(topo, fixed_bus_p_mw, fixed_bus_q_mvar, base_scale=base_scale)
    if not ac["converged"]:
        return {"feasible": False, "converged": False, "reason": "AC power flow did not converge"}
    min_v, max_v = ac["min_vm_pu"], ac["max_vm_pu"]
    feasible = (min_v >= v_min) and (max_v <= v_max)
    return {
        "feasible": feasible, "converged": True, "min_vm_pu": min_v, "max_vm_pu": max_v,
        "reason": None if feasible else (
            f"fixed load alone violates the band: min={min_v:.4f} (limit {v_min}), "
            f"max={max_v:.4f} (limit {v_max})"),
    }


# ---------------------------------------------------------------------------
# V23: lexicographic and AC-corrected projection
# ---------------------------------------------------------------------------
# Root cause of the V21/V22 "0/168" safety result: `project()` prices slack at
# SLACK_PENALTY * s^2 with s in p.u. A 0.002 p.u. slack costs 1e4 * 4e-6 = 0.04,
# whereas moving one device by 1 kW costs 0.5. The QP therefore preferred slack
# even when the linear model itself admitted a slack-free solution (147 of 168
# cases). The functions below remove that trade-off: slack is minimised first
# (exact LP), the distance objective second, and an AC-measured per-bus bias
# corrects the LinDistFlow model at the current operating point.

_SENS_CACHE: dict = {}


def _cached_sensitivity(topo, buses, base_scale):
    key = (id(topo), tuple(buses), float(base_scale))
    if key not in _SENS_CACHE:
        _SENS_CACHE[key] = voltage_sensitivity(topo, buses, base_scale=base_scale)
    return _SENS_CACHE[key]


def lexicographic_project(topo: FeederTopology, monitored_buses: list, devices: list[Device],
                          a_hat_kw: np.ndarray, fixed_bus_p_mw: dict, fixed_bus_q_mvar: dict | None = None,
                          v_min: float = 0.95, v_max: float = 1.05, homes_per_bus: float = 1.0,
                          base_scale: float = 1.0, bias_pu: np.ndarray | None = None) -> dict:
    """Two-stage projection onto the linearised voltage set.

    Stage 1 (LP):  s* = min 1^T s  s.t.  v_min - s <= v(a) + bias <= v_max + s,  lb <= a <= ub
    Stage 2 (QP):  a* = argmin ||a - a_hat||_W^2  s.t. the same constraints with s fixed at s*.
    Slack is therefore used only when no slack-free action exists in the model.
    Only fixed_pf / unity_pf devices are supported.
    """
    from scipy.optimize import linprog, LinearConstraint
    if any(d.inverter_mode == "independent_q" for d in devices):
        raise ValueError("lexicographic_project supports fixed_pf/unity_pf devices only")
    fixed_bus_q_mvar = fixed_bus_q_mvar or {}
    Sp, Sq, base_v = _cached_sensitivity(topo, monitored_buses, base_scale)
    Mp, Mq = _build_mapping(monitored_buses, devices, homes_per_bus)
    Pfix = np.array([fixed_bus_p_mw.get(b, 0.0) for b in monitored_buses])
    Qfix = np.array([fixed_bus_q_mvar.get(b, 0.0) for b in monitored_buses])
    nb, n = len(monitored_buses), len(devices)
    bias = np.zeros(nb) if bias_pu is None else np.asarray(bias_pu, dtype=float)
    v0 = base_v - Sp @ Pfix - Sq @ Qfix + bias
    G = Sp @ Mp + Sq @ Mq                     # v(a) = v0 - G a
    lb = np.array([d.lb_kw for d in devices], dtype=float)
    ub = np.array([d.ub_kw for d in devices], dtype=float)
    a_hat = np.clip(np.asarray(a_hat_kw, dtype=float), lb, ub)

    A = np.block([[G, -np.eye(nb)], [-G, -np.eye(nb)]])
    b = np.r_[v0 - v_min, v_max - v0]
    lp = linprog(np.r_[np.zeros(n), np.ones(nb)], A_ub=A, b_ub=b,
                 bounds=list(zip(lb, ub)) + [(0, None)] * nb, method="highs")
    s_star = (lp.x[n:] if lp.success else np.zeros(nb)) + 1e-9
    x0 = lp.x[:n] if lp.success else a_hat
    W = np.diag([d.weight for d in devices])
    cons = LinearConstraint(-G, v_min - s_star - v0, v_max + s_star - v0)
    res = minimize(lambda a: 0.5 * (a - a_hat) @ W @ (a - a_hat), x0,
                   jac=lambda a: W @ (a - a_hat), method="SLSQP", bounds=Bounds(lb, ub),
                   constraints=[cons], options={"ftol": 1e-10, "maxiter": 500})
    a_star = res.x if res.success else x0
    v_before, v_after = v0 - G @ a_hat, v0 - G @ a_star
    return {
        "a_star": a_star, "a_hat": a_hat,
        "min_total_slack": float(max(s_star.sum() - 1e-9 * nb, 0.0)),
        "lp_success": bool(lp.success), "qp_success": bool(res.success),
        "linear_v_before": dict(zip(monitored_buses, v_before.tolist())),
        "linear_v_after": dict(zip(monitored_buses, v_after.tolist())),
        "projection_distance": float(np.linalg.norm(a_star - a_hat)),
    }


def authority_certificate(topo: FeederTopology, devices: list[Device], fixed_bus_p_mw: dict,
                          fixed_bus_q_mvar: dict | None = None, homes_per_bus: float = 1.0,
                          base_scale: float = 1.0, v_min: float = 0.95) -> dict:
    """Exact (for undervoltage on a radial feeder with fixed_pf/unity_pf
    consumers) test of whether ANY device-feasible action satisfies v_min:
    every bus voltage is non-increasing in every device's P (and Q = k*P,
    k >= 0), so the all-at-lower-bound action maximises every bus voltage
    simultaneously. If it is AC-infeasible, no action is AC-feasible and the
    violation is attributable to the exogenous operating point."""
    fixed_bus_q_mvar = fixed_bus_q_mvar or {}
    lb = np.array([d.lb_kw for d in devices], dtype=float)
    p, q = dict(fixed_bus_p_mw), dict(fixed_bus_q_mvar)
    for d, kw in zip(devices, lb):
        p[d.bus] = p.get(d.bus, 0.0) + kw * homes_per_bus / 1000.0
        q[d.bus] = q.get(d.bus, 0.0) + (kw * d.q_over_p if d.inverter_mode == "fixed_pf" else 0.0) * homes_per_bus / 1000.0
    ac = ac_powerflow_check(topo, p, q, base_scale=base_scale)
    ok = bool(ac.get("converged") and ac["min_vm_pu"] >= v_min - AC_FEASIBILITY_TOL)
    return {"authority_feasible": ok, "best_min_vm_pu": ac.get("min_vm_pu", float("nan")),
            "converged": bool(ac.get("converged"))}


def ac_corrected_project(topo: FeederTopology, monitored_buses: list, devices: list[Device],
                         a_hat_kw: np.ndarray, fixed_bus_p_mw: dict, fixed_bus_q_mvar: dict | None = None,
                         v_min: float = 0.95, v_max: float = 1.05, homes_per_bus: float = 1.0,
                         base_scale: float = 1.0, max_corrections: int = 4) -> dict:
    """Sequential bias-corrected projection.

    Iteration k solves `lexicographic_project` with bias_k = V_AC(a_k) - V_lin(a_k)
    (per bus), then evaluates AC power flow at the new action. Terminates when
    the AC schedule satisfies the band or after `max_corrections` corrections.
    The returned `ac` result is the independent AC check of the applied action.
    """
    from .feeder_model import linearized_voltages
    fixed_bus_q_mvar = fixed_bus_q_mvar or {}
    bias = np.zeros(len(monitored_buses))
    history = []
    for k in range(max_corrections + 1):
        lex = lexicographic_project(topo, monitored_buses, devices, a_hat_kw, fixed_bus_p_mw,
                                    fixed_bus_q_mvar, v_min, v_max, homes_per_bus, base_scale, bias)
        # _apply_action_to_grid assumes 1 home per device; rescale explicitly
        p, q = dict(fixed_bus_p_mw), dict(fixed_bus_q_mvar)
        for d, kw in zip(devices, lex["a_star"]):
            p[d.bus] = p.get(d.bus, 0.0) + kw * homes_per_bus / 1000.0
            q[d.bus] = q.get(d.bus, 0.0) + (kw * d.q_over_p if d.inverter_mode == "fixed_pf" else 0.0) * homes_per_bus / 1000.0
        ac = ac_powerflow_check(topo, p, q, base_scale=base_scale)
        history.append({"correction": k, "ac_min_v": ac.get("min_vm_pu"), "min_total_slack": lex["min_total_slack"]})
        if not ac.get("converged"):
            break
        feasible = (ac["min_vm_pu"] >= v_min - AC_FEASIBILITY_TOL and ac["max_vm_pu"] <= v_max + AC_FEASIBILITY_TOL)
        if feasible or k == max_corrections:
            break
        lin = linearized_voltages(topo, p, q, base_scale=base_scale)
        bias = np.array([ac["vm_pu"][b] - lin[b] for b in monitored_buses])
    lex["ac"] = ac
    lex["corrections_used"] = k
    lex["ac_feasible"] = bool(ac.get("converged") and feasible) if ac.get("converged") else False
    lex["history"] = history
    return lex

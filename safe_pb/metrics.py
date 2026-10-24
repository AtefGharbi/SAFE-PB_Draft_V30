"""Metrics: economic, peak/PAR, fairness (Jain/Gini), and CVaR."""
from __future__ import annotations
import numpy as np


def daily_totals(step_results) -> dict:
    U = sum(r.U_t for r in step_results)
    C = sum(r.C_t for r in step_results)
    social_welfare = sum(r.r_t for r in step_results)  # rho*U - (1-rho)*C, summed = sum of r_t
    load = np.array([r.p_non.sum() + r.p_gen.sum() + r.p_pev.sum() for r in step_results])
    peak = float(load.max())
    avg = float(load.mean())
    par = peak / avg if avg > 0 else np.nan
    return {"total_utility_profit": U, "total_cost": C, "social_welfare": social_welfare,
            "peak_kw": peak, "avg_kw": avg, "PAR": par, "load_profile_kw": load}


def jain_index(x: np.ndarray) -> float:
    """Jain's fairness index. Only defined for non-negative x (it's a
    fairness measure over resource ALLOCATIONS, e.g. discomfort or energy,
    not over signed quantities like net savings). Raises rather than
    silently returning a misleading number if given negative values --
    caught in external review: an earlier version had no such guard."""
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError(
            "jain_index is undefined for negative inputs (got values < 0); "
            "if you need fairness over a signed quantity such as net bill "
            "savings, shift/rescale to a non-negative allocation first and "
            "document the transform, or use gini_coefficient on |x| with "
            "that caveat stated explicitly.")
    if np.allclose(x, 0):
        return 1.0
    return float((x.sum() ** 2) / (len(x) * np.sum(x ** 2)))


def gini_coefficient(x: np.ndarray) -> float:
    """Gini coefficient. Only defined for non-negative x; see jain_index's
    docstring for the same reasoning. Raises on negative input rather than
    returning a value that looks valid but isn't (a mean-based Gini formula
    can be silently nonsensical, e.g. >1 or negative, when x has mixed
    signs)."""
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError(
            "gini_coefficient is undefined for negative inputs (got values "
            "< 0); see jain_index's docstring for the same guard and why.")
    x = np.sort(x)
    n = len(x)
    if n == 0 or np.allclose(x.sum(), 0):
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Empirical CVaR (average of the worst (1-alpha) fraction of losses)."""
    losses = np.sort(np.asarray(losses, dtype=float))
    n = len(losses)
    k = max(int(np.ceil((1 - alpha) * n)), 1)
    return float(losses[-k:].mean())


def voltage_violation_summary(vm_pu: dict, v_min: float = 0.95, v_max: float = 1.05) -> dict:
    vals = np.array(list(vm_pu.values()))
    viol_low = vals < v_min
    viol_high = vals > v_max
    return {
        "n_buses": len(vals),
        "n_violations_low": int(viol_low.sum()),
        "n_violations_high": int(viol_high.sum()),
        "max_deviation_below_min": float(max(v_min - vals[viol_low].min(), 0.0)) if viol_low.any() else 0.0,
        "max_deviation_above_max": float(max(vals[viol_high].max() - v_max, 0.0)) if viol_high.any() else 0.0,
        "min_vm_pu": float(vals.min()),
        "max_vm_pu": float(vals.max()),
    }

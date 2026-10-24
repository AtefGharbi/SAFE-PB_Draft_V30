"""
Loads config/tier_a_wan2022.yaml and builds a reproducible Tier A scenario:
household/PEV objects (Wan et al. Table I parameters, verbatim) plus
approximated demand-shape time series (labeled [APPROX] where the source
paper does not publish numeric values, only figures).

Every returned field is tagged in `Scenario.provenance` as either
'table' (transcribed from the source paper's tables), 'approx' (shape-matched
synthetic reconstruction, no claim of numeric fidelity to the source
figures), or 'extension' (SAFE-PB-only, not in the source paper at all).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import yaml

from .household_models import GeneralDispatchableLoad, NonDispatchableLoad
from .ev_model import PEVModel
from .battery_model import BatteryModel
from .pv_model import synthetic_pv_profile


@dataclass
class Scenario:
    T: int
    dt_hours: float
    wholesale_price: np.ndarray          # [table]
    xi_elasticity: np.ndarray            # [table]
    rho: float                           # [table]
    eta_bounds: tuple                    # (lower[T], upper[T]) [table-derived]

    gen_loads: list                      # GeneralDispatchableLoad objects [table params]
    gen_demand_kwh: np.ndarray           # [n_gen, T] [approx]

    nondispatch_loads: list              # NonDispatchableLoad objects
    nondispatch_demand_kwh: np.ndarray   # [n_non, T] [approx]

    pevs: list                           # PEVModel objects [table params]
    pev_demand_kwh: np.ndarray           # [n_pev, T] [approx target-charge shape]

    batteries: list                      # BatteryModel objects [extension]
    pv_profiles_kw: np.ndarray           # [n_pv, T] [approx/synthetic]
    network_config: dict                 # authoritative feeder calibration

    provenance: dict = field(default_factory=dict)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _synthetic_general_load_shape(rng: np.random.Generator, T: int, peak_kwh: float) -> np.ndarray:
    hours = np.arange(T)
    peak1 = np.exp(-0.5 * ((hours - 12.5) / 2.5) ** 2)   # 10:00-15:00 bump
    peak2 = np.exp(-0.5 * ((hours - 20.5) / 1.5) ** 2)   # 19:00-22:00 bump
    base = 0.25 + 0.55 * peak1 + 0.85 * peak2
    base = base / base.max()
    noise = 1.0 + rng.normal(0, 0.04, size=T)
    return np.clip(peak_kwh * base * noise, 0.1, None)


def _synthetic_nondispatch_shape(rng: np.random.Generator, T: int, peak_kwh: float) -> np.ndarray:
    hours = np.arange(T)
    morning = np.exp(-0.5 * ((hours - 8.5) / 2.0) ** 2)
    midday = np.exp(-0.5 * ((hours - 14.0) / 3.5) ** 2)
    evening = np.exp(-0.5 * ((hours - 19.5) / 2.0) ** 2)
    base = 0.30 + 0.35 * morning + 0.45 * midday + 0.55 * evening
    base = base / base.max()
    noise = 1.0 + rng.normal(0, 0.03, size=T)
    return np.clip(peak_kwh * base * noise, 0.05, None)


def build_tier_a_scenario(config_path: str, random_seed: int | None = None) -> Scenario:
    """Build Tier-A scenario. ``random_seed`` overrides the YAML seed so
    multi-seed training/evaluation can generate genuinely different load/PV
    realizations while preserving all fixed source-paper parameters."""
    cfg = load_config(config_path)
    seed = cfg["random_seed"] if random_seed is None else int(random_seed)
    rng = np.random.default_rng(seed)
    T = cfg["time_slots"]
    dt = cfg["delta_t_hours"]

    lam = np.array(cfg["wholesale_price_usd_per_kwh"], dtype=float)
    xi = np.array(cfg["elasticity_xi_t"], dtype=float)
    assert len(lam) == T and len(xi) == T

    eta_lo = cfg["price_bounds"]["eta_min_multiplier"] * lam.min() * np.ones(T)
    eta_hi = cfg["price_bounds"]["eta_max_multiplier"] * lam.max() * np.ones(T)

    gen_loads, gen_demand = [], []
    for spec in cfg["dispatchable_loads"]:
        gen_loads.append(GeneralDispatchableLoad(
            id=spec["id"], alpha=spec["alpha"], beta=spec["beta"],
            dr_min_frac=spec["dr_min_frac"], dr_max_frac=spec["dr_max_frac"]))
        peak = rng.uniform(6, 12)
        gen_demand.append(_synthetic_general_load_shape(rng, T, peak))
    gen_demand = np.vstack(gen_demand)

    non_loads, non_demand = [], []
    n_non = cfg["non_dispatchable_loads"]["count"]
    lo, hi = cfg["non_dispatchable_loads"]["peak_kwh_range"]
    for i in range(n_non):
        non_loads.append(NonDispatchableLoad(id=f"Non{i+1}"))
        peak = rng.uniform(lo, hi)
        non_demand.append(_synthetic_nondispatch_shape(rng, T, peak))
    non_demand = np.vstack(non_demand)

    pevs, pev_demand = [], []
    for spec in cfg["pevs"]:
        pevs.append(PEVModel(id=spec["id"], alpha=spec["alpha"], beta=spec["beta"],
                              C_m=spec["C_m"], C_M=spec["C_M"], mu_c=spec["mu_c"],
                              mu_d=spec["mu_d"], p_rated_kw=spec["p_rated_kw"],
                              E0=spec["E0"], ET=spec["ET"]))
        # e^V_{n,t}: approximated as a uniform 24-hour target (Wan et al.'s own
        # Eq. (4) applies at EVERY t, with no charging-window restriction).
        # CORRECTED (external review, verified by reconstructing SoC from
        # actual No-DR output: shortfalls matched need*(1-mu_c) exactly,
        # e.g. PEV1 8.8 = 88*(1-0.90)): the target must be (ET-E0)/mu_c, not
        # (ET-E0), because Eq. (6)/(7)'s SoC update applies mu_c to whatever
        # is drawn -- requesting exactly (ET-E0) kWh of raw consumption only
        # delivers mu_c*(ET-E0) kWh of actual charge. This does NOT by
        # itself guarantee any baseline reaches the target when price
        # differs from wholesale (Eq. 4's elasticity term still changes
        # realized consumption from the target) -- terminal shortfall
        # remains a genuine, reported diagnostic, not something silently
        # assumed to be zero. See results/baseline_validation_report.md
        # Sec. 2 for actual achieved shortfalls per baseline.
        need = spec["ET"] - spec["E0"]
        target_raw_kwh = need / spec["mu_c"]
        target = np.full(T, target_raw_kwh / T)
        max_deliverable = spec["p_rated_kw"] * spec["mu_c"] * T
        assert max_deliverable >= need, (
            f"{spec['id']}: terminal target {need} kWh exceeds max deliverable "
            f"{max_deliverable} kWh even spread across all {T} hours -- config error")
        pev_demand.append(target)
    pev_demand = np.vstack(pev_demand)

    batteries = []
    n_bat = int(round(cfg["battery"]["enabled_fraction_of_households"] *
                       (len(gen_loads) + len(pevs) + n_non)))
    for i in range(max(n_bat, 1)):
        b = cfg["battery"]
        batteries.append(BatteryModel(id=f"Bat{i+1}", capacity_kwh=b["capacity_kwh"],
                                       p_max_kw=b["p_max_kw"], eta_c=b["eta_c"], eta_d=b["eta_d"],
                                       degradation_usd_per_kwh_throughput=b["degradation_usd_per_kwh_throughput"]))

    n_pv = int(round(cfg["pv"]["enabled_fraction_of_households"] *
                      (len(gen_loads) + len(pevs) + n_non)))
    pv_profiles = np.vstack([
        synthetic_pv_profile(cfg["pv"]["peak_kw"], T, dt, rng=rng)
        for _ in range(max(n_pv, 1))
    ])

    provenance = {
        "wholesale_price": "table", "xi_elasticity": "table", "rho": "table",
        "eta_bounds": "table-derived", "gen_load_params": "table",
        "gen_demand_shape": "approx", "nondispatch_demand_shape": "approx",
        "pev_params": "table", "pev_demand_shape": "approx",
        "battery": "extension", "pv": "extension/approx", "random_seed": seed,
    }

    return Scenario(T=T, dt_hours=dt, wholesale_price=lam, xi_elasticity=xi,
                     rho=cfg["weights"]["rho"], eta_bounds=(eta_lo, eta_hi),
                     gen_loads=gen_loads, gen_demand_kwh=gen_demand,
                     nondispatch_loads=non_loads, nondispatch_demand_kwh=non_demand,
                     pevs=pevs, pev_demand_kwh=pev_demand,
                     batteries=batteries, pv_profiles_kw=pv_profiles,
                     network_config=dict(cfg["network"]),
                     provenance=provenance)

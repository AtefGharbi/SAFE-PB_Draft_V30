"""
General-dispatchable and non-dispatchable household load models.

Implements Wan, Qin, Yu, Yang & Kang (2022), IEEE/CAA JAS 9(1):123-134
("Price-Based Residential Demand Response Management in Smart Grids"):
  - Eq. (1): consumption as a function of announced price (price-elasticity
    reaction curve)
  - Eq. (2): quadratic dissatisfaction cost
  - Eq. (3): demand-reduction bounds
  - Eq. (10): non-dispatchable load (consumption == demand, always)

These are DETERMINISTIC reaction functions, not learned/optimized responses:
given a price, Eq. (1) directly gives the resulting consumption. This
matches the source paper's Algorithm 1, Step S7, which computes consumption
directly from (1) and (4) once the price is chosen -- there is no separate
household-side optimizer in the source formulation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class GeneralDispatchableLoad:
    """General dispatchable load n (e.g. HVAC/appliance), Wan et al. Eqs. (1)-(3)."""
    id: str
    alpha: float   # $/kWh^2, Eq. (2)
    beta: float    # $/kWh, Eq. (2)
    dr_min_frac: float  # DR_min = dr_min_frac * e_{n,t}, Eq. (3)
    dr_max_frac: float  # DR_max = dr_max_frac * e_{n,t}, Eq. (3)

    def consumption(self, e_demand_kwh: np.ndarray, eta_price: np.ndarray,
                     wholesale_price: np.ndarray, xi_elasticity: np.ndarray) -> np.ndarray:
        """Eq. (1): p_{n,t} = e_{n,t} * (1 + xi_t * (eta_{n,t} - lambda_t) / lambda_t)."""
        p = e_demand_kwh * (1.0 + xi_elasticity * (eta_price - wholesale_price) / wholesale_price)
        # Enforce Eq. (3) demand-reduction bounds by clipping consumption to the
        # physically valid range implied by DR_min/DR_max (documented modeling
        # choice: the source paper relies on price bounds (13) to keep (1) inside
        # this range but does not state an explicit clip; we clip for physical
        # realism and log any clipping event).
        dr_min = self.dr_min_frac * e_demand_kwh
        dr_max = self.dr_max_frac * e_demand_kwh
        p_lo = e_demand_kwh - dr_max
        p_hi = e_demand_kwh - dr_min
        clipped = np.clip(p, p_lo, p_hi)
        was_clipped = bool(np.any(np.abs(clipped - p) > 1e-9))
        return clipped, was_clipped

    def dissatisfaction(self, e_demand_kwh: np.ndarray, p_consumption_kwh: np.ndarray) -> np.ndarray:
        """Eq. (2): phi = 0.5*alpha*(e-p)^2 + beta*(e-p)."""
        gap = e_demand_kwh - p_consumption_kwh
        return 0.5 * self.alpha * gap ** 2 + self.beta * gap


@dataclass
class NonDispatchableLoad:
    """Non-dispatchable load n, Wan et al. Eq. (10): p_{n,t} = e_{n,t} always."""
    id: str

    def consumption(self, e_demand_kwh: np.ndarray) -> np.ndarray:
        return np.asarray(e_demand_kwh, dtype=float)

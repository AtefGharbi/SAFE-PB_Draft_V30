"""
Stationary household battery model.

SOURCE = [SAFE-PB EXTENSION]: this device is NOT part of Wan et al. (2022);
it exists only in the SAFE-PB extension of the household portfolio (System
Model Sec. 3.2, "the stationary battery follows the same state equation
with a throughput degradation charge"). Included here so the safety
projection layer (Command 6) has a real controllable device with SoC
constraints to project, and so battery-related unit tests are meaningful.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BatteryModel:
    id: str
    capacity_kwh: float
    p_max_kw: float
    eta_c: float
    eta_d: float
    degradation_usd_per_kwh_throughput: float
    soc0_kwh: float | None = None

    def __post_init__(self):
        if self.soc0_kwh is None:
            self.soc0_kwh = 0.5 * self.capacity_kwh

    def step(self, soc_kwh: float, p_kw: float, dt_hours: float = 1.0):
        """Apply one step of candidate power p_kw (>0 charge, <0 discharge).
        Returns (clipped_p_kw, new_soc_kwh, degradation_cost_usd, was_clipped).
        """
        p = float(np.clip(p_kw, -self.p_max_kw, self.p_max_kw))
        mu = self.eta_c if p > 0 else (1.0 / self.eta_d if p < 0 else 1.0)
        prospective = soc_kwh + mu * p * dt_hours
        clipped = False
        if prospective > self.capacity_kwh:
            p = (self.capacity_kwh - soc_kwh) / (self.eta_c * dt_hours) if p > 0 else p
            prospective = self.capacity_kwh
            clipped = True
        elif prospective < 0.0:
            p = (0.0 - soc_kwh) * self.eta_d / dt_hours if p < 0 else p
            prospective = 0.0
            clipped = True
        degradation = self.degradation_usd_per_kwh_throughput * abs(p) * dt_hours
        return p, prospective, degradation, (abs(p - p_kw) > 1e-9 or clipped)

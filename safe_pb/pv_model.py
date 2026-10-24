"""
Rooftop PV generation model.

SOURCE = [APPROX / SYNTHETIC]: no measured irradiance trace is available in
this sandbox (no general internet access to public solar datasets such as
PVGIS or NSRDB). Generation is produced by a labeled synthetic clear-sky-like
bell curve, seeded for reproducibility. This must NOT be reported as
measured solar data in any results text; it exists to give the safety
projection layer (Command 6) a source of network-relevant variability
(reverse power flow / voltage rise) to project against.
"""
from __future__ import annotations
import numpy as np


def synthetic_pv_profile(peak_kw: float, time_slots: int = 24, dt_hours: float = 1.0,
                          sunrise_hour: float = 6.5, sunset_hour: float = 18.5,
                          rng: np.random.Generator | None = None,
                          cloud_noise_std: float = 0.05) -> np.ndarray:
    """Synthetic daily PV profile as a raised-cosine bell between sunrise/sunset,
    with small multiplicative cloud noise. NOT a measured irradiance trace."""
    if rng is None:
        rng = np.random.default_rng(0)
    hours = np.arange(time_slots) * dt_hours
    day_frac = (hours - sunrise_hour) / (sunset_hour - sunrise_hour)
    shape = np.where((day_frac >= 0) & (day_frac <= 1),
                      np.sin(np.pi * np.clip(day_frac, 0, 1)) ** 1.5, 0.0)
    noise = 1.0 + rng.normal(0.0, cloud_noise_std, size=time_slots)
    profile = np.clip(peak_kw * shape * noise, 0.0, None)
    return profile

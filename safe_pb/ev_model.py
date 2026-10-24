"""
Plug-in electric vehicle (PEV) model.

Implements Wan et al. (2022) Eqs. (4)-(9):
  (4) price-elasticity charging/discharging reaction curve
  (5) rated-power bound
  (6) terminal energy requirement (sum over horizon)
  (7) battery-capacity (SoC) bounds at every time step
  (8) dissatisfaction cost (same quadratic form as general loads)
  (9) linear degradation cost from charge/discharge throughput

Implementation note (documented, not silently approximated): the source
paper's Algorithm 1 computes p^V_{n,t} directly from (4) each step and does
not describe an explicit lower-level SoC-feasibility solve. We therefore
(a) compute the raw price-reaction value from (4), (b) clip it to the rated
power bound (5), and (c) clip it again if it would violate the SoC bounds
(7) at that step. The terminal-energy requirement (6) is NOT enforced by
construction (the source model does not enforce it by construction either);
we instead report the terminal shortfall E_T - E_actual as a diagnostic
metric, which is exactly the kind of gap the manuscript's Sec. 9 baseline
validation is supposed to surface.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


class EVTerminalInfeasible(Exception):
    """Raised when an EV's terminal target cannot be met: no hours remain
    and a real energy gap is still open. An earlier version of
    `viability_floor_kw` silently returned 0.0 in this case, which reads
    identically to 'target already met' -- indistinguishable from success.
    Silently returning 0 here was flagged as wrong in external review.

    DESIGN CONTRACT for callers, stated explicitly (external review's
    point): what a caller does when this is raised depends on context.
      - Diagnostics/analysis (current scope, e.g. run_safety_layer_demo.py):
        catching this, recording the event, and proceeding with a
        conservative floor (0: charge-only, no discharge) is acceptable --
        the goal is to characterize the scenario, not to train on it.
      - Training or final evaluation (NOT yet implemented -- Command 10):
        this must NOT be silently absorbed the same way. The correct
        handling is one of: reject the sampled scenario before training on
        it, truncate the episode at this step with an explicit terminal-
        failure flag/reward penalty, or count it as an infeasible episode
        in reported statistics (excluded from success-rate/welfare
        aggregates, or reported as a separate failure-rate metric). None
        of this exists yet because the training loop itself does not
        exist yet; this note exists so the eventual implementation does
        not silently reuse the diagnostic-context handling in a context
        where it would be wrong.
    """
    def __init__(self, ev_id: str, shortfall_kwh: float):
        self.ev_id = ev_id
        self.shortfall_kwh = shortfall_kwh
        super().__init__(f"{ev_id}: terminal target infeasible, {shortfall_kwh:.2f} kWh "
                          f"short with 0 hours remaining")


@dataclass
class PEVModel:
    id: str
    alpha: float
    beta: float
    C_m: float          # kWh, Eq. (7) lower capacity bound
    C_M: float           # kWh, Eq. (7) upper capacity bound
    mu_c: float          # charging efficiency
    mu_d: float          # discharging efficiency
    p_rated_kw: float    # Eq. (5)
    E0: float            # kWh, initial SoC
    ET: float            # kWh, target terminal SoC

    def simulate(self, e_demand_kwh: np.ndarray, eta_price: np.ndarray,
                 wholesale_price: np.ndarray, xi_elasticity: np.ndarray):
        """Run the PEV through one day. Returns dict of per-step arrays."""
        T = len(e_demand_kwh)
        p = np.zeros(T)
        soc = np.zeros(T + 1)
        soc[0] = self.E0
        clipped_rated = np.zeros(T, dtype=bool)
        clipped_soc = np.zeros(T, dtype=bool)

        for t in range(T):
            # Eq. (4)
            raw = e_demand_kwh[t] * (1.0 + xi_elasticity[t] *
                                      (eta_price[t] - wholesale_price[t]) / wholesale_price[t])
            # Eq. (5)
            bounded = np.clip(raw, -self.p_rated_kw, self.p_rated_kw)
            if abs(bounded - raw) > 1e-9:
                clipped_rated[t] = True

            # Prospective SoC update, Eq. analogous to (6)'s per-step increment
            mu = self.mu_c if bounded > 0 else (1.0 / self.mu_d if bounded < 0 else 1.0)
            prospective_soc = soc[t] + mu * bounded

            # Eq. (7): clip to stay within [C_m, C_M]
            if prospective_soc > self.C_M:
                # solve for the largest bounded (charging) that hits C_M exactly
                bounded = (self.C_M - soc[t]) / self.mu_c if bounded > 0 else bounded
                prospective_soc = self.C_M
                clipped_soc[t] = True
            elif prospective_soc < self.C_m:
                bounded = (self.C_m - soc[t]) * self.mu_d if bounded < 0 else bounded
                prospective_soc = self.C_m
                clipped_soc[t] = True

            p[t] = bounded
            soc[t + 1] = prospective_soc

        terminal_shortfall_kwh = self.ET - soc[-1]  # diagnostic, Eq. (6) gap
        dissatisfaction = 0.5 * self.alpha * (e_demand_kwh - p) ** 2 + self.beta * (e_demand_kwh - p)
        degradation_cost = 0.02 * np.abs(p)  # kappa placeholder; overridden by caller if configured

        return {
            "p_kw": p,
            "soc_kwh": soc[1:],
            "terminal_shortfall_kwh": terminal_shortfall_kwh,
            "clipped_rated_power": clipped_rated,
            "clipped_soc_bound": clipped_soc,
            "dissatisfaction": dissatisfaction,
            "degradation_cost": degradation_cost,
        }

    def viability_floor_kw(self, current_soc_kwh: float, hours_remaining: int,
                            dt_hours: float = 1.0) -> float:
        """Minimum charging power THIS step must deliver to keep the
        terminal target (ET) reachable given the hours left in the horizon.
        Raises EVTerminalInfeasible whenever the target is already
        unreachable -- either because no hours remain, OR because even
        charging at full rated power for every remaining hour (including
        this one) cannot close the gap. An earlier version only checked
        hours_remaining<=0 and otherwise capped the returned floor at
        p_rated_kw, which silently returned a valid-looking-but-
        insufficient number for an already-infeasible case (confirmed:
        88 kWh needed with 2 hours remaining at 9 kW rated power returned
        9.0 instead of raising, when only 16.2 kWh is actually
        deliverable)."""
        need = self.ET - current_soc_kwh
        if need <= 0:
            return 0.0
        if hours_remaining <= 0:
            raise EVTerminalInfeasible(self.id, need)
        max_total_deliverable = self.p_rated_kw * self.mu_c * dt_hours * hours_remaining
        if need > max_total_deliverable + 1e-9:
            raise EVTerminalInfeasible(self.id, need - max_total_deliverable)
        max_from_future_steps = self.p_rated_kw * self.mu_c * dt_hours * max(hours_remaining - 1, 0)
        shortfall_if_skip_now = need - max_from_future_steps
        if shortfall_if_skip_now <= 0:
            return 0.0  # skipping this step still leaves the target reachable
        floor_kw = shortfall_if_skip_now / (self.mu_c * dt_hours)
        return float(min(floor_kw, self.p_rated_kw))

    def simulate_with_viability_enforcement(self, e_demand_kwh: np.ndarray, eta_price: np.ndarray,
                                             wholesale_price: np.ndarray, xi_elasticity: np.ndarray,
                                             dt_hours: float = 1.0):
        """Trajectory-aware version of `simulate()`: at every hour, computes
        the price-reaction action (Eq. 4, clipped to Eq. 5/7 bounds as in
        `simulate()`), then applies `viability_floor_kw` as a LOWER bound
        on that step's charging power using the ACTUAL running SoC -- i.e.
        the viability constraint is enforced online through the
        trajectory, not just available as a disconnected helper. If even
        forcing every remaining hour to full rated power cannot reach the
        target, raises EVTerminalInfeasible with the exact shortfall
        rather than silently returning an under-charged trajectory.

        Returns the same dict shape as `simulate()`, plus
        `viability_forced` (bool array: True where the floor overrode the
        raw price-reaction action)."""
        T = len(e_demand_kwh)
        p = np.zeros(T)
        soc = np.zeros(T + 1)
        soc[0] = self.E0
        viability_forced = np.zeros(T, dtype=bool)

        for t in range(T):
            raw = e_demand_kwh[t] * (1.0 + xi_elasticity[t] *
                                      (eta_price[t] - wholesale_price[t]) / wholesale_price[t])
            bounded = float(np.clip(raw, -self.p_rated_kw, self.p_rated_kw))

            hours_remaining = T - t  # this step counts as one of the remaining hours
            floor = self.viability_floor_kw(soc[t], hours_remaining, dt_hours)
            if floor > bounded:
                bounded = floor
                viability_forced[t] = True

            mu = self.mu_c if bounded > 0 else (1.0 / self.mu_d if bounded < 0 else 1.0)
            prospective_soc = soc[t] + mu * bounded * dt_hours
            if prospective_soc > self.C_M:
                bounded = (self.C_M - soc[t]) / (self.mu_c * dt_hours) if bounded > 0 else bounded
                prospective_soc = self.C_M
            elif prospective_soc < self.C_m:
                bounded = (self.C_m - soc[t]) * self.mu_d / dt_hours if bounded < 0 else bounded
                prospective_soc = self.C_m

            p[t] = bounded
            soc[t + 1] = prospective_soc

        terminal_shortfall_kwh = self.ET - soc[-1]
        if terminal_shortfall_kwh > 1e-6:
            # Even forcing the viability floor at every step (each capped at
            # p_rated_kw within the loop) was not enough -- the target was
            # infeasible for this horizon/rated-power combination from the
            # start. Raise here rather than returning an under-charged
            # trajectory silently; mid-loop hours_remaining never reaches 0
            # for a T-step trajectory (the last step has hours_remaining=1),
            # so this end-of-trajectory check is where genuine infeasibility
            # actually surfaces.
            raise EVTerminalInfeasible(self.id, float(terminal_shortfall_kwh))
        dissatisfaction = 0.5 * self.alpha * (e_demand_kwh - p) ** 2 + self.beta * (e_demand_kwh - p)
        return {
            "p_kw": p, "soc_kwh": soc[1:], "terminal_shortfall_kwh": terminal_shortfall_kwh,
            "viability_forced": viability_forced, "dissatisfaction": dissatisfaction,
        }

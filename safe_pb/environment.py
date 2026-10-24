"""
SAFE-PB training environment (Command 10, third implementation pass).

CORRECTIONS FROM THE SECOND PASS (external review, each verified before fixing):
  1. Non-dispatchable household loads are now actually included in the
     safety projection and AC verification -- an earlier version's
     `build_safety_devices` had a `for ... : pass` loop that silently
     never added them to `fixed_p`/`fixed_q`, so the safety layer and the
     reported minimum voltage never accounted for them at all.
  2. PEV device bounds are now derived from actual SoC, capacity, and the
     terminal-viability floor (`PEVModel.viability_floor_kw`) -- an
     earlier version used static [-p_rated, +p_rated] bounds regardless
     of SoC, so a PEV could be assigned a candidate action that would
     over/undercharge it past its physical capacity or make its terminal
     target unreachable.
  3. The utility-household interaction is now genuinely SEQUENTIAL: the
     utility's price is computed and broadcast FIRST, and household
     observations reflect the CURRENT step's price before the household
     actor samples its action -- an earlier version had both actors act
     from the same stale observation (containing the PREVIOUS step's
     price), so households never actually saw the price they were
     supposedly responding to.
  4. `step()` now returns BOTH the linearized projection voltage and an
     independently-computed AC power-flow voltage in `info`, not just the
     linear estimate.

NEW IN THIS PASS -- battery and PV wiring (previously: "the scenario
builds them, the environment doesn't use them yet"):
  5. Stationary batteries (`scenario.batteries`) are now real controllable
     households: they get an observation row, a device in the safety
     projection with SoC/capacity-aware charge/discharge bounds (mirroring
     the PEV bounds function but with no terminal-energy target), a real
     dynamics update via `BatteryModel.step()` after projection, and a
     throughput-degradation cost (Eq. 9) included in C_t. Battery power is
     billed at the general-dispatchable rate eta_gen -- there is no
     separate eta_bat tariff category in this Tier-A scenario (category-
     level pricing only, per the disclosed limitation), so this is a
     documented modeling choice, not a hidden one. Batteries carry no
     "discomfort" term (Eq. 2 is about requested-vs-delivered flexible
     service, which does not apply to a battery), so their entry in
     `household_discomfort_all` is always 0.0.
  6. Rooftop PV (`scenario.pv_profiles_kw`) is now injected into the
     network as a real negative P injection at its assigned bus (unity
     power factor, i.e. zero reactive contribution) -- it now actually
     affects the linear voltage estimate and the independent AC check,
     which is the reason pv_model.py exists at all ("give the safety
     projection layer a source of network-relevant variability"). PV is
     STILL NOT economically modeled: it does not reduce any household's
     bill, there is no self-consumption or export-price accounting (Eq. 5
     /6's eta_exp), and there is no curtailment control yet. Those remain
     open items -- see the manuscript's Sec. 5.2/8 limitations.

Two features remain exactly as before: EV availability windows (uniform
09:00-17:00 away window, not per-vehicle) and every step routing through
the real safety projection.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .data_loader import Scenario
from .feeder_model import (FeederTopology, assign_households_to_buses,
                           ac_powerflow_check, linearized_voltages)
from .safety_projection import Device, project, ac_corrected_project, authority_certificate
from .ev_model import EVTerminalInfeasible
from .battery_model import BatteryModel


EV_AWAY_START, EV_AWAY_END = 9, 17


def ev_available(hour: int) -> bool:
    return not (EV_AWAY_START <= hour < EV_AWAY_END)


@dataclass
class EnvConfig:
    n_utility_obs: int = 9
    n_utility_act: int = 3     # (eta_non, eta_gen, eta_pev), category-level
    n_household_obs: int = 8
    n_household_act: int = 1   # single power-adjustment dimension per household
    homes_per_bus: int = 1
    # None loads the sole authoritative value from Scenario.network_config.
    # A numeric override is reserved for explicitly labeled stress tests.
    base_scale: float | None = None
    v_min: float = 0.95
    v_max: float = 1.05
    q_over_p: float = 0.35
    pev_infeasibility_policy: str = "terminate_and_penalize"
    pev_infeasibility_penalty_per_kwh: float = 5.0
    enable_safety_projection: bool = True
    # "ac_corrected" (V23 default): lexicographic min-slack projection with
    # AC-measured bias correction. "soft": legacy V21/V22 penalty QP, retained
    # only to reproduce the earlier (mis-scaled) results.
    projection_mode: str = "ac_corrected"
    max_ac_corrections: int = 4
    # Classify each step by whether ANY device-feasible action is AC-feasible
    # (authority certificate, one extra power flow).
    classify_authority: bool = True
    single_price: bool = False
    scenario_label: str = "nominal"
    # Candidate AC voltage is an evaluation diagnostic and does not enter
    # the transition, reward, projection, or next observation. Publication
    # evaluation keeps this enabled; training can disable it safely.
    verify_candidate_ac: bool = True


class SafePBEnv:
    """One episode = one day (scenario.T steps). Household order is fixed:
    [gen_loads..., pevs...] (matches data_loader's Scenario ordering).

    Usage (two-stage, matching the manuscript's utility-leader structure):
        u_obs, _ = env.reset()
        u_act = utility_actor(u_obs)
        h_obs = env.utility_announce_price(u_act)   # households now SEE this step's price
        h_act = household_actor(h_obs)
        next_u_obs, next_h_obs, reward, done, info = env.household_step(h_act)
    `step(u_act, h_act)` is kept as a convenience wrapper for callers that
    don't need the two-stage split (e.g. quick smoke checks), but internally
    it now calls the same two methods in the correct order -- it does NOT
    let households act on stale information either.
    """

    def __init__(self, scenario: Scenario, topo: FeederTopology, cfg: EnvConfig | None = None):
        self.scenario = scenario
        self.topo = topo
        self.cfg = cfg or EnvConfig()
        if self.cfg.base_scale is None:
            self.cfg.base_scale = float(scenario.network_config["base_scale"])
        if self.cfg.pev_infeasibility_policy not in {
            "terminate_and_penalize", "reject_scenario", "continue_and_report"
        }:
            raise ValueError("unknown pev_infeasibility_policy: " + self.cfg.pev_infeasibility_policy)
        self.n_gen = len(scenario.gen_loads)
        self.n_pev = len(scenario.pevs)
        self.n_bat = len(scenario.batteries)
        self.n_pv = scenario.pv_profiles_kw.shape[0] if scenario.pv_profiles_kw.size else 0
        self.n_nondispatch = len(scenario.nondispatch_loads)
        # Household order is fixed: [gen_loads..., pevs..., batteries...].
        # Existing PEV-only code that slices from self.n_gen onward (e.g.
        # `applied_kw[env.n_gen:]`) now also picks up battery entries --
        # batteries are appended AFTER pevs specifically so that no
        # existing `self.n_gen + j` PEV index math changes meaning.
        self.n_households = self.n_gen + self.n_pev + self.n_bat
        household_ids = ([f"gen_{l.id}" for l in scenario.gen_loads] +
                          [f"pev_{e.id}" for e in scenario.pevs] +
                          [f"bat_{b.id}" for b in scenario.batteries])
        non_ids = [f"non_{l.id}" for l in scenario.nondispatch_loads]
        pv_ids = [f"pv_{i}" for i in range(self.n_pv)]
        # non_ids and pv_ids are both "fixed" (uncontrolled) network
        # injections -- non_ids as positive load, pv_ids as negative
        # generation -- assigned buses in the SAME round-robin call so
        # bus assignment stays deterministic under a single seed.
        self.bus_map = assign_households_to_buses(topo, household_ids + non_ids + pv_ids,
                                                    self.cfg.homes_per_bus, seed=1)
        self.household_ids = household_ids
        self.non_ids = non_ids
        self.pv_ids = pv_ids
        self.reset()

    def reset(self):
        self.t = 0
        self.pev_soc = np.array([ev.E0 for ev in self.scenario.pevs], dtype=float)
        self.battery_soc = np.array([b.soc0_kwh for b in self.scenario.batteries], dtype=float)
        self.last_min_voltage_linear = 1.0
        self.last_min_voltage_ac = 1.0
        self.last_prices = np.array([1.0, 1.0, 1.0])
        self._pending_prices = None
        self._pev_infeasibility_events = []  # populated by _pev_bounds when EVTerminalInfeasible occurs
        self._terminal_failure_this_step = False
        self._step_infeasibility_penalty = 0.0
        return self._utility_obs(), self._household_obs_all(self.last_prices)

    def _utility_obs(self) -> np.ndarray:
        T = self.scenario.T
        t = self.t
        lam = self.scenario.wholesale_price
        lam_next = lam[min(t + 1, T - 1)]
        load_forecast = float(self.scenario.gen_demand_kwh[:, t].sum() +
                               self.scenario.nondispatch_demand_kwh[:, t].sum())
        pv_forecast = float(self.scenario.pv_profiles_kw[:, t].sum()) if self.scenario.pv_profiles_kw.size else 0.0
        return np.array([
            np.sin(2 * np.pi * t / T), np.cos(2 * np.pi * t / T),
            lam_next / max(lam.max(), 1e-6), load_forecast / 1000.0, pv_forecast / 100.0,
            0.1,  # forecast_error_std placeholder (no stochastic forecast model yet)
            self.last_min_voltage_ac, float(self.last_prices.mean()) / max(lam.max(), 1e-6),
            0.0,  # cvar_risk_state placeholder (trainer owns the real zeta)
        ], dtype=np.float32)

    def _household_obs_all(self, prices: np.ndarray) -> np.ndarray:
        """`prices` = (eta_non, eta_gen, eta_pev) that households are
        observing THIS step -- the caller controls whether this is the
        current or previous step's price, but `utility_announce_price`
        always passes the current one, which is the only correct usage."""
        obs = np.zeros((self.n_households, self.cfg.n_household_obs), dtype=np.float32)
        t = self.t
        for i, ld in enumerate(self.scenario.gen_loads):
            e = self.scenario.gen_demand_kwh[i, t]
            obs[i] = [e / 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, prices[1], self.last_min_voltage_ac]
        for j, ev in enumerate(self.scenario.pevs):
            i = self.n_gen + j
            avail = 1.0 if ev_available(t) else 0.0
            obs[i] = [0.0, 0.0, self.pev_soc[j] / max(ev.C_M, 1e-6), avail,
                      0.0, 0.0, prices[2], self.last_min_voltage_ac]
        for k, bat in enumerate(self.scenario.batteries):
            i = self.n_gen + self.n_pev + k
            # Batteries reuse observation columns 4/5 (unused by gen/pev
            # rows) for soc_frac/availability, and are billed at the
            # general-dispatchable price (prices[1]) -- see the module
            # docstring for why there is no separate battery tariff.
            obs[i] = [0.0, 0.0, 0.0, 0.0,
                      self.battery_soc[k] / max(bat.capacity_kwh, 1e-6), 1.0,
                      prices[1], self.last_min_voltage_ac]
        return obs

    def _available_hours_remaining(self, hour: int) -> int:
        """Counts only FUTURE hours where the EV is actually available to
        charge (i.e. excludes hours inside the away window) -- an earlier
        version used `scenario.T - hour` directly, which counted away
        hours as usable charging opportunity and could understate urgency,
        risking a terminal target that looks reachable but isn't once the
        away window is correctly excluded."""
        return sum(1 for h in range(hour, self.scenario.T) if ev_available(h))

    def _pev_bounds(self, j: int, ev, hour: int) -> tuple[float, float]:
        """Return capacity- and availability-aware power bounds.

        The terminal lower bound is derived in power space and may be
        negative when enough future charging opportunity remains. This is
        important: a zero minimum charging requirement does not mean that
        discharge must be forbidden; it means discharge is allowed only up
        to the amount that still leaves the terminal target reachable.
        """
        if not ev_available(hour):
            return 0.0, 0.0
        soc = self.pev_soc[j]
        hours_remaining = self._available_hours_remaining(hour)
        future_slots = max(hours_remaining - 1, 0)
        future_deliverable = future_slots * ev.p_rated_kw * ev.mu_c
        required_delta_now = ev.ET - soc - future_deliverable
        max_delta_now = ev.p_rated_kw * ev.mu_c

        if required_delta_now > max_delta_now + 1e-9:
            exc = EVTerminalInfeasible(ev.id, required_delta_now - max_delta_now)
            event = {"pev_id": ev.id, "hour": hour,
                     "shortfall_kwh": exc.shortfall_kwh}
            event_key = (ev.id, hour)
            if not any((e["pev_id"], e["hour"]) == event_key for e in self._pev_infeasibility_events):
                self._pev_infeasibility_events.append(event)
                self._step_infeasibility_penalty += (
                    exc.shortfall_kwh * self.cfg.pev_infeasibility_penalty_per_kwh
                )
            if self.cfg.pev_infeasibility_policy == "reject_scenario":
                raise exc
            if self.cfg.pev_infeasibility_policy == "terminate_and_penalize":
                self._terminal_failure_this_step = True
            terminal_lb_kw = ev.p_rated_kw
        elif required_delta_now >= 0:
            terminal_lb_kw = required_delta_now / max(ev.mu_c, 1e-9)
        else:
            terminal_lb_kw = required_delta_now * ev.mu_d

        max_charge_kw = min(ev.p_rated_kw, max((ev.C_M - soc) / max(ev.mu_c, 1e-6), 0.0))
        max_discharge_kw = min(ev.p_rated_kw, max((soc - ev.C_m) * ev.mu_d, 0.0))
        lb = min(terminal_lb_kw, max_charge_kw)
        lb = max(lb, -max_discharge_kw)
        ub = max_charge_kw
        return float(lb), float(ub)

    def _battery_bounds(self, k: int, bat: BatteryModel) -> tuple[float, float]:
        """Instantaneous charge/discharge power bounds from current SoC and
        capacity -- the same physical-headroom logic as `_pev_bounds`'
        max_charge_kw/max_discharge_kw terms, but with no terminal-energy
        target (a stationary battery has no departure deadline to meet)."""
        soc = self.battery_soc[k]
        max_charge_kw = min(bat.p_max_kw, max((bat.capacity_kwh - soc) / max(bat.eta_c, 1e-6), 0.0))
        max_discharge_kw = min(bat.p_max_kw, max(soc * bat.eta_d, 0.0))
        return float(-max_discharge_kw), float(max_charge_kw)

    def build_safety_devices(self, household_actions_kw: np.ndarray):
        """Builds the Device list + candidate action vector AND the fixed
        (uncontrollable) P/Q injection from non-dispatchable loads -- the
        latter was silently empty in an earlier version."""
        t = self.t
        devices, a_hat = [], []
        for i, ld in enumerate(self.scenario.gen_loads):
            e = self.scenario.gen_demand_kwh[i, t]
            lb, ub = e * (1 - ld.dr_max_frac), e * (1 - ld.dr_min_frac)
            devices.append(Device(name=f"gen_{ld.id}", bus=self.bus_map[f"gen_{ld.id}"],
                                   lb_kw=lb, ub_kw=ub, weight=1.0, q_over_p=self.cfg.q_over_p))
            a_hat.append(np.clip(household_actions_kw[i], lb, ub))
        for j, ev in enumerate(self.scenario.pevs):
            i = self.n_gen + j
            lb, ub = self._pev_bounds(j, ev, t)
            devices.append(Device(name=f"pev_{ev.id}", bus=self.bus_map[f"pev_{ev.id}"],
                                   lb_kw=lb, ub_kw=ub, weight=1.0, q_over_p=self.cfg.q_over_p))
            a_hat.append(np.clip(household_actions_kw[i], lb, ub))
        for k, bat in enumerate(self.scenario.batteries):
            i = self.n_gen + self.n_pev + k
            lb, ub = self._battery_bounds(k, bat)
            devices.append(Device(name=f"bat_{bat.id}", bus=self.bus_map[f"bat_{bat.id}"],
                                   lb_kw=lb, ub_kw=ub, weight=1.0, q_over_p=self.cfg.q_over_p))
            a_hat.append(np.clip(household_actions_kw[i], lb, ub))

        fixed_p, fixed_q = {}, {}
        for i, ld in enumerate(self.scenario.nondispatch_loads):
            kw = float(self.scenario.nondispatch_demand_kwh[i, t])
            bus = self.bus_map[self.non_ids[i]]
            fixed_p[bus] = fixed_p.get(bus, 0.0) + kw * self.cfg.homes_per_bus / 1000.0
            fixed_q[bus] = fixed_q.get(bus, 0.0) + kw * self.cfg.q_over_p * self.cfg.homes_per_bus / 1000.0
        # PV: negative P injection (generation offsets load), unity power
        # factor (no reactive contribution) -- see module docstring point 6
        # for what this does and does NOT yet model.
        for p, pv_id in enumerate(self.pv_ids):
            kw = float(self.scenario.pv_profiles_kw[p, t])
            bus = self.bus_map[pv_id]
            fixed_p[bus] = fixed_p.get(bus, 0.0) - kw * self.cfg.homes_per_bus / 1000.0
        return devices, np.array(a_hat), fixed_p, fixed_q

    def _network_injections(self, devices, actions_kw, fixed_p, fixed_q):
        """Return bus injections for one device-action vector.

        This common path is used for both the candidate schedule and the
        applied post-projection schedule so the two voltage checks differ
        only in the action vector being assessed.
        """
        bus_p, bus_q = dict(fixed_p), dict(fixed_q)
        for device, kw in zip(devices, actions_kw):
            bus_p[device.bus] = (bus_p.get(device.bus, 0.0) +
                                 float(kw) * self.cfg.homes_per_bus / 1000.0)
            bus_q[device.bus] = (bus_q.get(device.bus, 0.0) +
                                 float(kw) * device.q_over_p *
                                 self.cfg.homes_per_bus / 1000.0)
        return bus_p, bus_q

    def utility_announce_price(self, utility_action_norm: np.ndarray) -> np.ndarray:
        """STAGE 1: the utility's action is rescaled to physical prices and
        stored as `self._pending_prices`; returns household observations
        that reflect THIS step's price (not the previous one). Households
        must be sampled from this returned observation, not from a
        pre-existing one -- `household_step` enforces this by requiring
        `utility_announce_price` to have been called first."""
        t = self.t
        self._terminal_failure_this_step = False
        self._step_infeasibility_penalty = 0.0
        eta_lo, eta_hi = self.scenario.eta_bounds[0][t], self.scenario.eta_bounds[1][t]
        a = np.asarray(utility_action_norm, dtype=float).copy()
        if self.cfg.single_price:
            a[:] = float(np.mean(a))
        prices = eta_lo + (a + 1) / 2 * (eta_hi - eta_lo)
        self._pending_prices = prices
        return self._household_obs_all(prices)

    def household_step(self, household_actions_norm: np.ndarray):
        """STAGE 2: applies household actions (which must have been chosen
        using the observation from `utility_announce_price`), runs the
        safety projection (now including fixed non-dispatchable loads),
        applies the result, and returns both linearized and independently
        AC-verified voltage in `info`."""
        if self._pending_prices is None:
            raise RuntimeError("household_step called without a prior utility_announce_price "
                                "this timestep -- households cannot act on undetermined prices")
        t = self.t
        scenario = self.scenario
        eta_non, eta_gen, eta_pev = self._pending_prices
        self.last_prices = self._pending_prices

        household_kw = np.zeros(self.n_households)
        for i, ld in enumerate(scenario.gen_loads):
            e = scenario.gen_demand_kwh[i, t]
            lb, ub = e * (1 - ld.dr_max_frac), e * (1 - ld.dr_min_frac)
            household_kw[i] = lb + (household_actions_norm[i, 0] + 1) / 2 * (ub - lb)
        for j, ev in enumerate(scenario.pevs):
            i = self.n_gen + j
            lb, ub = self._pev_bounds(j, ev, t)
            household_kw[i] = lb + (household_actions_norm[i, 0] + 1) / 2 * (ub - lb) if ub > lb else lb
        for k, bat in enumerate(scenario.batteries):
            i = self.n_gen + self.n_pev + k
            lb, ub = self._battery_bounds(k, bat)
            household_kw[i] = lb + (household_actions_norm[i, 0] + 1) / 2 * (ub - lb) if ub > lb else lb

        devices, a_hat, fixed_p, fixed_q = self.build_safety_devices(household_kw)
        monitored = self.topo.non_slack_buses

        # Evaluate the device-feasible candidate before any network
        # projection. These diagnostics are essential for proving that the
        # safety layer was exposed to a binding constraint rather than an
        # already-feasible schedule.
        candidate_p, candidate_q = self._network_injections(
            devices, a_hat, fixed_p, fixed_q)
        candidate_linear = linearized_voltages(
            self.topo, candidate_p, candidate_q, base_scale=self.cfg.base_scale)
        candidate_ac = (ac_powerflow_check(
            self.topo, candidate_p, candidate_q, base_scale=self.cfg.base_scale)
            if self.cfg.verify_candidate_ac else {"converged": False})
        candidate_min_linear = (float(min(candidate_linear.values()))
                                if candidate_linear else float("nan"))
        candidate_min_ac = (float(candidate_ac.get("min_vm_pu", float("nan")))
                            if candidate_ac.get("converged") else float("nan"))
        candidate_max_ac = (float(candidate_ac.get("max_vm_pu", float("nan")))
                            if candidate_ac.get("converged") else float("nan"))
        candidate_ac_violated = bool(candidate_ac.get("converged") and (
            candidate_min_ac < self.cfg.v_min or candidate_max_ac > self.cfg.v_max))
        if self.cfg.enable_safety_projection:
            if self.cfg.projection_mode == "ac_corrected":
                proj = ac_corrected_project(
                    self.topo, monitored, devices, a_hat, fixed_bus_p_mw=fixed_p,
                    fixed_bus_q_mvar=fixed_q, v_min=self.cfg.v_min, v_max=self.cfg.v_max,
                    homes_per_bus=self.cfg.homes_per_bus, base_scale=self.cfg.base_scale,
                    max_corrections=self.cfg.max_ac_corrections)
                applied_kw = proj["a_star"]
                self.last_min_voltage_linear = min(proj["linear_v_after"].values())
                projection_distance = float(proj["projection_distance"])
                total_slack = float(proj["min_total_slack"])
                joint_projection_uses_slack = total_slack > 1e-6
                ac_corrections_used = int(proj["corrections_used"])
            elif self.cfg.projection_mode == "soft":
                proj = project(self.topo, monitored, devices, a_hat, fixed_bus_p_mw=fixed_p,
                                fixed_bus_q_mvar=fixed_q, v_min=self.cfg.v_min, v_max=self.cfg.v_max,
                                homes_per_bus=self.cfg.homes_per_bus, base_scale=self.cfg.base_scale)
                applied_kw = proj.a_star
                self.last_min_voltage_linear = min(proj.linear_v_after.values()) if proj.linear_v_after else 1.0
                projection_distance = float(proj.projection_distance)
                joint_projection_uses_slack = bool(proj.joint_projection_uses_slack)
                total_slack = float(proj.total_slack)
                ac_corrections_used = 0
            else:
                raise ValueError("unknown projection_mode: " + self.cfg.projection_mode)
        else:
            # Ablation: apply the policy candidate after device-bound clipping,
            # but deliberately bypass the network projection. AC verification
            # below remains active so violations are still measured honestly.
            applied_kw = a_hat.copy()
            self.last_min_voltage_linear = float('nan')
            projection_distance = 0.0
            joint_projection_uses_slack = False
            total_slack = 0.0
            ac_corrections_used = 0

        # Independent AC verification of the SAME applied schedule + fixed
        # loads. Uses SIGNED power (an earlier version used max(kw,0.0),
        # discarding discharge/export entirely, so AC verification did not
        # evaluate the same applied schedule whenever a device discharged --
        # the same class of bug already fixed once in safety_projection.py
        # and reintroduced here).
        authority = (authority_certificate(
            self.topo, devices, fixed_p, fixed_q, homes_per_bus=self.cfg.homes_per_bus,
            base_scale=self.cfg.base_scale, v_min=self.cfg.v_min)
            if self.cfg.classify_authority else {"authority_feasible": None, "best_min_vm_pu": float("nan")})
        ac_p, ac_q = self._network_injections(devices, applied_kw, fixed_p, fixed_q)
        ac_result = ac_powerflow_check(self.topo, ac_p, ac_q, base_scale=self.cfg.base_scale)
        self.last_min_voltage_ac = ac_result.get("min_vm_pu", float("nan")) if ac_result.get("converged") else float("nan")

        # Update PEV SoC from the applied (safety-projected) action,
        # energy-conserving (mu applied consistently with the bounds fn above)
        p_pev = np.zeros(self.n_pev)
        pev_dissatisfaction = np.zeros(self.n_pev)
        pev_degradation = np.zeros(self.n_pev)
        pev_energy_balance_residual = np.zeros(self.n_pev)
        for j, ev in enumerate(scenario.pevs):
            i = self.n_gen + j
            p = applied_kw[i]
            if not ev_available(t):
                p = 0.0
            mu = ev.mu_c if p > 0 else (1.0 / ev.mu_d if p < 0 else 1.0)
            old_soc = self.pev_soc[j]
            new_soc = old_soc + mu * p
            self.pev_soc[j] = float(np.clip(new_soc, ev.C_m, ev.C_M))
            pev_energy_balance_residual[j] = self.pev_soc[j] - old_soc - mu * p
            p_pev[j] = p
            # PEV dissatisfaction (Eq. 8) and degradation (Eq. 9) -- an
            # earlier version's reward omitted both entirely, only
            # including general-dispatchable-load discomfort.
            e_v = scenario.pev_demand_kwh[j, t]
            pev_dissatisfaction[j] = 0.5 * ev.alpha * (e_v - p) ** 2 + ev.beta * (e_v - p)
            pev_degradation[j] = 0.02 * abs(p)  # matches the kappa default used elsewhere in this codebase

        # Update battery SoC from the applied (safety-projected) action via
        # BatteryModel.step(), which already encodes the eta_c/eta_d
        # charge/discharge efficiency and re-clips at the SoC boundary as a
        # numerical safety net (the projection should already respect
        # `_battery_bounds`, but linearization error can leave a tiny
        # residual). Batteries have no availability window (always
        # dispatchable) and no terminal-energy target (Eq. 9's degradation
        # charge is the only battery-specific cost term; there is no
        # battery analogue of PEV dissatisfaction, Eq. 8, since a battery
        # has no requested/delivered service to compare).
        p_bat = np.zeros(self.n_bat)
        bat_degradation = np.zeros(self.n_bat)
        for k, bat in enumerate(scenario.batteries):
            i = self.n_gen + self.n_pev + k
            p_candidate = applied_kw[i]
            clipped_p, new_soc, degradation, _was_clipped = bat.step(
                self.battery_soc[k], p_candidate, dt_hours=1.0)
            self.battery_soc[k] = new_soc
            p_bat[k] = clipped_p
            bat_degradation[k] = degradation

        p_gen = applied_kw[:self.n_gen]
        p_non = scenario.nondispatch_demand_kwh[:, t].copy()
        lam_t = scenario.wholesale_price[t]
        # Battery power is folded into wholesale cost/revenue at the
        # general-dispatchable price eta_gen (no separate eta_bat tariff
        # category -- see module docstring point 5), mirroring exactly how
        # p_pev is folded in at eta_pev.
        c_t = p_non.sum() + p_gen.sum() + p_pev.sum() + p_bat.sum()
        U_t = (eta_non * p_non.sum() + eta_gen * p_gen.sum() + eta_pev * p_pev.sum() +
               eta_gen * p_bat.sum() - lam_t * c_t)
        discomfort = np.array([
            0.5 * ld.alpha * (scenario.gen_demand_kwh[i, t] - p_gen[i]) ** 2 +
            ld.beta * (scenario.gen_demand_kwh[i, t] - p_gen[i])
            for i, ld in enumerate(scenario.gen_loads)
        ])
        # Terminal-energy shortfall penalty, applied only on the last step
        # of the episode: an earlier version had no reward consequence for
        # ending the day short of a PEV's target beyond whatever the
        # (imperfect) viability floor already tried to prevent.
        terminal_penalty = self._step_infeasibility_penalty
        if t == scenario.T - 1:
            for j, ev in enumerate(scenario.pevs):
                shortfall = max(ev.ET - self.pev_soc[j], 0.0)
                terminal_penalty += shortfall * self.cfg.pev_infeasibility_penalty_per_kwh
        # Realized energy charges/credits are included below for both PEVs
        # and batteries so learned policies and controlled baselines use the
        # same accounting boundary.
        # Use the same realized energy-cost accounting as the controlled
        # baselines: charging/consumption is billed at the retail tariff and
        # exports receive the wholesale price. This closes the prior mismatch
        # between the learned method and baseline objective definitions.
        pev_energy_cost = np.where(p_pev >= 0.0, eta_pev * p_pev, lam_t * p_pev)
        bat_energy_cost = np.where(p_bat >= 0.0, eta_gen * p_bat, lam_t * p_bat)
        C_t = (eta_non * p_non.sum() + eta_gen * p_gen.sum() + discomfort.sum() +
               pev_dissatisfaction.sum() + pev_degradation.sum() + pev_energy_cost.sum() +
               bat_degradation.sum() + bat_energy_cost.sum() + terminal_penalty)
        reward = float(scenario.rho * U_t - (1 - scenario.rho) * C_t) / 1000.0

        self._pending_prices = None
        self.t += 1
        terminal_failure = bool(self._terminal_failure_this_step)
        done = self.t >= scenario.T or terminal_failure
        household_discomfort_all = np.concatenate(
            [discomfort, pev_dissatisfaction, np.zeros(self.n_bat)])
        fairness_discomfort = np.concatenate([discomfort, pev_dissatisfaction])
        household_energy_bill = np.concatenate(
            [eta_gen * p_gen, pev_energy_cost, bat_energy_cost])

        info = {
            "projection_distance": projection_distance,
            "projection_active": bool(projection_distance > 1e-8),
            "candidate_actions_kw": a_hat.copy(),
            "applied_actions_kw": applied_kw.copy(),
            "projection_delta_kw": (applied_kw - a_hat).copy(),
            "candidate_min_voltage_linear": candidate_min_linear,
            "candidate_min_voltage_ac": candidate_min_ac,
            "candidate_ac_converged": bool(candidate_ac.get("converged", False)),
            "candidate_ac_safety_violated": candidate_ac_violated,
            "min_voltage_linear": self.last_min_voltage_linear,
            "min_voltage_ac": self.last_min_voltage_ac,
            "ac_converged": bool(ac_result.get("converged", False)),
            "joint_projection_uses_slack": joint_projection_uses_slack,
            "projection_mode": self.cfg.projection_mode if self.cfg.enable_safety_projection else "none",
            "ac_corrections_used": ac_corrections_used,
            # True: some device-feasible action is AC-feasible (violation, if
            # any, is attributable to the controller/projection). False: the
            # exogenous operating point alone forces a violation.
            "authority_feasible": authority["authority_feasible"],
            "authority_best_min_v_ac": authority["best_min_vm_pu"],
            "total_slack": total_slack,
            "ac_safety_violated": bool(ac_result.get("converged") and (
                ac_result.get("min_vm_pu", 1.0) < self.cfg.v_min or
                ac_result.get("max_vm_pu", 0.0) > self.cfg.v_max)),
            "household_discomfort": discomfort,
            "pev_dissatisfaction": pev_dissatisfaction,
            "household_discomfort_all": household_discomfort_all,
            "fairness_discomfort": fairness_discomfort,
            "household_energy_bill": household_energy_bill,
            "p_non": p_non.copy(),
            "p_gen": p_gen.copy(),
            "p_pev": p_pev.copy(),
            "p_bat": p_bat.copy(),
            "battery_soc": self.battery_soc.copy(),
            "battery_degradation": bat_degradation,
            "prices": self.last_prices.copy(),
            "pev_degradation": pev_degradation,
            "pev_energy_balance_residual_kwh": pev_energy_balance_residual,
            "terminal_penalty": terminal_penalty,
            "applied_kw": applied_kw,
            "proposed_kw": household_kw,
            "device_feasible_candidate_kw": a_hat.copy(),
            "projected_kw": applied_kw.copy(),
            "pev_soc": self.pev_soc.copy(),
            "pev_infeasibility_events_so_far": list(self._pev_infeasibility_events),
            "episode_terminated_for_pev_infeasibility": terminal_failure,
            "base_scale": self.cfg.base_scale,
            "homes_per_bus": self.cfg.homes_per_bus,
            "scenario_label": self.cfg.scenario_label,
            "U_t": U_t, "C_t": C_t,
        }
        if done:
            info["terminal_pev_soc"] = self.pev_soc.copy()
            info["terminal_pev_shortfall_kwh"] = np.array(
                [max(ev.ET - self.pev_soc[j], 0.0) for j, ev in enumerate(scenario.pevs)])
        next_u_obs = self._utility_obs() if not done else np.zeros(self.cfg.n_utility_obs, dtype=np.float32)
        next_h_obs = self._household_obs_all(self.last_prices) if not done else np.zeros(
            (self.n_households, self.cfg.n_household_obs), dtype=np.float32)
        return next_u_obs, next_h_obs, reward, done, info

    def step(self, utility_action_norm: np.ndarray, household_actions_norm: np.ndarray):
        """Convenience wrapper for callers not using the two-stage API
        directly. IMPORTANT: this does NOT let households act on stale
        information -- it is only useful when household_actions_norm was
        already chosen independently of price (e.g. random-action smoke
        tests); real training must use utility_announce_price +
        household_step so the household actor's INPUT reflects the
        current price before it decides."""
        self.utility_announce_price(utility_action_norm)
        return self.household_step(household_actions_norm)

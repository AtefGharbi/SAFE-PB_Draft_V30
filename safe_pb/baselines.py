"""
Baselines for Command 9 (controlled baseline validation).

Implemented here, as an approximate reconstruction of Wan et al. (2022) -- see IMPLEMENTATION-CHOICE DEVIATIONS below for exactly where and why it departs from an exact reproduction:
  - No-DR         : loads consume exactly their demand, no price signal.
  - TOU           : a simple fixed two-tier time-of-use price (documented
                     assumption -- the source paper does not itself define
                     a TOU baseline; it is a SAFE-PB-side comparator).
  - Tabular Q-learning : an approximate reconstruction of Wan et al.
                     Algorithm 1, using the source paper's update rule
                     (their Eq. 18) and epsilon-greedy
                     policy, and stopping criterion |Q^k - Q^{k-1}| <= delta.

IMPLEMENTATION-CHOICE DEVIATIONS FROM THE SOURCE PAPER (must be reported,
not hidden, per Command 9):
  1. The source paper's action is, in principle, one retail price per
     individual load per time slot. Its own Fig. 6 reports results for only
     THREE price curves (non-dispatchable / general-dispatchable / PEV), so
     we take the same category-level action here: a_t = (eta_non_t,
     eta_gen_t, eta_pev_t), applied uniformly within each load category.
  2. The source paper discretizes price by 0.1 $/kWh; with a 3-dimensional
     category action that grid is combinatorially large (tens of thousands
     of joint actions). We use a coarser grid (configurable, default 7
     points per dimension) so that tabular Q-learning converges within a
     practical compute budget for this validation pass. This is exactly
     the kind of deviation Command 9 requires us to disclose and explain,
     not paper over.
  3. omega_t (the PEV discharge selling price, Eq. in Sec. II-B) is not
     tabulated in the source paper; we set omega_t = lambda_t (wholesale
     price), a documented default, not a source value.

Reward decomposition note: because consumption at each t depends only on
the price chosen AT t (not on prior prices/state; see Wan et al. Eqs. 1, 4,
10), and gamma = 1, the 24-step episode decomposes into 24 independent
single-step decisions. We exploit this ONLY to build an independent
myopic per-slot benchmark (`myopic_exhaustive_benchmark`) for sanity-
checking Q-learning against a strong reference point -- NOT a certified
global optimum, since PEV state of charge is genuinely path-dependent
across t (see that function's docstring). The Q-learning agent itself
still runs the full sequential Algorithm 1 with no shortcuts.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np

from .data_loader import Scenario


@dataclass
class StepResult:
    U_t: float
    C_t: float
    r_t: float
    p_non: np.ndarray
    p_gen: np.ndarray
    p_pev: np.ndarray


def _ev_available_publication(hour: int) -> bool:
    return not (9 <= hour < 17)


def step_reward(scenario: Scenario, t: int, eta_non: float, eta_gen: float, eta_pev: float,
                 pev_soc: np.ndarray, enforce_viability: bool = True,
                 respect_ev_availability: bool = False) -> StepResult:
    """Compute one time slot's reward exactly per Wan et al. Eqs. (1)-(12),
    under the category-price simplification documented above. `pev_soc` is
    the CURRENT per-PEV state of charge (mutated by the caller across t)."""
    lam_t = scenario.wholesale_price[t]
    xi_t = scenario.xi_elasticity[t]
    omega_t = lam_t  # documented default, see module docstring point 3

    # non-dispatchable: consumption == demand always (Eq. 10)
    p_non = scenario.nondispatch_demand_kwh[:, t].copy()

    # general dispatchable (Eq. 1, clipped to Eq. 3 DR bounds)
    e_gen = scenario.gen_demand_kwh[:, t]
    p_gen = np.zeros_like(e_gen)
    phi_gen = np.zeros_like(e_gen)
    for i, ld in enumerate(scenario.gen_loads):
        p, _ = ld.consumption(np.array([e_gen[i]]), np.array([eta_gen]),
                               np.array([lam_t]), np.array([xi_t]))
        p_gen[i] = p[0]
        phi_gen[i] = ld.dissatisfaction(np.array([e_gen[i]]), p)[0]

    # PEV (Eq. 4-5, clipped to rated power and SoC bounds; SoC mutated in place)
    e_pev = scenario.pev_demand_kwh[:, t]
    p_pev = np.zeros_like(e_pev)
    phi_pev = np.zeros_like(e_pev)
    deg_pev = np.zeros_like(e_pev)
    net_pev = np.zeros_like(e_pev)
    for i, ev in enumerate(scenario.pevs):
        raw = e_pev[i] * (1.0 + xi_t * (eta_pev - lam_t) / lam_t)
        bounded = float(np.clip(raw, -ev.p_rated_kw, ev.p_rated_kw))

        if respect_ev_availability and not _ev_available_publication(t):
            bounded = 0.0

        if enforce_viability and (not respect_ev_availability or
                                  _ev_available_publication(t)):
            # External review: terminal EV enforcement must apply
            # consistently to every baseline, not be a disconnected
            # helper. hours_remaining counts THIS step as one of the
            # remaining hours (matches PEVModel.simulate_with_viability_
            # enforcement's convention).
            hours_remaining = (scenario.T - t if not respect_ev_availability else
                               sum(_ev_available_publication(h)
                                   for h in range(t, scenario.T)))
            floor = ev.viability_floor_kw(pev_soc[i], hours_remaining)
            if floor > bounded:
                bounded = floor

        mu = ev.mu_c if bounded > 0 else (1.0 / ev.mu_d if bounded < 0 else 1.0)
        prospective = pev_soc[i] + mu * bounded
        if prospective > ev.C_M:
            bounded = (ev.C_M - pev_soc[i]) / ev.mu_c if bounded > 0 else bounded
            prospective = ev.C_M
        elif prospective < ev.C_m:
            bounded = (ev.C_m - pev_soc[i]) * ev.mu_d if bounded < 0 else bounded
            prospective = ev.C_m
        pev_soc[i] = prospective
        p_pev[i] = bounded
        phi_pev[i] = 0.5 * ev.alpha * (e_pev[i] - bounded) ** 2 + ev.beta * (e_pev[i] - bounded)
        deg_pev[i] = 0.02 * abs(bounded)
        # Eq. net electricity cost, Sec. II-B: C^V,net
        if bounded > 0:
            net_pev[i] = eta_pev * bounded
        else:
            net_pev[i] = omega_t * bounded

    c_t = p_non.sum() + p_gen.sum() + p_pev.sum()
    U_t = eta_non * p_non.sum() + eta_gen * p_gen.sum() + eta_pev * p_pev.sum() - lam_t * c_t
    C_t = (eta_non * p_non.sum() + (eta_gen * p_gen.sum() + phi_gen.sum())
           + (phi_pev.sum() + deg_pev.sum() + net_pev.sum()))
    r_t = scenario.rho * U_t - (1 - scenario.rho) * C_t
    return StepResult(U_t=U_t, C_t=C_t, r_t=r_t, p_non=p_non, p_gen=p_gen, p_pev=p_pev)


def _init_soc(scenario: Scenario) -> np.ndarray:
    return np.array([ev.E0 for ev in scenario.pevs], dtype=float)


def run_no_dr(scenario: Scenario, respect_ev_availability: bool = False):
    """TRUE no-demand-response baseline: every load consumes exactly its
    demand (p = e), bypassing Eq. (1)/(4)'s price-reaction formula entirely
    -- NOT the same as setting eta=lambda and running it through the normal
    model. An earlier version of this function did the latter, which is
    wrong: even at eta=lambda, Eq. (3)'s DR_min=0.1*e still forces general
    loads to shed >=10% of demand (see test_general_dispatchable_price_
    equals_wholesale_still_hits_dr_min), so that was a "minimal markup"
    baseline, not a genuine no-response one. Caught in external review,
    verified, and fixed here."""
    T = scenario.T
    results = []
    for t in range(T):
        p_non = scenario.nondispatch_demand_kwh[:, t].copy()
        p_gen = scenario.gen_demand_kwh[:, t].copy()      # p == e, no shedding
        p_pev = scenario.pev_demand_kwh[:, t].copy()       # p == e, no shifting
        if respect_ev_availability and not _ev_available_publication(t):
            p_pev[:] = 0.0
        lam_t = scenario.wholesale_price[t]
        c_t = p_non.sum() + p_gen.sum() + p_pev.sum()
        # Utility charges wholesale price with zero markup -> zero margin by
        # construction (a genuine no-DR reference has no pricing program at
        # all; U_t=0 reflects that no revenue-generating tariff exists here).
        U_t = 0.0
        C_t = lam_t * c_t   # customers pay wholesale-equivalent cost, no dissatisfaction/degradation
        r_t = scenario.rho * U_t - (1 - scenario.rho) * C_t
        results.append(StepResult(U_t=U_t, C_t=C_t, r_t=r_t, p_non=p_non, p_gen=p_gen, p_pev=p_pev))
    return results


def run_tou(scenario: Scenario, peak_hours=range(17, 22), peak_multiplier=1.6,
            offpeak_multiplier=1.05, respect_ev_availability: bool = False):
    """Simple fixed 2-tier TOU price on top of wholesale price.
    Documented SAFE-PB-side baseline, not from the source paper."""
    soc = _init_soc(scenario)
    T = scenario.T
    results = []
    for t in range(T):
        lam_t = scenario.wholesale_price[t]
        mult = peak_multiplier if t in peak_hours else offpeak_multiplier
        eta = lam_t * mult
        eta_lo, eta_hi = scenario.eta_bounds[0][t], scenario.eta_bounds[1][t]
        eta = float(np.clip(eta, eta_lo, eta_hi))
        r = step_reward(scenario, t, eta, eta, eta, soc,
                        respect_ev_availability=respect_ev_availability)
        results.append(r)
    return results


def myopic_exhaustive_benchmark(scenario: Scenario, grid_points: int = 15):
    """A greedy, per-slot exhaustive-action search: at each t, grid-search
    the price triple that maximizes THAT SLOT's immediate reward given the
    SoC trajectory realized so far, then lock it in and move to t+1.

    IMPORTANT CORRECTION (caught in external review): this is NOT a
    certified global optimum, and an earlier version of this module called
    it "brute_force_optimum," which overstated what it proves. For general
    and non-dispatchable loads the reward really is decoupled across t (see
    module docstring), so greedy-per-slot IS globally optimal for those
    components. But PEV state of charge is genuinely path-dependent: Eq.
    (7)'s capacity clipping means a locally suboptimal PEV action at t can
    leave more SoC headroom that enables a much better outcome later, and
    the terminal-energy target (Eq. 6) is a horizon-wide requirement this
    greedy search does not look ahead for. A certified global optimum would
    need a horizon-wide dynamic program or MILP over the joint SoC state,
    which this function does not implement. Treat its output as a strong
    reference point (it still cannot do worse than any policy that ignores
    the future entirely), not as ground truth for an "optimality gap" claim.
    """
    soc = _init_soc(scenario)
    T = scenario.T
    results = []
    chosen = []
    for t in range(T):
        eta_lo, eta_hi = scenario.eta_bounds[0][t], scenario.eta_bounds[1][t]
        grid = np.linspace(eta_lo, eta_hi, grid_points)
        best = None
        for eta_non, eta_gen, eta_pev in itertools.product(grid, grid, grid):
            soc_try = soc.copy()
            r = step_reward(scenario, t, eta_non, eta_gen, eta_pev, soc_try)
            if best is None or r.r_t > best[0].r_t:
                best = (r, (eta_non, eta_gen, eta_pev), soc_try)
        results.append(best[0])
        chosen.append(best[1])
        soc = best[2]
    return results, chosen


# Backward-compatible alias -- DO NOT use in new code or in any text that
# claims optimality; see the correction in myopic_exhaustive_benchmark's
# docstring above.
brute_force_optimum = myopic_exhaustive_benchmark


def terminal_soc_shortfall(scenario: Scenario, results: list) -> list:
    """Reconstruct each PEV's realized SoC trajectory from a sequence of
    StepResults and return (ET - achieved_SoC) per PEV -- the genuine,
    reported gap between Eq. (6)'s terminal target and what a given
    baseline's chosen prices actually achieved. This is diagnostic, not
    enforcement: nothing in this codebase claims a baseline reaches its
    EV terminal targets by construction (see ev_model.py's module
    docstring) -- reporting the achieved number, not just theoretical
    feasibility of the target, is what closes that gap. Added after
    external review correctly pointed out that a feasible target and an
    achieved target are two different claims."""
    shortfalls = []
    for i, ev in enumerate(scenario.pevs):
        soc = ev.E0
        for t in range(scenario.T):
            p = results[t].p_pev[i]
            mu = ev.mu_c if p > 0 else (1.0 / ev.mu_d if p < 0 else 1.0)
            soc += mu * p
        shortfalls.append(float(ev.ET - soc))
    return shortfalls


@dataclass
class QLearningConfig:
    grid_points: int = 7
    n_episodes: int = 4000
    theta: float = 0.9     # learning rate, source paper Table II
    delta: float = 0.01    # stopping tolerance, source paper Table II
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 3000
    seed: int = 0
    respect_ev_availability: bool = False


class TabularQLearningAgent:
    """Approximate reconstruction of Wan et al. (2022) Algorithm 1 (see
    module docstring for the specific, documented deviations -- this is
    NOT an exact reproduction).

    State = time index t (0..T-1); Action = discretized (eta_non, eta_gen,
    eta_pev) triple (see module docstring for the category-action and
    coarser-grid deviations). Update rule matches source Eq. (18):
        Q^k(s,a) <- (1-theta) Q^{k-1}(s,a) + theta [ r + gamma * max_a' Q^{k-1}(s',a') ]
    with gamma = 1 (source paper: "the discount factor is 1 in our problem
    formulation").
    """

    def __init__(self, scenario: Scenario, cfg: QLearningConfig):
        self.scenario = scenario
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.T = scenario.T
        self.action_grids = []
        for t in range(self.T):
            lo, hi = scenario.eta_bounds[0][t], scenario.eta_bounds[1][t]
            g = np.linspace(lo, hi, cfg.grid_points)
            self.action_grids.append(np.array(list(itertools.product(g, g, g))))
        self.n_actions = len(self.action_grids[0])
        self.Q = np.zeros((self.T, self.n_actions))
        self.Q_history = []          # for convergence diagnostics
        self.max_delta_history = []

    def _epsilon(self, ep: int) -> float:
        frac = min(ep / max(self.cfg.epsilon_decay_episodes, 1), 1.0)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def train(self):
        cfg = self.cfg
        converged_at = None
        for ep in range(cfg.n_episodes):
            Q_prev_episode = self.Q.copy()
            soc = _init_soc(self.scenario)
            eps = self._epsilon(ep)
            for t in range(self.T):
                if self.rng.random() < eps:
                    a_idx = self.rng.integers(0, self.n_actions)
                else:
                    a_idx = int(np.argmax(self.Q[t]))
                eta_non, eta_gen, eta_pev = self.action_grids[t][a_idx]
                r = step_reward(self.scenario, t, eta_non, eta_gen, eta_pev, soc,
                                respect_ev_availability=cfg.respect_ev_availability)
                # gamma = 1; bootstrap on next state's max Q (0 for terminal t=T-1)
                next_max = np.max(self.Q[t + 1]) if t + 1 < self.T else 0.0
                target = r.r_t + 1.0 * next_max
                self.Q[t, a_idx] = (1 - cfg.theta) * self.Q[t, a_idx] + cfg.theta * target

            max_delta = float(np.max(np.abs(self.Q - Q_prev_episode)))
            self.max_delta_history.append(max_delta)
            if ep % 50 == 0 or ep == cfg.n_episodes - 1:
                self.Q_history.append((ep, self.Q.copy()))
            if max_delta <= cfg.delta and converged_at is None and ep > 10:
                converged_at = ep
        return converged_at

    def greedy_policy_rollout(self):
        soc = _init_soc(self.scenario)
        results = []
        chosen = []
        for t in range(self.T):
            a_idx = int(np.argmax(self.Q[t]))
            eta_non, eta_gen, eta_pev = self.action_grids[t][a_idx]
            r = step_reward(self.scenario, t, eta_non, eta_gen, eta_pev, soc,
                            respect_ev_availability=self.cfg.respect_ev_availability)
            results.append(r)
            chosen.append((eta_non, eta_gen, eta_pev))
        return results, chosen

    def greedy_price_schedule(self) -> list[tuple[float, float, float]]:
        """Return the learned 24-hour price policy without executing it.

        The schedule can then be evaluated on held-out scenario realizations,
        keeping training seeds independent from evaluation uncertainty seeds.
        """
        return [tuple(self.action_grids[t][int(np.argmax(self.Q[t]))])
                for t in range(self.T)]


def run_price_schedule(scenario: Scenario, prices,
                       respect_ev_availability: bool = False) -> list[StepResult]:
    """Execute a fixed category-price schedule on a supplied scenario."""
    if len(prices) != scenario.T:
        raise ValueError(f"expected {scenario.T} price actions, got {len(prices)}")
    soc = _init_soc(scenario)
    results = []
    for t, action in enumerate(prices):
        results.append(step_reward(
            scenario, t, *map(float, action), soc,
            respect_ev_availability=respect_ev_availability))
    return results


def run_rolling_horizon_mpc(scenario: Scenario, horizon: int = 3,
                            grid_points: int = 3, beam_width: int = 8,
                            respect_ev_availability: bool = False):
    """Approximate price MPC using a documented receding-horizon beam search.

    At each hour, the controller searches category-price sequences over a
    finite horizon, applies only the first action, observes the updated PEV
    state, and solves again. It is a deployable causal comparator, not a
    clairvoyant full-day optimum. The modest grid and beam width keep the
    publication protocol practical across 20 or more seeds.
    """
    if horizon < 1 or grid_points < 2 or beam_width < 1:
        raise ValueError("horizon>=1, grid_points>=2, and beam_width>=1 are required")
    soc = _init_soc(scenario)
    results, chosen = [], []
    for t0 in range(scenario.T):
        beams = [(0.0, soc.copy(), [])]
        stop = min(scenario.T, t0 + horizon)
        for t in range(t0, stop):
            lo, hi = scenario.eta_bounds[0][t], scenario.eta_bounds[1][t]
            grid = np.linspace(lo, hi, grid_points)
            actions = list(itertools.product(grid, grid, grid))
            expanded = []
            for score, state, seq in beams:
                for action in actions:
                    state_try = state.copy()
                    result = step_reward(
                        scenario, t, *action, state_try,
                        respect_ev_availability=respect_ev_availability)
                    expanded.append((score + float(result.r_t), state_try, seq + [action]))
            expanded.sort(key=lambda item: item[0], reverse=True)
            beams = expanded[:beam_width]
        action = beams[0][2][0]
        result = step_reward(
            scenario, t0, *action, soc,
            respect_ev_availability=respect_ev_availability)
        results.append(result)
        chosen.append(tuple(map(float, action)))
    return results, chosen

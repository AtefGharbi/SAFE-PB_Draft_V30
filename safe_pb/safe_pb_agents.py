"""
SAFE-PB hierarchical actor-critic architecture (manuscript Sec. 5.1 / Command 4).

STATUS UPDATE: a real PyTorch implementation now exists in
`safe_pb.sac_trainer` (real gradient descent, twin critics, target
networks, alternating updates, automatic entropy tuning, CVaR + fairness
terms) and `safe_pb.environment` (a real step/reset environment that runs
every step through the safety projection, with EV availability windows).
This numpy-only module is KEPT as the original obs/action-space
specification (it is what Command 8's file structure asked for), not
deleted, but `HierarchicalSAFEPBAgent`/`train_step` below are no longer
where any real training happens -- see `sac_trainer.HierarchicalSACTrainer`
and `tests/test_sac_training.py` for that.

ORIGINAL SCOPE NOTE (kept for history): this module implements the
ARCHITECTURE (utility actor, parameter-shared household actor, centralized
critics, bounded Gaussian action heads). It is deliberately dependency-
light (pure NumPy, no PyTorch/JAX) so it installs and runs anywhere, but
that also means the update rules below are simplified placeholders
(vanilla REINFORCE-style gradient steps), not real training.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _init_layer(rng, n_in, n_out, scale=0.1):
    return rng.normal(0, scale, size=(n_in, n_out)), np.zeros(n_out)


class GaussianMLPPolicy:
    """Small 2-layer MLP producing a bounded Gaussian action (tanh-squashed
    mean, learned log-std), matching the manuscript's 'bounded Gaussian
    policy' description (Sec. 5.1)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1, self.b1 = _init_layer(rng, obs_dim, hidden)
        self.W2, self.b2 = _init_layer(rng, hidden, act_dim)
        self.log_std = np.full(act_dim, -1.0)
        self.obs_dim, self.act_dim = obs_dim, act_dim

    def forward(self, obs: np.ndarray):
        h = np.tanh(obs @ self.W1 + self.b1)
        mean = np.tanh(h @ self.W2 + self.b2)   # in [-1, 1]; caller rescales to bounds
        return mean

    def sample(self, obs: np.ndarray, rng: np.random.Generator):
        mean = self.forward(obs)
        std = np.exp(self.log_std)
        raw = mean + std * rng.normal(size=self.act_dim)
        return np.clip(raw, -1.0, 1.0), mean, std


class CentralizedCritic:
    """Q(s_global, a_global) -> scalar. Centralized-training-only, per
    manuscript Sec. 5.1 (execution uses only local info; the critic is not
    part of the execution-time policy)."""

    def __init__(self, global_obs_dim: int, global_act_dim: int, hidden: int = 64, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.W1, self.b1 = _init_layer(rng, global_obs_dim + global_act_dim, hidden)
        self.W2, self.b2 = _init_layer(rng, hidden, 1)

    def forward(self, global_obs: np.ndarray, global_act: np.ndarray) -> float:
        x = np.concatenate([global_obs, global_act])
        h = np.tanh(x @ self.W1 + self.b1)
        return float(h @ self.W2 + self.b2)


@dataclass
class UtilityObsSpec:
    """Command 4: utility observation = time, wholesale/PV/load forecasts +
    errors, feeder-state summary, previous prices, risk state."""
    fields: tuple = ("time_sin", "time_cos", "wholesale_price_forecast",
                      "load_forecast", "pv_forecast", "forecast_error_std",
                      "feeder_voltage_summary", "prev_price", "cvar_risk_state")


@dataclass
class HouseholdObsSpec:
    """Command 4: household observation = local demand, indoor/service
    state, EV/battery SoC + availability, PV, current price, limited
    feeder signal."""
    fields: tuple = ("local_demand", "service_state", "ev_soc", "ev_available",
                      "battery_soc", "pv_now", "current_price", "limited_feeder_signal")


class HierarchicalSAFEPBAgent:
    """Ties the utility actor, shared household actor, and centralized
    critics together with the interfaces the manuscript specifies. See the
    module docstring: `train_step` is a REINFORCE-style placeholder, not
    the full SAC update -- calling it does not reproduce Command 10."""

    def __init__(self, utility_obs_dim: int, household_obs_dim: int,
                 n_price_zones: int, household_act_dim: int, seed: int = 0):
        self.utility_actor = GaussianMLPPolicy(utility_obs_dim, n_price_zones, seed=seed)
        self.household_actor = GaussianMLPPolicy(household_obs_dim, household_act_dim, seed=seed + 1)
        self.critic1 = CentralizedCritic(utility_obs_dim + household_obs_dim,
                                          n_price_zones + household_act_dim, seed=seed + 2)
        self.critic2 = CentralizedCritic(utility_obs_dim + household_obs_dim,
                                          n_price_zones + household_act_dim, seed=seed + 3)
        self.cvar_threshold = 0.0   # zeta, Eq. (9) auxiliary variable
        self.entropy_coef = 0.1
        self.rng = np.random.default_rng(seed + 100)

    def act(self, utility_obs, household_obs):
        price_action, _, _ = self.utility_actor.sample(utility_obs, self.rng)
        device_action, _, _ = self.household_actor.sample(household_obs, self.rng)
        return price_action, device_action

    def train_step(self, *args, **kwargs):
        # NOTE, stated explicitly because an earlier round's report
        # inaccurately implied otherwise: this method DOES still raise.
        # It was never converted into a working wrapper around the real
        # trainer, because the two classes' interfaces don't match
        # (numpy arrays + a bespoke .sample(obs, rng) signature here vs.
        # torch tensors + the two-stage environment API in sac_trainer.py)
        # -- attempting a compatibility shim risked new bugs for no real
        # benefit, since HierarchicalSACTrainer is directly usable on its
        # own. Use that class for anything real; this one still just raises.
        raise NotImplementedError(
            "This numpy-only class was a pre-implementation architecture "
            "SCAFFOLD (obs/action space definitions, forward-pass shape "
            "checks) and intentionally never trained anything. A real "
            "PyTorch implementation now exists: see "
            "safe_pb.sac_trainer.HierarchicalSACTrainer, which has actual "
            "gradient descent, twin critics, target networks, alternating "
            "utility/household updates, automatic entropy tuning, a CVaR "
            "auxiliary loss, and fairness regularization -- tested in "
            "tests/test_sac_training.py (parameters update, losses stay "
            "finite, checkpoints restore exactly, a full episode runs "
            "through the real safety-projected environment). This class "
            "is kept only as the original obs/action-space specification "
            "reference; use HierarchicalSACTrainer for anything real."
        )

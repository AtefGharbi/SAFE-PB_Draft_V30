"""
SAC actor/critic network architectures (Command 10, first implementation
pass). This replaces the numpy-only scaffold in `safe_pb_agents.py` (which
is kept as an architecture-spec reference, not deleted, but its
`train_step` now points here) with genuine PyTorch modules: real gradient
descent, real backpropagation, real optimizer state.

SCOPE STATEMENT: this is a first, tested implementation of the network
architecture and one training step -- not the full 20-seed protocol, not
ablations, not populated manuscript results (explicitly out of scope for
this pass, per instruction). What IS real: parameters update via gradient
descent, losses are finite, checkpoints save and restore exactly, and a
full episode runs end-to-end through the environment and safety layer.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class GaussianActor(nn.Module):
    """Bounded Gaussian policy: outputs a tanh-squashed mean and a learned
    log-std per action dimension, matching the manuscript's Sec. 5.1
    "bounded Gaussian policy" description. Used for both the utility actor
    and the parameter-shared household actor (same class, different obs/
    action dimensions and separate weights)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        """Returns (action in [-1,1]^act_dim, log_prob, tanh_mean). Uses the
        standard SAC reparameterized-sampling + tanh-squash correction so
        gradients flow through the sampled action."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        if deterministic:
            pre_tanh = mean
        else:
            eps = torch.randn_like(mean)
            pre_tanh = mean + std * eps  # reparameterization trick
        action = torch.tanh(pre_tanh)
        # log-prob of a tanh-squashed Gaussian (SAC's standard correction)
        if deterministic:
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            normal_log_prob = (-0.5 * ((pre_tanh - mean) / (std + 1e-8)) ** 2
                                - log_std - 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(-1)
            correction = torch.log(1 - action.pow(2) + 1e-6).sum(-1)
            log_prob = normal_log_prob - correction
        return action, log_prob, torch.tanh(mean)


class TwinCritic(nn.Module):
    """Centralized Q(s_global, a_global) -> scalar, two independent heads
    (SAC's twin-critic trick to reduce overestimation bias). Centralized-
    training-only: at execution time only the actors are used (Sec. 5.1)."""

    def __init__(self, global_obs_dim: int, global_act_dim: int, hidden: int = 128):
        super().__init__()
        in_dim = global_obs_dim + global_act_dim
        self.q1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, global_obs: torch.Tensor, global_act: torch.Tensor):
        x = torch.cat([global_obs, global_act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak-averaged target-network update: target <- tau*source + (1-tau)*target."""
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())

"""Fixed-size FIFO replay buffer. Structural component for Command 10's
full training loop; used here only by unit tests to confirm the interface
is correct."""
from __future__ import annotations
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim))
        self.next_obs = np.zeros((capacity, obs_dim))
        self.act = np.zeros((capacity, act_dim))
        self.rew = np.zeros(capacity)
        self.done = np.zeros(capacity, dtype=bool)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = self.rng.integers(0, self.size, size=min(batch_size, self.size))
        return (self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx])

    def __len__(self):
        return self.size

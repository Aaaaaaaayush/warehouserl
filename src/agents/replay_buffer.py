"""
replay_buffer.py
----------------
Shared experience replay buffer for QMIX training.

WHY THIS EXISTS:
  Neural networks learn best from diverse, uncorrelated samples.
  If you train on the last 5 steps in order, each update is strongly
  correlated with the previous one — the network overfits to recent
  experience and "forgets" what it learned earlier.

  A replay buffer stores a large pool of past experiences and serves
  random mini-batches for each training update. This breaks temporal
  correlation and dramatically stabilises learning.

  "Shared" means all agents' transitions go into the same buffer —
  QMIX needs joint transitions (all agents' observations, actions,
  and rewards together) rather than per-agent buffers.

KEY CONCEPTS:
  - Episode-based storage: QMIX requires full episode trajectories
    (not individual transitions) because the GRU needs sequential obs.
    Each stored "episode" is a T-step sequence for all N agents.
  - Circular buffer: once full, new episodes overwrite the oldest.
    This keeps memory bounded without sorting or pruning.
  - [V2-READY]: prioritised=True parameter stub for Prioritised
    Experience Replay (PER) — V2 can weight important transitions higher.

STORED PER EPISODE:
  obs[T, N, obs_size]        — observation for each agent at each step
  actions[T, N]              — action taken by each agent at each step
  rewards[T, N]              — reward received by each agent at each step
  obs_next[T, N, obs_size]   — next observation
  terminated[T, N]           — whether agent was terminated at step T
  state[T, state_size]       — global state (for QMIX mixing network)
  state_next[T, state_size]  — next global state
"""

from __future__ import annotations

import numpy as np
import torch
from collections import deque
from typing import NamedTuple


class EpisodeBatch(NamedTuple):
    """A sampled mini-batch of episodes, ready for training."""
    obs:        torch.Tensor   # (batch, T, N, obs_size)
    actions:    torch.Tensor   # (batch, T, N)
    rewards:    torch.Tensor   # (batch, T, N)
    obs_next:   torch.Tensor   # (batch, T, N, obs_size)
    terminated: torch.Tensor   # (batch, T, N)
    state:      torch.Tensor   # (batch, T, state_size)
    state_next: torch.Tensor   # (batch, T, state_size)
    filled:     torch.Tensor   # (batch, T) — 1 if step exists, 0 if padded


class ReplayBuffer:
    """
    Circular episode replay buffer for QMIX.

    Stores full episodes (not individual transitions) to support
    the GRU-based Q-networks that require sequential observations.

    Parameters
    ----------
    capacity : int
        Maximum number of episodes to store. Oldest are overwritten
        when the buffer is full.
    n_agents : int
        Number of agents per episode.
    obs_size : int
        Observation vector length per agent.
    state_size : int
        Global state vector length.
    max_ep_len : int
        Maximum steps per episode. Shorter episodes are zero-padded.
    prioritised : bool
        [V2-READY] If True, enables prioritised experience replay.
        Currently raises NotImplementedError — stub for V2.
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_size: int,
        state_size: int,
        max_ep_len: int,
        prioritised: bool = False,   # [V2-READY]
    ):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_size = obs_size
        self.state_size = state_size
        self.max_ep_len = max_ep_len

        if prioritised:
            raise NotImplementedError(
                "[V2-READY] Prioritised replay not yet implemented. "
                "Set prioritised=False for V1 training."
            )

        # Pre-allocate numpy arrays for efficiency.
        # Each episode occupies one slot.
        shape_obs   = (capacity, max_ep_len, n_agents, obs_size)
        shape_act   = (capacity, max_ep_len, n_agents)
        shape_rew   = (capacity, max_ep_len, n_agents)
        shape_term  = (capacity, max_ep_len, n_agents)
        shape_state = (capacity, max_ep_len, state_size)
        shape_fill  = (capacity, max_ep_len)

        self._obs        = np.zeros(shape_obs,   dtype=np.float32)
        self._actions    = np.zeros(shape_act,   dtype=np.int64)
        self._rewards    = np.zeros(shape_rew,   dtype=np.float32)
        self._obs_next   = np.zeros(shape_obs,   dtype=np.float32)
        self._terminated = np.zeros(shape_term,  dtype=np.float32)
        self._state      = np.zeros(shape_state, dtype=np.float32)
        self._state_next = np.zeros(shape_state, dtype=np.float32)
        self._filled     = np.zeros(shape_fill,  dtype=np.float32)

        self._idx = 0       # Next write position
        self._size = 0      # Current number of stored episodes

    # ── Writing ───────────────────────────────────────────────────────────────

    def store_episode(
        self,
        obs:        np.ndarray,   # (T, N, obs_size)
        actions:    np.ndarray,   # (T, N)
        rewards:    np.ndarray,   # (T, N)
        obs_next:   np.ndarray,   # (T, N, obs_size)
        terminated: np.ndarray,   # (T, N)
        state:      np.ndarray,   # (T, state_size)
        state_next: np.ndarray,   # (T, state_size)
    ) -> None:
        """
        Store one complete episode into the buffer.

        Episodes shorter than max_ep_len are zero-padded. The `filled`
        mask records which timesteps are real vs. padding.
        """
        T = obs.shape[0]
        assert T <= self.max_ep_len, (
            f"Episode length {T} exceeds max_ep_len {self.max_ep_len}"
        )

        slot = self._idx
        self._obs[slot, :T]        = obs
        self._actions[slot, :T]    = actions
        self._rewards[slot, :T]    = rewards
        self._obs_next[slot, :T]   = obs_next
        self._terminated[slot, :T] = terminated
        self._state[slot, :T]      = state
        self._state_next[slot, :T] = state_next
        self._filled[slot, :T]     = 1.0
        # Zero out any remaining padding slots (circular buffer may reuse)
        if T < self.max_ep_len:
            self._obs[slot, T:]        = 0
            self._filled[slot, T:]     = 0

        # Advance circular write pointer
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(self, batch_size: int, device: str = "cpu") -> EpisodeBatch:
        """
        Sample a random mini-batch of episodes.

        Parameters
        ----------
        batch_size : int
            Number of episodes per batch.
        device : str
            Torch device string: 'cpu' or 'cuda'.

        Returns
        -------
        EpisodeBatch
            Named tuple of tensors, all on the specified device.
        """
        assert self._size >= batch_size, (
            f"Buffer has only {self._size} episodes, requested {batch_size}."
        )
        idxs = np.random.choice(self._size, batch_size, replace=False)

        def t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr[idxs], dtype=torch.float32, device=device)

        return EpisodeBatch(
            obs        = t(self._obs),
            actions    = torch.tensor(self._actions[idxs], dtype=torch.long, device=device),
            rewards    = t(self._rewards),
            obs_next   = t(self._obs_next),
            terminated = t(self._terminated),
            state      = t(self._state),
            state_next = t(self._state_next),
            filled     = t(self._filled),
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._size

    def can_sample(self, batch_size: int) -> bool:
        """True if the buffer contains at least batch_size episodes."""
        return self._size >= batch_size

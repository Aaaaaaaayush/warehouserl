"""
q_network.py
------------
Individual agent Q-networks for QMIX.

WHY THIS EXISTS:
  Every agent has its own copy of this network that takes its local
  observation history as input and outputs a Q-value for each action.
  "Q-value" means: "If I'm in this situation and take action A,
  how much total future reward do I expect?"

  In QMIX, these individual Qᵢ values are then fed into the mixing
  network (qmix.py) which combines them into the joint team value Qtot.
  At execution time, each robot uses only its own QNetwork — no
  communication required.

KEY CONCEPTS:
  - GRU (Gated Recurrent Unit): a type of memory cell. Each agent
    maintains a hidden state across timesteps so it can reason about
    history, not just the current observation. This is crucial because
    our environment is partially observable — you can't know whether a
    corridor is clear from a single snapshot.
  - Shared weights: all agents share the same QNetwork parameters
    during training. This is not a limitation — it's a regulariser
    that forces each agent to develop a general strategy rather than
    one that overfits to its particular starting position.
  - [V2-READY]: The forward() signature includes comm_input parameter.
    V2 will pass communication vectors from comm_hooks.py here.

ARCHITECTURE:
  Input (obs_size,) → FC(64) → ReLU → GRU(64) → FC(act_size)
  Output: Q-values for each action, shape (act_size,)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Recurrent Q-network for a single MARL agent.

    All agents share one set of these weights during training
    (parameter sharing = implicit regularisation + reduced memory footprint).

    Parameters
    ----------
    obs_size : int
        Length of the flattened observation vector. For our env with
        radius=2: 25 (grid) + 6 (state) = 31.
    act_size : int
        Number of discrete actions. For our env: 7.
    hidden_size : int
        Hidden dimension of the GRU and FC layers. Default 64.
    """

    def __init__(self, obs_size: int, act_size: int, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size

        # Input embedding: raw obs → latent representation
        self.fc_input = nn.Linear(obs_size, hidden_size)

        # GRU: maintains agent's memory across timesteps
        # Input: hidden_size. Output: hidden_size, new_hidden_state.
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # Output head: hidden state → Q-value per action
        self.fc_out = nn.Linear(hidden_size, act_size)

        # Flatten GRU parameters for contiguous memory layout in VRAM
        self.gru.flatten_parameters()

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        comm_input: torch.Tensor | None = None,  # [V2-READY]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one agent at one timestep.

        Parameters
        ----------
        obs : torch.Tensor
            Shape (batch, obs_size). The agent's current observation.
        hidden : torch.Tensor
            Shape (1, batch, hidden_size). The GRU's previous hidden state.
            Pass zeros on episode start: QNetwork.init_hidden(batch).
        comm_input : torch.Tensor | None
            [V2-READY] Communication vector from other agents.
            V2 will concatenate this into the input embedding.
            Currently ignored (passthrough).

        Returns
        -------
        q_values : torch.Tensor
            Shape (batch, act_size). Q-value for each action.
        new_hidden : torch.Tensor
            Shape (1, batch, hidden_size). Updated GRU hidden state.
        """
        # Embed the observation
        x = F.relu(self.fc_input(obs))            # (batch, hidden_size)

        # GRU expects input of shape (batch, seq_len, input_size)
        x, new_hidden = self.gru(x.unsqueeze(1), hidden)  # x: (batch, 1, hidden)

        # Compute Q-values from GRU output
        q_values = self.fc_out(x.squeeze(1))      # (batch, act_size)

        return q_values, new_hidden

    @staticmethod
    def init_hidden(batch_size: int, hidden_size: int = 64) -> torch.Tensor:
        """
        Create a zero hidden state for the start of a new episode.

        Parameters
        ----------
        batch_size : int
            Number of parallel environments (usually 1 during eval,
            more during training with vectorised envs).
        hidden_size : int
            Must match the hidden_size used to construct the network.

        Returns
        -------
        torch.Tensor
            Shape (1, batch_size, hidden_size) of zeros.
        """
        return torch.zeros(1, batch_size, hidden_size)

"""
qmix.py
-------
The QMIX mixing network — the heart of the QMIX algorithm.

WHY THIS EXISTS:
  Each agent's QNetwork outputs Qᵢ: "how good is this action for ME?"
  The mixing network takes all Qᵢ values and combines them into Qtot:
  "how good is this joint action for THE TEAM?"

  The critical constraint: the mixing must be MONOTONE. This means:
  if any individual Qᵢ increases, Qtot can only increase or stay the same.
  It can never decrease.

  WHY MONOTONICITY MATTERS (plain English):
  Imagine you're in a warehouse team. If Robot A improves its decision
  (goes from "block the corridor" to "clear the corridor"), the team's
  overall performance can only get better — never worse. This is obvious
  from a common-sense standpoint.

  Mathematically, monotonicity means:
    argmax_a Qtot = (argmax_a1 Q1, argmax_a2 Q2, ..., argmax_an Qn)

  Translation: the best joint action = each agent independently picking
  its own best action. This is what makes decentralised execution work.
  No agent needs to know what the others chose — it just picks its own
  best action and, thanks to monotonicity, the team result is optimal.

ARCHITECTURE:
  Global state → Hypernetwork → Mixing weights (all non-negative via abs())
  Mixing: Qtot = w2 * ReLU(w1 * [Q1, Q2, ..., Qn] + b1) + b2

  The hypernetwork generates w1 and b1 conditioned on the global state.
  This means the "importance weight" of each agent's contribution
  adapts to the current situation — during a bottleneck, clearing it
  matters more; during open running, raw throughput matters more.

  References: Rashid et al., "QMIX: Monotonic Value Function
  Factorisation for Deep Multi-Agent Reinforcement Learning", ICML 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixingNetwork(nn.Module):
    """
    QMIX monotonic mixing network.

    Takes individual Q-values [Q1, ..., Qn] and the global state,
    and produces a single scalar Qtot that satisfies the monotonicity
    constraint required for decentralised execution.

    Parameters
    ----------
    n_agents : int
        Number of agents. Determines input dimension (one Qᵢ per agent).
    state_size : int
        Size of the global state vector fed to the hypernetwork.
        For our env: H * W * cell_types (grid flattened) + all agent states.
    embed_dim : int
        Dimension of the mixing network's hidden layer. Default 32.
    hypernet_embed : int
        Hidden dim of the hypernetwork that generates mixing weights. Default 64.
    """

    def __init__(
        self,
        n_agents: int,
        state_size: int,
        embed_dim: int = 32,
        hypernet_embed: int = 64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        # --- Hypernetwork for w1 ---
        # w1 has shape (n_agents, embed_dim). Hypernetwork maps state → w1.
        # We use abs() on the output to enforce non-negativity (monotonicity).
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_size, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, n_agents * embed_dim),
        )

        # --- Hypernetwork for b1 ---
        # Biases do NOT need to be non-negative (they shift, not scale).
        self.hyper_b1 = nn.Linear(state_size, embed_dim)

        # --- Hypernetwork for w2 ---
        # w2 has shape (embed_dim, 1). Non-negative via abs().
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_size, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim),
        )

        # --- Final bias ---
        # A small MLP that generates the final scalar bias b2.
        # This is the only part that can produce negative outputs,
        # and it allows Qtot to be shifted below zero when needed.
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        agent_qs: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Qtot from individual Q-values and global state.

        Parameters
        ----------
        agent_qs : torch.Tensor
            Shape (batch, n_agents). One Q-value per agent (for their
            chosen action — the max Q-value for that agent).
        global_state : torch.Tensor
            Shape (batch, state_size). The full global state of the
            environment at the current timestep (available only during
            training — not at execution time).

        Returns
        -------
        q_tot : torch.Tensor
            Shape (batch, 1). The joint team Q-value.
        """
        batch = agent_qs.size(0)

        # Reshape agent_qs for matrix multiplication: (batch, 1, n_agents)
        qs = agent_qs.unsqueeze(1)

        # --- Layer 1 ---
        # Hypernetwork generates w1 and b1 from global state
        w1 = torch.abs(self.hyper_w1(global_state))          # non-negative!
        w1 = w1.view(batch, self.n_agents, self.embed_dim)    # (batch, n, embed)
        b1 = self.hyper_b1(global_state).unsqueeze(1)         # (batch, 1, embed)

        # Mix: (batch, 1, n_agents) @ (batch, n_agents, embed) → (batch, 1, embed)
        hidden = F.elu(torch.bmm(qs, w1) + b1)

        # --- Layer 2 ---
        w2 = torch.abs(self.hyper_w2(global_state))           # non-negative!
        w2 = w2.view(batch, self.embed_dim, 1)                 # (batch, embed, 1)
        b2 = self.hyper_b2(global_state).view(batch, 1, 1)    # (batch, 1, 1)

        # Mix: (batch, 1, embed) @ (batch, embed, 1) → (batch, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.view(batch, 1)   # (batch, 1)

"""
comm_hooks.py
-------------
Communication hook stubs — empty passthroughs in V1.

WHY THIS EXISTS (even though it does nothing yet):
  In V1, agents do NOT communicate. Each agent acts purely on its own
  local observation. This is the standard QMIX setup.

  In V2, we will add explicit agent-to-agent communication channels —
  agents will broadcast short message vectors to neighbours before
  choosing their actions. This changes the observation input to the
  QNetwork from (obs,) to (obs + received_messages,).

  By adding these hooks NOW as identity functions (output = input,
  no modification), V2 can fill them with real communication logic
  without changing any other file. The QNetwork already has a
  comm_input parameter stub waiting for exactly this.

[V2-READY]: Replace passthrough functions below with:
  - Attention-based communication (ATOC, TarMAC)
  - Bandwidth-limited message passing
  - Emergent language protocols (DIAL, CommNet)
"""

from __future__ import annotations

import torch
from typing import Sequence


def pre_act_comm(
    agent_hidden_states: dict[str, torch.Tensor],
    agent_ids: Sequence[str],
) -> dict[str, torch.Tensor]:
    """
    Pre-action communication hook.

    In V1: returns an empty zero tensor for each agent (no messages).
    In V2: agents broadcast their hidden states and receive messages
           from neighbours, filtered by proximity and bandwidth limits.

    Parameters
    ----------
    agent_hidden_states : dict[agent_id -> Tensor]
        Each agent's current GRU hidden state, shape (1, 1, hidden_size).
    agent_ids : list[str]
        All active agent IDs.

    Returns
    -------
    dict[agent_id -> Tensor]
        Communication input for each agent. Shape (1, comm_dim).
        V1: all zeros, comm_dim=0 (no-op).
        V2: received message vectors.
    """
    # V1: return None for each agent — QNetwork.forward() ignores comm_input=None
    return {agent_id: None for agent_id in agent_ids}


def post_act_comm(
    agent_ids: Sequence[str],
    actions: dict[str, int],
) -> dict[str, object]:
    """
    Post-action communication hook.

    In V1: no-op.
    In V2: agents broadcast chosen actions or intent signals to allow
           other agents to anticipate movements.

    Parameters
    ----------
    agent_ids : list[str]
        All active agent IDs.
    actions : dict[agent_id -> int]
        The action chosen by each agent this step.

    Returns
    -------
    dict[agent_id -> object]
        V1: None for all agents.
        V2: received intent signals.
    """
    return {agent_id: None for agent_id in agent_ids}

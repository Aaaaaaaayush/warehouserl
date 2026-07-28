"""
evaluator.py
------------
Deterministic evaluation engine for trained WarehouseRL QMIX models.

WHY THIS EXISTS:
  During training, agents use epsilon-greedy exploration to discover strategies.
  To evaluate true performance, we set epsilon = 0.0 (pure greedy/deterministic).
  This module runs N evaluation episodes, tracks full agent trajectories, and
  returns clean evaluation metrics without exploration noise.

KEY CONCEPTS:
  - Trajectory collection: records [step, agent_id, row, col, battery, carrying, action]
    for every step of every evaluation episode. This raw trajectory data is passed
    directly to behavior_detector.py to detect emergent coordination strategies.
  - Epsilon = 0.0: guarantees agents execute their best learned policies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.environment.config_loader import load_config
from src.environment.warehouse_env import WarehouseEnv, Action
from src.agents.q_network import QNetwork


class Evaluator:
    """
    Runs deterministic evaluation episodes for a trained scenario checkpoint.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded scenario config.
    checkpoint_path : Path
        Path to trained model checkpoint (.pt).
    device : str
        Torch device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        cfg: Any,
        checkpoint_path: Path,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

        self.env = WarehouseEnv(cfg)
        sample_obs, _ = self.env.reset()
        self.obs_size = list(sample_obs.values())[0].shape[0]

        # Inspect checkpoint to determine saved obs_size
        ckpt_obs_size = self.obs_size
        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=device)
            ckpt_obs_size = ckpt.get("obs_size", self.obs_size)

        self.act_size = 7
        self.hidden_size = cfg.training.hidden_size

        # Load Q-network weights with matching obs_size
        self.q_net = QNetwork(
            obs_size=ckpt_obs_size,
            act_size=self.act_size,
            hidden_size=self.hidden_size,
        ).to(device)

        if self.checkpoint_path.exists():
            self.q_net.load_state_dict(ckpt["q_net_state"])
            self.q_net.eval()
        else:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self._ckpt_obs_size = ckpt_obs_size

    def evaluate(self, n_episodes: int = 100) -> tuple[dict[str, float], list[dict]]:
        """
        Run n_episodes deterministically (epsilon = 0.0).

        Returns
        -------
        metrics : dict[str, float]
            Aggregate performance metrics (mean_reward, deliveries, collision_rate, etc.).
        trajectories : list[dict]
            List of full episode trajectory dicts (used by BehaviorDetector and Recorder).
        """
        total_rewards = []
        total_deliveries = []
        total_collisions = 0
        total_steps = 0
        total_deadlocks = 0
        total_battery_events = []

        all_trajectories = []

        for ep_idx in range(n_episodes):
            obs_t, _ = self.env.reset(seed=ep_idx)
            hidden = {
                a: QNetwork.init_hidden(1, self.hidden_size).to(self.device)
                for a in self.env.possible_agents
            }

            ep_frames = []
            ep_reward = 0.0
            ep_collisions = 0
            ep_deadlocks = 0

            prev_pos = {
                a: (self.env._agent_states[a].row, self.env._agent_states[a].col)
                for a in self.env.possible_agents
            }

            t = 0
            max_steps = self.cfg.training.max_steps_per_episode

            while self.env.agents and t < max_steps:
                # Capture frame state for trajectory logging
                frame_agents = []
                for a_id in self.env.possible_agents:
                    if a_id in self.env._agent_states:
                        st = self.env._agent_states[a_id]
                        frame_agents.append({
                            "id": a_id,
                            "row": st.row,
                            "col": st.col,
                            "battery": float(st.battery),
                            "carrying": bool(st.carrying),
                            "frozen": bool(st.frozen),
                        })
                    else:
                        frame_agents.append({
                            "id": a_id,
                            "row": -1,
                            "col": -1,
                            "battery": 0.0,
                            "carrying": False,
                            "frozen": True,
                        })

                ep_frames.append({
                    "step": t,
                    "agents": frame_agents,
                    "items_on_shelves": [
                        [r, c] for (r, c), has_item in self.env._items_on_shelves.items() if has_item
                    ],
                })

                # Deterministic greedy action selection (epsilon = 0.0)
                actions = {}
                with torch.no_grad():
                    for a_id in self.env.possible_agents:
                        if a_id not in self.env.agents:
                            actions[a_id] = Action.STAY
                            continue

                        raw_ob = obs_t[a_id][:self._ckpt_obs_size]
                        ob = torch.tensor(
                            raw_ob, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        q_vals, new_h = self.q_net(ob, hidden[a_id])
                        hidden[a_id] = new_h
                        actions[a_id] = int(q_vals.argmax(dim=-1).item())

                obs_next, rewards, terms, truncs, infos = self.env.step(actions)
                ep_reward += sum(rewards.values())

                # Track collisions & deadlocks
                for a in self.env.possible_agents:
                    if infos.get(a, {}).get("collision", False):
                        ep_collisions += 1

                curr_pos = {
                    a: (self.env._agent_states[a].row, self.env._agent_states[a].col)
                    for a in self.env.possible_agents
                    if a in self.env._agent_states
                }
                if not any(curr_pos.get(a) != prev_pos.get(a) for a in curr_pos):
                    ep_deadlocks += 1
                prev_pos = curr_pos

                obs_t = obs_next
                t += 1

            stats = self.env.get_stats()
            total_rewards.append(ep_reward)
            total_deliveries.append(stats["deliveries"])
            total_collisions += ep_collisions
            total_steps += t
            total_deadlocks += ep_deadlocks
            total_battery_events.append(stats["frozen_agents"])

            all_trajectories.append({
                "episode": ep_idx,
                "steps": t,
                "frames": ep_frames,
                "deliveries": stats["deliveries"],
            })

        metrics = {
            "mean_reward": float(np.mean(total_rewards)),
            "mean_deliveries": float(np.mean(total_deliveries)),
            "collision_rate": float(total_collisions / max(total_steps, 1)),
            "deadlock_frequency": float(total_deadlocks / max(total_steps, 1)),
            "mean_battery_depletion": float(np.mean(total_battery_events)),
        }

        return metrics, all_trajectories

"""
trainer.py
----------
Full QMIX training loop for one scenario.

ALGORITHM OVERVIEW (QMIX):
  1. Run episode with epsilon-greedy actions using individual QNetworks
  2. Store full episode trajectory in ReplayBuffer
  3. Sample batch of episodes from buffer
  4. For each step t in each episode:
       a. Forward pass: QNetwork(obs_t) → Q-values per agent
       b. Select Q(s, a_taken) for the action actually executed
       c. Mix with QMixingNetwork(Q_values, global_state) → Q_tot
  5. Compute TD target using TARGET networks:
       y = reward + γ × Q_tot_target(next_obs, greedy_actions, next_state)
  6. Loss = MSE(Q_tot, y) masked by episode fill mask
  7. Backprop, clip gradients at norm 10.0, step optimiser
  8. Periodically sync target networks (hard copy every N episodes)
  9. Log to MLflow, save checkpoints

KEY DESIGN CHOICES:
  - Episode-based replay (not transition-based) to support GRU hidden states
  - Shared QNetwork weights across all agents (parameter sharing)
  - Global state fed to mixing network via env.get_global_state()
  - All hyperparameters from YAML — nothing hardcoded here
  - MLflow logs every eval_interval episodes
  - Checkpoints named: qmix_{n}agents_{k}episodes.pt [V2-READY]
"""

from __future__ import annotations

import time
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm

from src.environment.config_loader import load_config
from src.environment.warehouse_env import WarehouseEnv, Action
from src.agents.q_network import QNetwork
from src.agents.qmix import QMixingNetwork
from src.agents.replay_buffer import ReplayBuffer, EpisodeBatch
from src.agents.comm_hooks import pre_act_comm


# ── QMIXTrainer ───────────────────────────────────────────────────────────────

class QMIXTrainer:
    """
    Manages the full QMIX training lifecycle for one scenario.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded scenario config from load_config().
    device : str
        Torch device — 'cuda' for RTX 5080, 'cpu' for Oracle serving.
    run_name : str
        MLflow run name. Auto-generated from scenario if not provided.
    checkpoint_dir : Path
        Directory to save .pt checkpoints.
    """

    def __init__(
        self,
        cfg: Any,
        device: str = "cuda",
        run_name: str | None = None,
        checkpoint_dir: str | Path = "models",
    ):
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.run_name = run_name or f"qmix_scenario_{cfg.scenario.id}"

        # Shorthand references to nested config sections
        self.t_cfg = cfg.training    # training hyperparameters
        self.r_cfg = cfg.rewards     # reward weights

        # Components built by _build_components()
        self.env: WarehouseEnv | None = None
        self.q_net: QNetwork | None = None
        self.q_net_target: QNetwork | None = None
        self.mixer: QMixingNetwork | None = None
        self.mixer_target: QMixingNetwork | None = None
        self.optimiser: torch.optim.Optimizer | None = None
        self.buffer: ReplayBuffer | None = None

        # Training state
        self.epsilon: float = self.t_cfg.epsilon_start
        self.episode: int = 0
        self._metrics_window: list[dict] = []   # Rolling window for eval logging

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_components(self) -> None:
        """Instantiate all networks, buffer, and optimiser from config."""
        cfg, t = self.cfg, self.t_cfg

        # Environment
        self.env = WarehouseEnv(cfg)
        # Compute observation and state sizes from the live env
        sample_obs, _ = self.env.reset()
        obs_size   = list(sample_obs.values())[0].shape[0]   # 31 for radius=2
        state_size = self.env.state_size

        n_agents = cfg.agents.count
        act_size = 7   # Move×4 + Stay + PickUp + DropOff

        # Individual agent Q-network (shared weights across all agents)
        self.q_net = QNetwork(
            obs_size=obs_size,
            act_size=act_size,
            hidden_size=t.hidden_size,
        ).to(self.device)

        # Target Q-network (frozen copy, synced periodically)
        self.q_net_target = copy.deepcopy(self.q_net)
        self.q_net_target.eval()

        # QMIX mixing network
        self.mixer = QMixingNetwork(
            n_agents=n_agents,
            state_size=state_size,
            embed_dim=t.embed_dim,
            hypernet_embed=t.hypernet_embed,
        ).to(self.device)

        # Target mixing network
        self.mixer_target = copy.deepcopy(self.mixer)
        self.mixer_target.eval()

        # Single optimiser over both live networks' parameters
        self.optimiser = torch.optim.Adam(
            list(self.q_net.parameters()) + list(self.mixer.parameters()),
            lr=t.lr,
        )

        # Episode replay buffer
        self.buffer = ReplayBuffer(
            capacity=t.buffer_capacity,
            n_agents=n_agents,
            obs_size=obs_size,
            state_size=state_size,
            max_ep_len=cfg.training.max_steps_per_episode,
        )

        self._obs_size   = obs_size
        self._state_size = state_size
        self._act_size   = act_size
        self._n_agents   = n_agents

    # ── Checkpoint management ─────────────────────────────────────────────────

    def save_checkpoint(self, episode: int) -> Path:
        """
        Save a checkpoint with all network weights + training state.

        Filename: qmix_{n}agents_{k}episodes.pt
        [V2-READY]: This naming scheme lets the dynamic model loader
        enumerate available checkpoints for any scenario.
        """
        n = self.cfg.agents.count
        path = self.checkpoint_dir / f"qmix_{n}agents_{episode}episodes.pt"
        torch.save({
            "q_net_state":      self.q_net.state_dict(),
            "mixer_state":      self.mixer.state_dict(),
            "optimiser_state":  self.optimiser.state_dict(),
            "epsilon":          self.epsilon,
            "episode":          episode,
            "scenario_id":      self.cfg.scenario.id,
            "obs_size":         self._obs_size,
            "state_size":       self._state_size,
            "n_agents":         self._n_agents,
            "hidden_size":      self.cfg.training.hidden_size,
        }, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        """
        Load weights from a checkpoint.

        WEIGHT TRANSFER STRATEGY between scenarios:
          - QNetwork: full weight transfer (obs_size=31 is identical across all scenarios)
          - QMixingNetwork: RE-INITIALISED (state_size and n_agents differ per scenario)
          - Optimiser: reset (fresh start for new scenario)
          - Epsilon: overridden by scenario config (starts lower for transferred weights)

        This is why curriculum learning works: agents already know how to navigate
        and manage batteries — they only need to learn new coordination at scale.
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Always transfer Q-network weights (handles shape growth e.g. obs 31 -> 43)
        q_state = checkpoint["q_net_state"]
        curr_state = self.q_net.state_dict()

        if q_state["fc_input.weight"].shape != curr_state["fc_input.weight"].shape:
            ckpt_obs = q_state["fc_input.weight"].shape[1]
            print(f"  [Trainer] Partial weight transfer for fc_input: prior obs={ckpt_obs} -> current obs={self._obs_size}")
            curr_state["fc_input.weight"][:, :ckpt_obs] = q_state["fc_input.weight"]
            curr_state["fc_input.bias"] = q_state["fc_input.bias"]
            for k in list(q_state.keys()):
                if k not in ("fc_input.weight", "fc_input.bias"):
                    curr_state[k] = q_state[k]
            self.q_net.load_state_dict(curr_state)
        else:
            self.q_net.load_state_dict(q_state)

        self.q_net.gru.flatten_parameters()
        self.q_net_target = copy.deepcopy(self.q_net)
        self.q_net_target.gru.flatten_parameters()
        self.q_net_target.eval()

        # Mixing network: only transfer if n_agents and state_size match
        if (checkpoint["n_agents"] == self._n_agents and
                checkpoint["state_size"] == self._state_size):
            self.mixer.load_state_dict(checkpoint["mixer_state"])
            self.mixer_target = copy.deepcopy(self.mixer)
            self.mixer_target.eval()
        else:
            print(
                f"  [Trainer] Mixing network NOT transferred: "
                f"prior({checkpoint['n_agents']} agents, state={checkpoint['state_size']}) "
                f"-> current({self._n_agents} agents, state={self._state_size}). "
                f"QNetwork weights transferred only."
            )

        # Epsilon from config (lower for transferred weights — agents already know basics)
        self.epsilon = self.cfg.training.epsilon_start
        print(f"  [Trainer] Checkpoint loaded from {path}")
        print(f"  [Trainer] Epsilon reset to {self.epsilon:.3f} (from config)")

    # ── Episode execution ─────────────────────────────────────────────────────

    def _run_episode(self, epsilon: float) -> dict:
        """
        Execute one full episode with epsilon-greedy action selection.

        Returns episode statistics:
          total_reward, deliveries, collisions, steps,
          deadlock_steps, battery_events
        """
        cfg = self.cfg
        n = cfg.agents.count

        obs_t, _ = self.env.reset()
        state_t  = self.env.get_global_state()

        # GRU hidden states — one per agent, shape (1, 1, hidden_size)
        # [V2-READY]: comm_hooks will modify hidden states here in V2
        hidden = {
            a: QNetwork.init_hidden(1, cfg.training.hidden_size).to(self.device)
            for a in self.env.possible_agents
        }

        # Episode storage (pre-allocated numpy arrays)
        T = cfg.training.max_steps_per_episode
        ep_obs      = np.zeros((T, n, self._obs_size),   dtype=np.float32)
        ep_actions  = np.zeros((T, n),                   dtype=np.int64)
        ep_rewards  = np.zeros((T, n),                   dtype=np.float32)
        ep_obs_next = np.zeros((T, n, self._obs_size),   dtype=np.float32)
        ep_term     = np.zeros((T, n),                   dtype=np.float32)
        ep_state    = np.zeros((T, self._state_size),    dtype=np.float32)
        ep_state_nx = np.zeros((T, self._state_size),    dtype=np.float32)
        ep_filled   = np.zeros(T,                        dtype=np.float32)

        # Metrics
        total_reward    = 0.0
        total_collisions = 0
        deadlock_steps  = 0
        prev_positions  = {a: (self.env._agent_states[a].row,
                               self.env._agent_states[a].col)
                           for a in self.env.possible_agents}

        t = 0
        while self.env.agents and t < T:
            # ── Comm hooks (V1: no-op) ────────────────────────────────────
            _ = pre_act_comm(hidden, self.env.agents)

            # ── Action selection ──────────────────────────────────────────
            actions: dict[str, int] = {}
            with torch.no_grad():
                for agent_id in self.env.possible_agents:
                    if agent_id not in self.env.agents:
                        # Terminated agent — use STAY as placeholder
                        actions[agent_id] = Action.STAY
                        continue

                    ob = torch.tensor(
                        obs_t[agent_id], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)   # (1, obs_size)

                    q_vals, new_h = self.q_net(ob, hidden[agent_id])
                    hidden[agent_id] = new_h

                    if np.random.random() < epsilon:
                        actions[agent_id] = self.env.action_space(agent_id).sample()
                    else:
                        actions[agent_id] = int(q_vals.argmax(dim=-1).item())

            # ── Step environment ──────────────────────────────────────────
            obs_next_t, rewards_t, terms_t, truncs_t, infos_t = self.env.step(actions)
            state_next_t = self.env.get_global_state()

            # ── Record step ───────────────────────────────────────────────
            for i, agent_id in enumerate(self.env.possible_agents):
                if agent_id in obs_t:
                    ep_obs[t, i]      = obs_t[agent_id]
                if agent_id in obs_next_t:
                    ep_obs_next[t, i] = obs_next_t[agent_id]
                elif agent_id in obs_t:
                    # Terminated this step — use last obs as next
                    ep_obs_next[t, i] = obs_t[agent_id]

                ep_actions[t, i] = actions.get(agent_id, Action.STAY)
                ep_rewards[t, i] = rewards_t.get(agent_id, 0.0)
                ep_term[t, i]    = float(terms_t.get(agent_id, False))

            ep_state[t]    = state_t
            ep_state_nx[t] = state_next_t
            ep_filled[t]   = 1.0

            # ── Metrics ───────────────────────────────────────────────────
            step_reward = sum(rewards_t.values())
            total_reward += step_reward

            for a in self.env.possible_agents:
                if infos_t.get(a, {}).get("collision", False):
                    total_collisions += 1

            # Deadlock detection: no agent moved this step
            curr_positions = {
                a: (self.env._agent_states[a].row, self.env._agent_states[a].col)
                for a in self.env.possible_agents
                if a in self.env._agent_states
            }
            moved = any(
                curr_positions.get(a) != prev_positions.get(a)
                for a in curr_positions
            )
            if not moved:
                deadlock_steps += 1
            prev_positions = curr_positions

            obs_t   = obs_next_t
            state_t = state_next_t
            t += 1

        # Store episode in buffer
        actual_t = t
        self.buffer.store_episode(
            obs        = ep_obs[:actual_t],
            actions    = ep_actions[:actual_t],
            rewards    = ep_rewards[:actual_t],
            obs_next   = ep_obs_next[:actual_t],
            terminated = ep_term[:actual_t],
            state      = ep_state[:actual_t],
            state_next = ep_state_nx[:actual_t],
        )

        env_stats = self.env.get_stats()

        return {
            "total_reward":    total_reward,
            "deliveries":      env_stats["deliveries"],
            "collisions":      total_collisions,
            "steps":           actual_t,
            "deadlock_steps":  deadlock_steps,
            "battery_events":  env_stats["frozen_agents"],
        }

    # ── Training update ────────────────────────────────────────────────────────

    def _update(self) -> float:
        """
        Sample one batch from the buffer and perform one QMIX gradient update.

        Returns the scalar loss value for logging.

        QMIX LOSS DERIVATION:
          For each step t in each episode:
            - Get Q(o_i_t, a_i_t) for each agent i (Q of chosen action)
            - Feed into mixer with state_t → Q_tot_t
            - Compute target: y_t = r_t + γ × Q_tot_target(o_i_t+1, greedy_a_i, s_t+1)
          Loss = mean((Q_tot_t - y_t)² × filled_mask) / sum(filled_mask)
          The filled_mask zeros out padded timesteps so they don't affect learning.
        """
        t_cfg = self.t_cfg
        batch: EpisodeBatch = self.buffer.sample(t_cfg.batch_size, device=self.device)

        B, T, N, obs_size = batch.obs.shape

        # ── Forward pass: Q-values for all (t, agent) in batch ───────────────

        # Reshape to (B*N, T, obs_size) for batched GRU
        # We run GRU across T timesteps for all episodes and all agents at once
        obs_flat      = batch.obs.permute(0, 2, 1, 3).reshape(B * N, T, obs_size)
        obs_next_flat = batch.obs_next.permute(0, 2, 1, 3).reshape(B * N, T, obs_size)

        h0 = QNetwork.init_hidden(B * N, t_cfg.hidden_size).to(self.device)

        # LIVE network — Q-values for all steps (used to get Q(s,a))
        # Forward through GRU: input (B*N, T, obs_size) → output (B*N, T, act_size)
        x = torch.relu(self.q_net.fc_input(obs_flat))        # (B*N, T, H)
        gru_out, _ = self.q_net.gru(x, h0)                   # (B*N, T, H)
        q_all = self.q_net.fc_out(gru_out)                    # (B*N, T, act_size)
        q_all = q_all.reshape(B, N, T, self._act_size).permute(0, 2, 1, 3)
        # Shape: (B, T, N, act_size)

        # TARGET network — Q-values for next obs (used for greedy target)
        h0_tgt = QNetwork.init_hidden(B * N, t_cfg.hidden_size).to(self.device)
        with torch.no_grad():
            x_next = torch.relu(self.q_net_target.fc_input(obs_next_flat))
            gru_out_next, _ = self.q_net_target.gru(x_next, h0_tgt)
            q_next_all = self.q_net_target.fc_out(gru_out_next)
        q_next_all = q_next_all.reshape(B, N, T, self._act_size).permute(0, 2, 1, 3)
        # Shape: (B, T, N, act_size)

        # ── Select Q(s, a_taken) for each agent ──────────────────────────────

        # batch.actions: (B, T, N) — action indices
        actions_idx = batch.actions.unsqueeze(-1)               # (B, T, N, 1)
        q_taken = q_all.gather(-1, actions_idx).squeeze(-1)     # (B, T, N)

        # Greedy max Q for next state (used in target computation)
        q_next_max = q_next_all.max(dim=-1)[0]                  # (B, T, N)

        # ── QMIX mixing ───────────────────────────────────────────────────────

        # Flatten B×T for the mixing network
        state      = batch.state.reshape(B * T, self._state_size)       # (B*T, S)
        state_next = batch.state_next.reshape(B * T, self._state_size)  # (B*T, S)
        q_taken_bt = q_taken.reshape(B * T, N)                          # (B*T, N)
        q_next_bt  = q_next_max.reshape(B * T, N)                       # (B*T, N)

        q_tot = self.mixer(q_taken_bt, state).reshape(B, T)             # (B, T)

        with torch.no_grad():
            q_tot_target = self.mixer_target(q_next_bt, state_next).reshape(B, T)

        # ── TD target ─────────────────────────────────────────────────────────

        # Team reward = mean reward across all agents per step
        rewards = batch.rewards.mean(dim=-1)                    # (B, T)

        # Episode is done if ANY agent terminated (or all truncated)
        terminated = batch.terminated.max(dim=-1)[0]            # (B, T)

        y = rewards + t_cfg.gamma * q_tot_target * (1.0 - terminated)

        # ── Masked MSE loss ───────────────────────────────────────────────────

        filled = batch.filled                                   # (B, T) — 0 for padding
        td_error  = (q_tot - y.detach()) ** 2                   # (B, T)
        loss = (td_error * filled).sum() / (filled.sum() + 1e-8)

        # ── Backprop ──────────────────────────────────────────────────────────

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q_net.parameters()) + list(self.mixer.parameters()),
            max_norm=t_cfg.gradient_clip_norm,
        )
        self.optimiser.step()

        return float(loss.item())

    # ── Target sync ───────────────────────────────────────────────────────────

    def _sync_target_networks(self) -> None:
        """Hard copy live network weights into target networks."""
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.mixer_target.load_state_dict(self.mixer.state_dict())

    # ── Epsilon decay ─────────────────────────────────────────────────────────

    def _decay_epsilon(self) -> None:
        """
        Linear epsilon decay over the first epsilon_anneal_fraction of training.
        After that fraction, epsilon stays at epsilon_min.
        """
        t = self.t_cfg
        anneal_episodes = int(t.total_episodes * t.epsilon_anneal_fraction)
        if self.episode < anneal_episodes:
            self.epsilon = t.epsilon_start - (
                (t.epsilon_start - t.epsilon_min) * self.episode / anneal_episodes
            )
        else:
            self.epsilon = t.epsilon_min

    # ── Main training loop ────────────────────────────────────────────────────

    def train(
        self,
        transfer_checkpoint: Path | None = None,
        mlflow_experiment: str = "WarehouseRL",
    ) -> dict:
        """
        Run the complete training loop for this scenario.

        Parameters
        ----------
        transfer_checkpoint : Path | None
            If provided, loads QNetwork weights from this checkpoint before
            training starts (curriculum transfer from prior scenario).
        mlflow_experiment : str
            MLflow experiment name to log metrics under.

        Returns
        -------
        dict
            Final aggregate metrics:
            {mean_reward, packages_delivered, collision_rate,
             deadlock_frequency, battery_events, total_episodes}
        """
        self._build_components()

        # Curriculum weight transfer
        if transfer_checkpoint is not None and transfer_checkpoint.exists():
            print(f"\n[Trainer] Loading transfer weights from {transfer_checkpoint}")
            self.load_checkpoint(transfer_checkpoint)
        else:
            print(f"\n[Trainer] Starting from random initialisation")

        t_cfg = self.t_cfg
        total_eps = t_cfg.total_episodes

        # MLflow tracking
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name=self.run_name):

            # Log all hyperparameters
            mlflow.log_params({
                "scenario_id":   self.cfg.scenario.id,
                "n_agents":      self.cfg.agents.count,
                "grid":          f"{self.cfg.grid.width}x{self.cfg.grid.height}",
                "total_episodes": total_eps,
                "lr":            t_cfg.lr,
                "gamma":         t_cfg.gamma,
                "batch_size":    t_cfg.batch_size,
                "hidden_size":   t_cfg.hidden_size,
                "embed_dim":     t_cfg.embed_dim,
                "epsilon_start": t_cfg.epsilon_start,
                "epsilon_min":   t_cfg.epsilon_min,
            })

            # Rolling window accumulators
            window_reward      = []
            window_deliveries  = []
            window_collisions  = []
            window_steps       = []
            window_deadlocks   = []
            window_battery_ev  = []
            window_losses      = []

            t_start = time.time()
            last_checkpoint_ep = 0

            pbar = tqdm(
                total=total_eps,
                desc=f"S{self.cfg.scenario.id} {self.cfg.scenario.name}",
                unit="ep",
                dynamic_ncols=True,
            )

            for ep in range(1, total_eps + 1):
                self.episode = ep

                # Run one episode
                ep_stats = self._run_episode(self.epsilon)

                # Epsilon decay
                self._decay_epsilon()

                # Accumulate rolling window
                window_reward.append(ep_stats["total_reward"])
                window_deliveries.append(ep_stats["deliveries"])
                window_collisions.append(ep_stats["collisions"])
                window_steps.append(ep_stats["steps"])
                window_deadlocks.append(ep_stats["deadlock_steps"])
                window_battery_ev.append(ep_stats["battery_events"])

                # Training update
                if (self.buffer.can_sample(t_cfg.batch_size) and
                        ep % t_cfg.train_every_n_episodes == 0):
                    loss = self._update()
                    window_losses.append(loss)

                # Target network sync
                if ep % t_cfg.target_update_interval == 0:
                    self._sync_target_networks()

                # ── MLflow logging every eval_interval ───────────────────
                if ep % t_cfg.eval_interval == 0:
                    elapsed    = time.time() - t_start
                    fps        = ep / elapsed if elapsed > 0 else 0.0
                    steps_total = sum(window_steps[-t_cfg.eval_interval:])

                    mean_reward     = float(np.mean(window_reward[-t_cfg.eval_interval:]))
                    mean_deliveries = float(np.mean(window_deliveries[-t_cfg.eval_interval:]))
                    mean_collisions = (
                        sum(window_collisions[-t_cfg.eval_interval:]) /
                        max(steps_total, 1)
                    )
                    mean_deadlock   = (
                        sum(window_deadlocks[-t_cfg.eval_interval:]) /
                        max(steps_total, 1)
                    )
                    mean_battery_ev = float(np.mean(window_battery_ev[-t_cfg.eval_interval:]))
                    mean_loss       = float(np.mean(window_losses[-100:])) if window_losses else 0.0

                    mlflow.log_metrics({
                        "mean_reward":          mean_reward,
                        "packages_delivered":   mean_deliveries,
                        "collision_rate":        mean_collisions,
                        "deadlock_frequency":    mean_deadlock,
                        "battery_depletion_ev":  mean_battery_ev,
                        "training_fps":          fps,
                        "epsilon":               self.epsilon,
                        "loss":                  mean_loss,
                    }, step=ep)

                    # Also write to a JSON file for the FastAPI /stats endpoint
                    self._append_metrics_json(ep, {
                        "episode":              ep,
                        "mean_reward":          mean_reward,
                        "packages_delivered":   mean_deliveries,
                        "collision_rate":        mean_collisions,
                        "deadlock_frequency":    mean_deadlock,
                        "battery_depletion_ev":  mean_battery_ev,
                        "training_fps":          fps,
                        "epsilon":               self.epsilon,
                        "loss":                  mean_loss,
                    })

                    pbar.set_postfix({
                        "R":    f"{mean_reward:.1f}",
                        "pkg":  f"{mean_deliveries:.1f}",
                        "col":  f"{mean_collisions:.3f}",
                        "ε":    f"{self.epsilon:.3f}",
                        "fps":  f"{fps:.0f}",
                    })

                # ── Checkpoint ────────────────────────────────────────────
                if (ep % t_cfg.checkpoint_interval == 0 and
                        ep != last_checkpoint_ep):
                    ckpt_path = self.save_checkpoint(ep)
                    print(f"\n  [OK] Checkpoint saved: {ckpt_path}")
                    last_checkpoint_ep = ep

                pbar.update(1)

            pbar.close()

            # Final checkpoint
            final_ckpt = self.save_checkpoint(total_eps)
            print(f"\n[Trainer] Final checkpoint: {final_ckpt}")

            # Final metrics
            final_metrics = {
                "mean_reward":        float(np.mean(window_reward[-1000:])),
                "packages_delivered": float(np.mean(window_deliveries[-1000:])),
                "collision_rate":     (sum(window_collisions[-1000:]) /
                                       max(sum(window_steps[-1000:]), 1)),
                "deadlock_frequency": (sum(window_deadlocks[-1000:]) /
                                       max(sum(window_steps[-1000:]), 1)),
                "battery_events":     float(np.mean(window_battery_ev[-1000:])),
                "total_episodes":     total_eps,
            }
            mlflow.log_metrics({k: v for k, v in final_metrics.items()
                                 if isinstance(v, float)},
                                step=total_eps)

            return final_metrics

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _append_metrics_json(self, episode: int, metrics: dict) -> None:
        """
        Append a metrics snapshot to logs/scenario_N_metrics.json.
        This JSON file is read directly by GET /api/stats/{scenario_id}.
        Format: list of {episode, metric_name: value, ...} objects.
        """
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        path = logs_dir / f"scenario_{self.cfg.scenario.id}_metrics.json"

        existing = []
        if path.exists():
            with path.open() as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []

        existing.append(metrics)

        with path.open("w") as f:
            json.dump(existing, f)

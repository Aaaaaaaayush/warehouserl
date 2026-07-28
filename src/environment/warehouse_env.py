"""
warehouse_env.py
----------------
Custom PettingZoo Parallel environment for the WarehouseRL simulation.

WHY PETTINGZOO — PARALLEL API (not AEC):
  PettingZoo offers two APIs:
  - AEC (Agent-Environment Cycle): agents act one at a time, like turns
    in a board game. Simple to implement but slow — you can't vectorise
    agent steps.
  - Parallel: all agents observe and act simultaneously each step, like
    real robots operating in real time. Faster, maps naturally to QMIX's
    joint-action assumption, and required for centralised training where
    we need all agents' actions at once to compute Qtot.
  We use Parallel for speed and correctness.

KEY CONCEPTS IMPLEMENTED HERE:
  - Grid world with typed cells (see CellType enum)
  - Partial observability: each agent sees only a (2r+1)×(2r+1) window
  - Battery as a hard resource constraint (empty battery = agent freeze)
  - Item spawning with configurable rate
  - Reward shaping (all weights from YAML config, never hardcoded)
  - Scenario 3 random obstacle generation on reset()

OBSERVATION SPACE (per agent):
  Flattened 5×5 local grid (25 ints, one cell type per cell)
  + own state vector: [x, y, battery/capacity, carrying (0/1), target_x, target_y]
  Total: 25 + 6 = 31 floats per agent

ACTION SPACE (per agent, Discrete(7)):
  0=Up, 1=Down, 2=Left, 3=Right, 4=Stay, 5=PickUp, 6=DropOff
"""

from __future__ import annotations

import random
import numpy as np
from enum import IntEnum
from typing import Any

import gymnasium as gym
from pettingzoo import ParallelEnv


# ── Cell types ────────────────────────────────────────────────────────────────

class CellType(IntEnum):
    EMPTY = 0
    SHELF = 1
    DISPATCH = 2
    CHARGER = 3
    WALL = 4
    AGENT = 5
    ITEM = 6          # Shelf that currently has an item on it


# ── Action constants ──────────────────────────────────────────────────────────

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    PICK_UP = 5
    DROP_OFF = 6


_ACTION_DELTAS = {
    Action.UP:    (-1,  0),
    Action.DOWN:  ( 1,  0),
    Action.LEFT:  ( 0, -1),
    Action.RIGHT: ( 0,  1),
    Action.STAY:  ( 0,  0),
}


# ── Agent state dataclass ─────────────────────────────────────────────────────

class AgentState:
    """Holds all per-agent mutable state for one environment step."""

    __slots__ = ("row", "col", "battery", "carrying", "target_row", "target_col", "frozen")

    def __init__(self, row: int, col: int, battery_capacity: int):
        self.row = row
        self.col = col
        self.battery = battery_capacity
        self.carrying = False
        self.target_row = -1   # -1 = no current target
        self.target_col = -1
        self.frozen = False    # True when battery hits 0


# ── Main environment class ────────────────────────────────────────────────────

class WarehouseEnv(ParallelEnv):
    """
    WarehouseRL custom PettingZoo Parallel environment.

    All grid/agent/reward parameters come from a loaded config namespace
    (see config_loader.py). Nothing is hardcoded.

    Usage
    -----
    >>> from src.environment.config_loader import load_config
    >>> from src.environment.warehouse_env import WarehouseEnv
    >>> cfg = load_config(scenario_id=1)
    >>> env = WarehouseEnv(cfg)
    >>> observations, infos = env.reset()
    >>> # observations is a dict: agent_id -> np.ndarray of shape (31,)
    """

    metadata = {"render_modes": ["rgb_array"], "name": "warehouse_v1"}

    def __init__(self, cfg: Any, render_mode: str | None = None):
        """
        Parameters
        ----------
        cfg : SimpleNamespace
            Loaded scenario config (output of load_config()).
        render_mode : str | None
            'rgb_array' to enable renderer.py to capture frames.
        """
        self.cfg = cfg
        self.render_mode = render_mode

        # Agent IDs: "agent_0", "agent_1", ..., "agent_N-1"
        self.possible_agents = [
            f"agent_{i}" for i in range(cfg.agents.count)
        ]
        # After reset(), active agents = possible_agents (none start frozen)
        self.agents: list[str] = []

        # Observation space: 25 grid cells + 6 state values + 12 one-hot agent ID slots = 43 floats
        obs_size = (2 * cfg.agents.observation_radius + 1) ** 2 + 6 + 12
        self._obs_space = gym.spaces.Box(
            low=0.0, high=float(max(CellType)),
            shape=(obs_size,), dtype=np.float32
        )

        # Action space: Discrete(7) — see Action enum above
        self._act_space = gym.spaces.Discrete(len(Action))

        # Internal grid and state (populated on reset())
        self._grid: np.ndarray | None = None          # shape (H, W)
        self._items_on_shelves: dict[tuple, bool] = {}
        self._agent_states: dict[str, AgentState] = {}
        self._step_count = 0

        # Throughput streak tracker for Scenario 3 bonus
        self._delivery_history: list[int] = []   # Step timestamps of deliveries

    # ── PettingZoo required properties ───────────────────────────────────────

    def observation_space(self, agent: str) -> gym.spaces.Space:
        return self._obs_space

    def action_space(self, agent: str) -> gym.spaces.Space:
        return self._act_space

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """
        Reset the environment for a new episode.

        Returns
        -------
        observations : dict[agent_id -> np.ndarray]
        infos        : dict[agent_id -> dict]
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._delivery_history = []

        # Build clean grid
        H, W = self.cfg.grid.height, self.cfg.grid.width
        self._grid = np.zeros((H, W), dtype=np.int32)

        # Place walls (border)
        self._grid[0, :] = CellType.WALL
        self._grid[-1, :] = CellType.WALL
        self._grid[:, 0] = CellType.WALL
        self._grid[:, -1] = CellType.WALL

        # Place fixed cells from config
        for r, c in self.cfg.grid.shelves:
            self._grid[r][c] = CellType.SHELF
        for r, c in self.cfg.grid.dispatch_points:
            self._grid[r][c] = CellType.DISPATCH
        for r, c in self.cfg.grid.charging_stations:
            self._grid[r][c] = CellType.CHARGER
        for r, c in self.cfg.grid.obstacles:
            self._grid[r][c] = CellType.WALL

        # Scenario 3: randomised extra obstacles
        if getattr(self.cfg, "randomisation", None) and self.cfg.randomisation.enabled:
            self._place_random_obstacles()

        # Initialise item presence on shelves
        self._items_on_shelves = {
            (r, c): False for r, c in self.cfg.grid.shelves
        }

        # Initialise agents
        self._agent_states = {}
        starts = self.cfg.grid.agent_start_positions
        for i, agent_id in enumerate(self.agents):
            r, c = starts[i]
            self._agent_states[agent_id] = AgentState(
                row=r, col=c,
                battery_capacity=self.cfg.agents.battery_capacity
            )

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """
        Advance the environment by one timestep.

        Parameters
        ----------
        actions : dict[agent_id -> int]
            One action per active agent (see Action enum).

        Returns
        -------
        observations, rewards, terminations, truncations, infos
        All dicts keyed by agent_id.
        """
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {"delivered": 0, "collision": False} for a in self.agents}

        # --- Phase 1: Resolve movement ---
        self._resolve_movement(actions, rewards, infos)

        # --- Phase 2: Resolve pick-up / drop-off ---
        self._resolve_item_actions(actions, rewards, infos)

        # --- Phase 3: Battery update ---
        self._update_batteries(actions, rewards)

        # --- Phase 4: Item spawning ---
        self._spawn_items()

        # --- Phase 5: Step penalty (encourages efficiency) ---
        r_cfg = self.cfg.rewards
        for a in self.agents:
            rewards[a] += r_cfg.step_penalty

        # --- Phase 6: Check termination conditions ---
        self._step_count += 1
        max_steps = self.cfg.training.max_steps_per_episode
        if self._step_count >= max_steps:
            truncations = {a: True for a in self.agents}

        # Remove agents with empty batteries (frozen)
        for a in list(self.agents):
            if self._agent_states[a].frozen:
                terminations[a] = True
                self.agents.remove(a)

        observations = {a: self._get_obs(a) for a in self.agents}
        return observations, rewards, terminations, truncations, infos

    def render(self) -> np.ndarray | None:
        """
        Return an RGB array of the current grid state.
        Used by renderer.py to capture frames for video recording.
        Returns None if render_mode is not 'rgb_array'.
        """
        if self.render_mode != "rgb_array":
            return None
        # Rendering logic implemented in renderer.py and called externally.
        # This stub satisfies the PettingZoo interface.
        raise NotImplementedError(
            "Call renderer.py render_frame(env) instead of env.render()."
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self, agent_id: str) -> np.ndarray:
        """Build the observation vector for one agent."""
        state = self._agent_states[agent_id]
        r = self.cfg.agents.observation_radius
        H, W = self.cfg.grid.height, self.cfg.grid.width

        # Extract (2r+1)×(2r+1) local grid window, padded with WALL
        window_size = 2 * r + 1
        local_grid = np.full((window_size, window_size), CellType.WALL, dtype=np.float32)

        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                nr, nc = state.row + dr, state.col + dc
                if 0 <= nr < H and 0 <= nc < W:
                    cell = self._grid[nr][nc]
                    # Overlay: if another agent is there, show AGENT type
                    for other_id, other_state in self._agent_states.items():
                        if other_id != agent_id and other_state.row == nr and other_state.col == nc:
                            cell = CellType.AGENT
                            break
                    local_grid[dr + r][dc + r] = float(cell)

        grid_flat = local_grid.flatten()  # Shape: (25,) for radius=2

        # Own state vector (normalised)
        state_vec = np.array([
            state.row / H,
            state.col / W,
            state.battery / self.cfg.agents.battery_capacity,
            float(state.carrying),
            state.target_row / H if state.target_row >= 0 else -1.0,
            state.target_col / W if state.target_col >= 0 else -1.0,
        ], dtype=np.float32)

        # One-hot Agent ID embedding (fixed 12 slots for all scenarios)
        agent_idx = int(agent_id.split("_")[1])
        one_hot_id = np.zeros(12, dtype=np.float32)
        if agent_idx < 12:
            one_hot_id[agent_idx] = 1.0

        return np.concatenate([grid_flat, state_vec, one_hot_id])

    def _resolve_movement(
        self,
        actions: dict[str, int],
        rewards: dict[str, float],
        infos: dict[str, dict],
    ) -> None:
        """
        Move agents, detect collisions.
        Collision rule: if two agents try to swap positions or occupy
        the same cell, neither moves and both receive collision_penalty.
        """
        r_cfg = self.cfg.rewards
        H, W = self.cfg.grid.height, self.cfg.grid.width

        # Compute intended next positions
        intended: dict[str, tuple[int, int]] = {}
        for agent_id in self.agents:
            if self._agent_states[agent_id].frozen:
                continue
            action = Action(actions.get(agent_id, Action.STAY))
            state = self._agent_states[agent_id]
            if action in _ACTION_DELTAS:
                dr, dc = _ACTION_DELTAS[action]
                nr, nc = state.row + dr, state.col + dc
                # Boundary + wall check
                if 0 <= nr < H and 0 <= nc < W and self._grid[nr][nc] != CellType.WALL:
                    intended[agent_id] = (nr, nc)
                else:
                    intended[agent_id] = (state.row, state.col)
                    if action != Action.STAY:
                        rewards[agent_id] += r_cfg.collision_penalty
                        infos[agent_id]["collision"] = True
            else:
                intended[agent_id] = (state.row, state.col)

        # Detect agent-agent collisions (same target cell)
        from collections import Counter
        target_counts = Counter(intended.values())
        colliding_cells = {cell for cell, count in target_counts.items() if count > 1}

        for agent_id, (nr, nc) in intended.items():
            if (nr, nc) in colliding_cells:
                rewards[agent_id] += r_cfg.collision_penalty
                infos[agent_id]["collision"] = True
            else:
                self._agent_states[agent_id].row = nr
                self._agent_states[agent_id].col = nc

    def _resolve_item_actions(
        self,
        actions: dict[str, int],
        rewards: dict[str, float],
        infos: dict[str, dict],
    ) -> None:
        """Handle PICK_UP and DROP_OFF actions."""
        r_cfg = self.cfg.rewards
        for agent_id in self.agents:
            action = Action(actions.get(agent_id, Action.STAY))
            state = self._agent_states[agent_id]
            pos = (state.row, state.col)

            if action == Action.PICK_UP:
                if not state.carrying and self._items_on_shelves.get(pos, False):
                    state.carrying = True
                    self._items_on_shelves[pos] = False
                    self._grid[pos[0]][pos[1]] = CellType.SHELF
                    rewards[agent_id] += r_cfg.pick_up_reward
                    state.target_row, state.target_col = self._nearest_dispatch(pos)

            elif action == Action.DROP_OFF:
                if state.carrying and self._grid[state.row][state.col] == CellType.DISPATCH:
                    state.carrying = False
                    state.target_row, state.target_col = -1, -1
                    rewards[agent_id] += r_cfg.delivery_reward
                    infos[agent_id]["delivered"] = 1
                    self._delivery_history.append(self._step_count)
                    # Throughput streak bonus (Scenario 3)
                    if hasattr(r_cfg, "throughput_streak_bonus"):
                        rewards[agent_id] += self._streak_bonus(r_cfg)

                    # Dispatch balance bonus (Scenarios 2 & 3)
                    if hasattr(r_cfg, "dispatch_balance_bonus"):
                        rewards[agent_id] += r_cfg.dispatch_balance_bonus * 0.5

    def _update_batteries(
        self,
        actions: dict[str, int],
        rewards: dict[str, float],
    ) -> None:
        """Deplete or recharge batteries, freeze agents at 0."""
        r_cfg = self.cfg.rewards
        a_cfg = self.cfg.agents
        for agent_id in self.agents:
            state = self._agent_states[agent_id]
            cell = self._grid[state.row][state.col]

            if cell == CellType.CHARGER:
                state.battery = min(
                    a_cfg.battery_capacity,
                    state.battery + a_cfg.battery_charge_per_step
                )
                rewards[agent_id] += r_cfg.charge_reward
            else:
                state.battery -= a_cfg.battery_depletion_per_step
                if state.battery <= 0:
                    state.battery = 0
                    state.frozen = True
                    rewards[agent_id] += r_cfg.battery_empty_penalty

    def _spawn_items(self) -> None:
        """
        Randomly place items on empty shelves at configured spawn rate.
        max_items_on_shelf is a per-shelf cap — enforced by tracking how many
        items are currently on each shelf position separately.
        (In this grid model each shelf holds at most 1 item at a time;
        the cap applies across the total active item count per cluster.)
        """
        spawn_rate = self.cfg.items.spawn_rate
        total_items = sum(self._items_on_shelves.values())
        max_total = self.cfg.items.max_items_on_shelf * len(self._items_on_shelves)
        if total_items >= max_total:
            return
        for pos in self._items_on_shelves:
            if not self._items_on_shelves[pos] and random.random() < spawn_rate:
                self._items_on_shelves[pos] = True
                self._grid[pos[0]][pos[1]] = CellType.ITEM

    def _nearest_dispatch(self, from_pos: tuple[int, int]) -> tuple[int, int]:
        """Return the (row, col) of the nearest dispatch point by Manhattan distance."""
        best, best_dist = None, float("inf")
        for r, c in self.cfg.grid.dispatch_points:
            d = abs(from_pos[0] - r) + abs(from_pos[1] - c)
            if d < best_dist:
                best_dist = d
                best = (r, c)
        return best

    def _streak_bonus(self, r_cfg) -> float:
        """
        Compute throughput streak bonus for Scenario 3.
        Returns the bonus if >= threshold deliveries happened in the
        last `window` steps, else 0.0.
        """
        window = r_cfg.throughput_streak_window
        threshold = r_cfg.throughput_streak_threshold
        recent = [s for s in self._delivery_history if self._step_count - s <= window]
        if len(recent) >= threshold:
            return r_cfg.throughput_streak_bonus
        return 0.0

    def _place_random_obstacles(self) -> None:
        """
        Scenario 3: place N random extra obstacles on cells that are
        not shelves, dispatch points, chargers, or agent starts.
        [V2-READY]: Seed parameter allows reproducible random layouts.
        """
        H, W = self.cfg.grid.height, self.cfg.grid.width
        exclusion = set()
        # Add 1-cell neighborhood around critical functional points to prevent bottleneck blockage
        def add_with_neighbors(r: int, c: int):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        exclusion.add((nr, nc))

        for r, c in self.cfg.grid.shelves:
            exclusion.add((r, c))
        for r, c in self.cfg.grid.dispatch_points:
            add_with_neighbors(r, c)
        for r, c in self.cfg.grid.charging_stations:
            add_with_neighbors(r, c)
        for r, c in self.cfg.grid.agent_start_positions:
            add_with_neighbors(r, c)

        candidates = [
            (r, c)
            for r in range(1, H - 1)
            for c in range(1, W - 1)
            if (r, c) not in exclusion and self._grid[r][c] == CellType.EMPTY
        ]

        n = self.cfg.randomisation.extra_obstacle_count
        chosen = random.sample(candidates, min(n, len(candidates)))
        for r, c in chosen:
            self._grid[r][c] = CellType.WALL

    # ── Global state (used by QMIX mixing network during training only) ───────

    @property
    def state_size(self) -> int:
        """
        Size of the global state vector returned by get_global_state().

        Composition:
          - Flattened grid:   H × W  integers (one CellType per cell)
          - Per-agent state:  N × 6  floats   (row, col, battery%, carrying,
                                               target_row, target_col)
        Total: H*W + N*6

        This property lets trainer.py and the replay buffer compute
        state_size without importing magic constants.
        """
        H = self.cfg.grid.height
        W = self.cfg.grid.width
        N = self.cfg.agents.count
        return H * W + N * 6

    def get_global_state(self) -> np.ndarray:
        """
        Return the full global state vector for the QMIX mixing network.

        WHY THIS EXISTS:
          Individual agents can only see a 5×5 local window (partial obs).
          But the QMIX mixing network — which runs only during centralised
          TRAINING — needs the full picture to learn how to weight each
          agent's Q-value contribution correctly.

          At EXECUTION time, this method is never called. Each robot acts
          using only its own local observation. This is the CTDE principle.

        Returns
        -------
        np.ndarray
            Shape (state_size,) — flat float32 vector.
            Concatenation of:
              [grid_flat (H*W,)] + [agent_0_state (6,)] + ... + [agent_N_state (6,)]
        """
        H = self.cfg.grid.height
        W = self.cfg.grid.width

        # Full grid as normalised floats (divide by max cell type value)
        grid_flat = self._grid.flatten().astype(np.float32) / float(max(CellType))

        # Per-agent state vectors — always in fixed order (possible_agents),
        # padding terminated agents with zeros so the vector is constant-length.
        agent_vecs = []
        for agent_id in self.possible_agents:
            if agent_id in self._agent_states:
                s = self._agent_states[agent_id]
                vec = np.array([
                    s.row  / H,
                    s.col  / W,
                    s.battery / self.cfg.agents.battery_capacity,
                    float(s.carrying),
                    s.target_row / H if s.target_row >= 0 else -1.0,
                    s.target_col / W if s.target_col >= 0 else -1.0,
                ], dtype=np.float32)
            else:
                # Agent has been terminated — fill with zeros
                vec = np.zeros(6, dtype=np.float32)
            agent_vecs.append(vec)

        return np.concatenate([grid_flat] + agent_vecs)  # shape: (state_size,)

    def get_stats(self) -> dict:
        """
        Return a snapshot of the current episode's aggregate statistics.
        Called by trainer.py after each episode to feed into MLflow.

        Returns
        -------
        dict with keys:
          step_count, deliveries, frozen_agents, live_agents,
          items_available, total_battery_pct
        """
        total_bat = sum(
            s.battery / self.cfg.agents.battery_capacity
            for s in self._agent_states.values()
        )
        return {
            "step_count":        self._step_count,
            "deliveries":        len(self._delivery_history),
            "frozen_agents":     sum(1 for s in self._agent_states.values() if s.frozen),
            "live_agents":       len(self.agents),
            "items_available":   sum(self._items_on_shelves.values()),
            "total_battery_pct": total_bat / max(len(self._agent_states), 1),
        }

"""
renderer.py
-----------
Pygame-based renderer for the WarehouseRL environment.

WHY THIS EXISTS:
  The PettingZoo env produces simulation state (numpy arrays).
  This module turns that state into pixels — either for real-time
  display during development or frame-by-frame capture during
  evaluation for MP4 export and the frontend trajectory player.

KEY CONCEPTS:
  - Completely decoupled from the env: takes an env instance + agent
    states as input, draws the grid, returns an RGB numpy array.
  - recorder.py calls render_frame() once per step and accumulates
    frames into a video.
  - [V2-READY]: color_overrides parameter allows V2's live inference
    mode to highlight specific agents or cells on demand.

COLOR SCHEME (matches frontend CSS design tokens):
  Empty:    #04070f (near-black bg)
  Shelf:    #1a2d4a (surface-border)
  Item:     #ffaa00 (warning amber — item waiting to be picked up)
  Dispatch: #0066ff (accent blue)
  Charger:  #00cc88 (success green)
  Wall:     #2a3a4a (slightly lighter than bg)
  Agent (moving):   #4a6080 (muted — grey, en-route to shelf)
  Agent (carrying): #0066ff (accent blue — carrying item)
  Agent (low bat.): #ffaa00 (warning — battery < 20%)
  Agent (frozen):   #ff3355 (danger red — dead battery)
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.environment.warehouse_env import WarehouseEnv

# Colour palette (RGB tuples)
_COLORS = {
    "bg":       (4,   7,  15),
    "empty":    (10,  22, 40),
    "shelf":    (26,  45, 74),
    "item":     (255, 170, 0),
    "dispatch": (0,   102, 255),
    "charger":  (0,   204, 136),
    "wall":     (42,  58, 74),
    "agent_moving":   (74,  96, 128),
    "agent_carrying": (0,   102, 255),
    "agent_low_bat":  (255, 170, 0),
    "agent_frozen":   (255, 51,  85),
    "text":     (232, 237, 245),
}

CELL_PX = 32           # Each grid cell renders at 32×32 pixels
INFO_PANEL_W = 200     # Right-side info panel width in pixels


def render_frame(
    env: "WarehouseEnv",
    cell_px: int = CELL_PX,
    color_overrides: dict[str, Any] | None = None,  # [V2-READY]
) -> np.ndarray:
    """
    Render the current environment state to an RGB numpy array.

    Parameters
    ----------
    env : WarehouseEnv
        The environment to render. Must have been reset() at least once.
    cell_px : int
        Pixel size per grid cell. Default 32.
    color_overrides : dict | None
        [V2-READY] Override colours for specific agent IDs or cell types.
        e.g. {"agent_0": (255, 0, 0)} highlights agent 0 in red.

    Returns
    -------
    np.ndarray
        RGB array of shape (H*cell_px, W*cell_px + INFO_PANEL_W, 3).
    """
    import pygame
    pygame.init()

    H = env.cfg.grid.height
    W = env.cfg.grid.width
    canvas_h = H * cell_px
    canvas_w = W * cell_px + INFO_PANEL_W

    surface = pygame.Surface((canvas_w, canvas_h))
    surface.fill(_COLORS["bg"])

    # Draw grid cells
    for r in range(H):
        for c in range(W):
            cell_val = env._grid[r][c]
            color = _cell_color(cell_val)
            rect = pygame.Rect(c * cell_px, r * cell_px, cell_px - 1, cell_px - 1)
            pygame.draw.rect(surface, color, rect)

    # Draw agents on top
    for agent_id, state in env._agent_states.items():
        color = _agent_color(state, env.cfg.agents.battery_capacity)
        if color_overrides and agent_id in color_overrides:
            color = color_overrides[agent_id]
        cx = state.col * cell_px + cell_px // 2
        cy = state.row * cell_px + cell_px // 2
        pygame.draw.circle(surface, color, (cx, cy), cell_px // 2 - 3)

        # Carrying indicator: small dot in center
        if state.carrying:
            pygame.draw.circle(surface, _COLORS["item"], (cx, cy), 4)

    # Draw info panel
    _draw_info_panel(surface, env, canvas_w - INFO_PANEL_W, cell_px)

    # Convert surface to RGB numpy array
    return np.transpose(
        np.array(pygame.surfarray.array3d(surface)), axes=(1, 0, 2)
    )


def _cell_color(cell_val: int) -> tuple[int, int, int]:
    """Map a CellType integer to an RGB colour tuple."""
    from src.environment.warehouse_env import CellType
    mapping = {
        CellType.EMPTY:    _COLORS["empty"],
        CellType.SHELF:    _COLORS["shelf"],
        CellType.ITEM:     _COLORS["item"],
        CellType.DISPATCH: _COLORS["dispatch"],
        CellType.CHARGER:  _COLORS["charger"],
        CellType.WALL:     _COLORS["wall"],
        CellType.AGENT:    _COLORS["agent_moving"],
    }
    return mapping.get(cell_val, _COLORS["empty"])


def _agent_color(state: Any, battery_capacity: int) -> tuple[int, int, int]:
    """Pick agent dot colour based on its state."""
    if state.frozen:
        return _COLORS["agent_frozen"]
    if state.battery / battery_capacity < 0.2:
        return _COLORS["agent_low_bat"]
    if state.carrying:
        return _COLORS["agent_carrying"]
    return _COLORS["agent_moving"]


def _draw_info_panel(
    surface: Any,
    env: "WarehouseEnv",
    panel_x: int,
    cell_px: int,
) -> None:
    """Draw the right-side info panel showing agent statuses and step count."""
    import pygame
    pygame.font.init()
    font_sm = pygame.font.SysFont("monospace", 11)
    font_lg = pygame.font.SysFont("monospace", 14, bold=True)

    # Panel background
    panel_rect = pygame.Rect(panel_x, 0, INFO_PANEL_W, surface.get_height())
    pygame.draw.rect(surface, (10, 22, 40), panel_rect)
    pygame.draw.line(surface, _COLORS["wall"], (panel_x, 0), (panel_x, surface.get_height()), 1)

    y = 10
    step_surf = font_lg.render(f"Step: {env._step_count}", True, _COLORS["text"])
    surface.blit(step_surf, (panel_x + 8, y))
    y += 24

    for agent_id, state in env._agent_states.items():
        pct = state.battery / env.cfg.agents.battery_capacity
        bat_color = _COLORS["charger"] if pct > 0.5 else (_COLORS["item"] if pct > 0.2 else _COLORS["agent_frozen"])
        line = f"{agent_id[-7:]} bat:{int(pct*100):3d}% {'[C]' if state.carrying else '   '}"
        surf = font_sm.render(line, True, bat_color)
        surface.blit(surf, (panel_x + 8, y))
        y += 15

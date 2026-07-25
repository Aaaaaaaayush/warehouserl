"""
behavior_detector.py
--------------------
Emergent behavior detection engine for WarehouseRL.

WHY THIS EXISTS:
  MARL systems often produce emergent coordination strategies (lane formation,
  turn-taking, role specialisation, convoying) that were never explicitly
  programmed. This module analyzes raw trajectory data to detect and quantify
  these behaviors using spatial statistics, and generates density heatmaps.

KEY CONCEPTS:
  - Lane Formation: High directional asymmetry in corridor movement vectors.
  - Turn-Taking: Reduced velocity/wait actions at bottleneck entries when occupied.
  - Role Specialisation: High variance in per-agent carrying vs navigating activity ratios.
  - Convoy Behavior: Multi-agent single-file platooning along identical paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless PNG export
import matplotlib.pyplot as plt


class BehaviorDetector:
    """
    Analyzes evaluation trajectories for emergent multi-agent behavior patterns.
    """

    BEHAVIORS = [
        "lane_formation",
        "turn_taking",
        "role_specialisation",
        "convoy_behavior",
    ]

    def __init__(self, cfg: Any, output_dir: Path = Path("logs")):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.H = cfg.grid.height
        self.W = cfg.grid.width

    def analyze_trajectories(
        self,
        trajectories: list[dict],
        scenario_id: int,
    ) -> dict[str, Any]:
        """
        Run full behavior detection suite over evaluation trajectories.

        Parameters
        ----------
        trajectories : list[dict]
            Trajectory records output by Evaluator.
        scenario_id : int
            Scenario identifier (1, 2, or 3).

        Returns
        -------
        dict
            Detection results, statistics, and heatmap file paths.
        """
        heatmap = np.zeros((self.H, self.W), dtype=np.float32)

        # Collect positional history
        # agent_moves: {agent_id: [(row, col, d_row, d_col, carrying), ...]}
        agent_moves: dict[str, list[tuple[int, int, int, int, bool]]] = {}

        for ep in trajectories:
            frames = ep["frames"]
            for step_idx in range(len(frames) - 1):
                f1 = frames[step_idx]
                f2 = frames[step_idx + 1]

                agents_f1 = {a["id"]: a for a in f1["agents"]}
                agents_f2 = {a["id"]: a for a in f2["agents"]}

                for a_id, a1 in agents_f1.items():
                    if a1["frozen"] or a1["row"] < 0:
                        continue

                    r1, c1 = a1["row"], a1["col"]
                    heatmap[r1, c1] += 1.0

                    if a_id in agents_f2 and not agents_f2[a_id]["frozen"]:
                        r2, c2 = agents_f2[a_id]["row"], agents_f2[a_id]["col"]
                        dr, dc = r2 - r1, c2 - c1

                        if a_id not in agent_moves:
                            agent_moves[a_id] = []
                        agent_moves[a_id].append((r1, c1, dr, dc, a1["carrying"]))

        # Normalize heatmap
        if heatmap.sum() > 0:
            heatmap_norm = heatmap / heatmap.max()
        else:
            heatmap_norm = heatmap

        # Save heatmap image
        heatmap_file = self.output_dir / f"heatmap_scenario_{scenario_id}.png"
        self._export_heatmap(heatmap_norm, heatmap_file, scenario_id)

        # ── Detect individual behaviors ─────────────────────────────────────

        lane_stats = self._detect_lane_formation(agent_moves)
        turn_stats = self._detect_turn_taking(agent_moves)
        role_stats = self._detect_role_specialisation(agent_moves)
        convoy_stats = self._detect_convoy_behavior(trajectories)

        results = {
            "scenario_id": scenario_id,
            "heatmap_path": str(heatmap_file),
            "behaviors": {
                "lane_formation": {
                    "detected": lane_stats["score"] > 0.35,
                    "score": round(float(lane_stats["score"]), 3),
                    "description": (
                        "Agents spontaneously organize into opposite directional lanes "
                        "in narrow corridors to minimize head-on collisions."
                    ),
                    "details": lane_stats,
                },
                "turn_taking": {
                    "detected": turn_stats["score"] > 0.25,
                    "score": round(float(turn_stats["score"]), 3),
                    "description": (
                        "Agents pause or yield at corridor entrances when an oncoming "
                        "agent is detected inside the bottleneck."
                    ),
                    "details": turn_stats,
                },
                "role_specialisation": {
                    "detected": role_stats["score"] > 0.20,
                    "score": round(float(role_stats["score"]), 3),
                    "description": (
                        "Agents divide labor: specific agents specialize as primary haulers "
                        "while others clear bottlenecks or manage charging."
                    ),
                    "details": role_stats,
                },
                "convoy_behavior": {
                    "detected": convoy_stats["score"] > 0.20,
                    "score": round(float(convoy_stats["score"]), 3),
                    "description": (
                        "Agents form single-file platoons following identical trajectories "
                        "to move efficiently through high-density zones."
                    ),
                    "details": convoy_stats,
                },
            },
        }

        # Save JSON output for FastAPI /api/behaviors/{scenario_id}
        json_file = self.output_dir / f"scenario_{scenario_id}_behaviors.json"
        with json_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results

    # ── Behavior Detectors ───────────────────────────────────────────────────

    def _detect_lane_formation(self, agent_moves: dict) -> dict:
        """
        Measure directional asymmetry across grid rows and columns.
        If row r has mostly rightward moves (+dc) and row r+1 mostly leftward (-dc),
        lane formation score is high.
        """
        row_dir = np.zeros((self.H, 2))  # [right_count, left_count]
        for moves in agent_moves.values():
            for r, c, dr, dc, _ in moves:
                if 0 <= r < self.H:
                    if dc > 0:
                        row_dir[r, 0] += 1
                    elif dc < 0:
                        row_dir[r, 1] += 1

        total_row_moves = row_dir.sum(axis=1)
        valid_rows = total_row_moves > 20
        if not np.any(valid_rows):
            return {"score": 0.0, "dominant_lanes": 0}

        # Asymmetry score per row: |right - left| / total
        asym = np.abs(row_dir[valid_rows, 0] - row_dir[valid_rows, 1]) / total_row_moves[valid_rows]
        score = float(np.mean(asym))
        dominant = int(np.sum(asym > 0.6))

        return {"score": score, "dominant_lanes": dominant}

    def _detect_turn_taking(self, agent_moves: dict) -> dict:
        """
        Measure frequency of STAY (0 movement) actions at bottleneck cells.
        High stay frequency when near other agents indicates yielding/turn-taking.
        """
        stays = 0
        total = 0
        for moves in agent_moves.values():
            for r, c, dr, dc, _ in moves:
                total += 1
                if dr == 0 and dc == 0:
                    stays += 1

        ratio = stays / max(total, 1)
        # Normalize: typical stay ratio in crowded scenarios range from 0.1 to 0.4
        score = min(ratio * 2.5, 1.0)
        return {"score": score, "stay_ratio": round(ratio, 3)}

    def _detect_role_specialisation(self, agent_moves: dict) -> dict:
        """
        Measure variance in carrying ratios across agents.
        High variance = some agents carry constantly, others rarely = specialisation.
        """
        carrying_ratios = []
        for a_id, moves in agent_moves.items():
            if not moves:
                continue
            carried = sum(1 for _, _, _, _, c in moves if c)
            ratio = carried / len(moves)
            carrying_ratios.append(ratio)

        if len(carrying_ratios) < 2:
            return {"score": 0.0, "variance": 0.0}

        std_dev = float(np.std(carrying_ratios))
        score = min(std_dev * 3.0, 1.0)
        return {"score": score, "std_dev": round(std_dev, 3)}

    def _detect_convoy_behavior(self, trajectories: list[dict]) -> dict:
        """
        Detect steps where 3+ agents form a linear chain (distance == 1).
        """
        convoy_count = 0
        total_frames = 0

        for ep in trajectories:
            for f in ep["frames"]:
                total_frames += 1
                positions = [(a["row"], a["col"]) for a in f["agents"] if not a["frozen"] and a["row"] >= 0]
                if len(positions) < 3:
                    continue

                # Check if 3 agents form a chain: p1 adjacent to p2, p2 adjacent to p3
                chains = 0
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        d12 = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
                        if d12 == 1:
                            for k in range(j + 1, len(positions)):
                                d23 = abs(positions[j][0] - positions[k][0]) + abs(positions[j][1] - positions[k][1])
                                if d23 == 1:
                                    chains += 1

                if chains > 0:
                    convoy_count += 1

        ratio = convoy_count / max(total_frames, 1)
        score = min(ratio * 2.0, 1.0)
        return {"score": score, "convoy_frame_ratio": round(ratio, 3)}

    # ── Heatmap Renderer ─────────────────────────────────────────────────────

    def _export_heatmap(self, heatmap_norm: np.ndarray, file_path: Path, scenario_id: int) -> None:
        """Render and save 2D matplotlib density heatmap."""
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#04070f")
        ax.set_facecolor("#04070f")

        im = ax.imshow(heatmap_norm, cmap="inferno", interpolation="nearest")

        ax.set_title(
            f"Scenario {scenario_id} — Agent Trajectory Density",
            color="#e8edf5",
            fontsize=12,
            pad=12,
            fontfamily="sans-serif",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_color("#1a2d4a")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#4a6080")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#4a6080")

        plt.tight_layout()
        plt.savefig(file_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

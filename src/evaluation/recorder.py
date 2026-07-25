"""
recorder.py
-----------
Episode trajectory JSON and MP4 video exporter.

WHY THIS EXISTS:
  To present RL results cleanly, we need two formats:
  1. Trajectory JSON: raw structured frame data consumed by the HTML5 Canvas player
     in the frontend dashboard. Schema is [V2-READY] (same schema streamed via WS in V2).
  2. MP4 Video: rendered video clips used for generating high-quality README GIFs
     comparing Episode 1 (random chaos) vs Final (smooth coordination).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio
import numpy as np

from src.environment.config_loader import load_config
from src.environment.renderer import render_frame
from src.evaluation.evaluator import Evaluator


class EpisodeRecorder:
    """
    Records evaluation episodes and exports trajectory JSON + MP4 videos.

    Parameters
    ----------
    episodes_dir : Path
        Directory to save trajectory JSON files.
    videos_dir : Path
        Directory to save MP4 video recordings.
    device : str
        Torch device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        episodes_dir: Path = Path("episodes"),
        videos_dir: Path = Path("videos"),
        device: str = "cpu",
    ):
        self.episodes_dir = Path(episodes_dir)
        self.videos_dir = Path(videos_dir)
        self.device = device

        self.episodes_dir.mkdir(exist_ok=True)
        self.videos_dir.mkdir(exist_ok=True)

    def record_stage(
        self,
        scenario_id: int,
        checkpoint_path: Path,
        stage: str,
        fps: int = 15,
    ) -> tuple[Path, Path]:
        """
        Record one episode for a scenario stage and export both JSON and MP4.

        Parameters
        ----------
        scenario_id : int
            Scenario ID (1, 2, or 3).
        checkpoint_path : Path
            Path to model checkpoint.
        stage : str
            Stage label: "episode_1" | "25pct" | "50pct" | "75pct" | "final".
        fps : int
            Frame rate for exported MP4.

        Returns
        -------
        json_path, mp4_path : tuple[Path, Path]
        """
        cfg = load_config(scenario_id)
        evaluator = Evaluator(cfg=cfg, checkpoint_path=checkpoint_path, device=self.device)

        # Run 1 evaluation episode
        metrics, trajectories = evaluator.evaluate(n_episodes=1)
        traj = trajectories[0]

        # ── Export Trajectory JSON ───────────────────────────────────────────
        json_data = {
            "version": "1.0",   # [V2-READY]
            "scenario": scenario_id,
            "stage": stage,
            "grid": {
                "width": cfg.grid.width,
                "height": cfg.grid.height,
                "shelves": cfg.grid.shelves,
                "dispatch_points": cfg.grid.dispatch_points,
                "charging_stations": cfg.grid.charging_stations,
                "obstacles": cfg.grid.obstacles,
            },
            "agents_count": cfg.agents.count,
            "total_steps": traj["steps"],
            "deliveries": traj["deliveries"],
            "frames": traj["frames"],
        }

        json_path = self.episodes_dir / f"scenario_{scenario_id}_{stage}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        # ── Export MP4 Video ─────────────────────────────────────────────────
        mp4_path = self.videos_dir / f"scenario_{scenario_id}_{stage}.mp4"
        self._export_mp4(evaluator.env, traj["frames"], mp4_path, fps=fps)

        print(f"  [Recorder] Stage '{stage}' recorded:")
        print(f"    JSON: {json_path}")
        print(f"    MP4:  {mp4_path}")

        return json_path, mp4_path

    def _export_mp4(
        self,
        env: Any,
        frames_data: list[dict],
        output_path: Path,
        fps: int = 15,
    ) -> None:
        """Render RGB frames from trajectory frame data and write MP4."""
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)

        # Reconstruct env state step by step for Pygame renderer
        for f in frames_data:
            # Update internal env step count
            env._step_count = f["step"]

            # Update agent states
            for agent_info in f["agents"]:
                a_id = agent_info["id"]
                if a_id in env._agent_states:
                    st = env._agent_states[a_id]
                    st.row = agent_info["row"]
                    st.col = agent_info["col"]
                    st.battery = agent_info["battery"]
                    st.carrying = agent_info["carrying"]
                    st.frozen = agent_info["frozen"]

            # Render frame using renderer.py
            rgb_frame = render_frame(env)
            writer.append_data(rgb_frame)

        writer.close()

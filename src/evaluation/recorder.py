"""
recorder.py — Trajectory JSON + MP4 export.
Records 5 episodes per scenario (ep1, 25%, 50%, 75%, final).
[V2-READY]: JSON schema used by frontend Canvas + V2 live inference.
Full implementation: Step 4 (Episode Recording).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any


# Schema for trajectory JSON (V2-compatible)
# {
#   "scenario": int,
#   "stage": str,          # "episode_1" | "25pct" | "50pct" | "75pct" | "final"
#   "grid": {...},          # grid config snapshot
#   "frames": [            # one entry per step
#     {
#       "step": int,
#       "agents": [
#         {"id": str, "row": int, "col": int, "battery": float,
#          "carrying": bool, "frozen": bool}
#       ],
#       "items_on_shelves": [[row, col], ...],
#       "deliveries": int
#     }
#   ]
# }
TRAJECTORY_SCHEMA_VERSION = "1.0"   # [V2-READY]: bump to 2.0 for live inference


class EpisodeRecorder:
    """Records episodes to trajectory JSON and MP4."""

    def __init__(self, episodes_dir: Path, videos_dir: Path):
        self.episodes_dir = episodes_dir
        self.videos_dir = videos_dir

    def record(self, cfg: Any, checkpoint_path: Path, stage: str) -> Path:
        """Record one episode and export JSON + MP4. Returns JSON path."""
        raise NotImplementedError("Implemented in Step 4: Episode Recording")

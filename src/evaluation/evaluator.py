"""
evaluator.py — Episode runner and metric collection.
recorder.py will call this to collect trajectory data.
Full implementation: Step 4 (Evaluation + Emergent Behavior Detection).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any


class Evaluator:
    """Runs N evaluation episodes and collects per-step metrics."""

    def __init__(self, cfg: Any, checkpoint_path: Path, device: str = "cpu"):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.device = device

    def run(self, n_episodes: int = 1000) -> dict:
        """Run evaluation episodes. Returns dict of aggregate metrics."""
        raise NotImplementedError("Implemented in Step 4: Evaluation")

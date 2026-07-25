"""
curriculum.py
-------------
Curriculum manager — controls scenario progression and weight transfer.

WHY CURRICULUM LEARNING EXISTS:
  Training directly on the hardest scenario (12 agents, 16×16, dynamic
  obstacles) from scratch is extremely inefficient. Agents start with
  random behaviour and the credit assignment problem is vastly harder
  when many agents collide simultaneously.

  Curriculum learning stages difficulty: start simple (4 agents, 8×8),
  let agents learn the fundamentals (navigate, pick up, deliver), then
  transfer those weights to the next harder scenario where they already
  know the basics and only need to learn coordination at scale.

  Think of it like: you don't teach calculus before arithmetic.

WHY WEIGHT TRANSFER WORKS:
  The lower-level skills (navigation, battery management, basic pickup)
  are encoded in the QNetwork's weights. These skills generalise across
  grid sizes because the observation is normalised (positions divided by
  grid dimensions) and the local grid window is the same size (5×5).
  Only the coordination strategy needs to be re-learned at scale.

FULL IMPLEMENTATION: Coming in Step 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CurriculumManager:
    """
    Manages the three-scenario curriculum progression.

    Determines when to advance scenarios and handles weight transfer
    between QMIXTrainer instances across scenario boundaries.
    """

    SCENARIO_ORDER = [1, 2, 3]

    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = models_dir
        self.current_scenario = 1

    def get_transfer_checkpoint(self, scenario_id: int) -> Path | None:
        """
        Return the checkpoint path to initialise scenario N from.
        Returns None for Scenario 1 (no prior training).
        """
        raise NotImplementedError("Implemented in Step 3: Training Pipeline")

    def run_full_curriculum(self, base_cfg_dir: Path = Path("configs")) -> None:
        """
        Run all three scenarios in sequence with weight transfer.
        Logs aggregate metrics across all scenarios to MLflow.
        """
        raise NotImplementedError("Implemented in Step 3: Training Pipeline")

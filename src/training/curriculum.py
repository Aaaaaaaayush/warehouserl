"""
curriculum.py
-------------
Curriculum manager — runs all three scenarios in sequence with weight transfer.

WHY CURRICULUM LEARNING:
  Training directly on Scenario 3 (12 agents, 16×16, random obstacles) from
  scratch is extremely sample-inefficient. Agents start random, and with 12
  robots colliding constantly, the credit assignment signal is overwhelmed
  by noise. The training loss oscillates and convergence takes 10× longer.

  Curriculum learning stages difficulty:
    S1: Learn to navigate, pick up, deliver (4 agents, simple grid)
    S2: Learn routing and lane formation (8 agents, shelf clusters)
    S3: Learn role specialisation under dynamic conditions (12 agents)

  At each stage transition, QNetwork weights are transferred.
  The QNetwork has the same architecture across all scenarios (obs_size=31,
  act_size=7, hidden_size=64) because observation space is normalised.
  Only the mixing network is reinitialized (n_agents and state_size differ).

HOW TO USE:
  # Run full curriculum (all 3 scenarios, ~900K total episodes)
  from src.training.curriculum import CurriculumManager
  mgr = CurriculumManager()
  mgr.run_full_curriculum()

  # Run a single scenario
  from src.training.trainer import QMIXTrainer
  from src.environment.config_loader import load_config
  trainer = QMIXTrainer(load_config(1), device='cuda')
  trainer.train()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow

from src.environment.config_loader import load_config
from src.training.trainer import QMIXTrainer


class CurriculumManager:
    """
    Orchestrates the three-scenario curriculum with weight transfer.

    Parameters
    ----------
    models_dir : Path
        Directory where checkpoints are saved and looked up.
    logs_dir : Path
        Directory for aggregate curriculum metrics.
    device : str
        Torch device ('cuda' for RTX 5080, 'cpu' for Oracle).
    """

    SCENARIO_ORDER = [1, 2, 3]

    def __init__(
        self,
        models_dir: Path = Path("models"),
        logs_dir:   Path = Path("logs"),
        device:     str  = "cuda",
    ):
        self.models_dir = models_dir
        self.logs_dir   = logs_dir
        self.device     = device
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

    def get_transfer_checkpoint(self, scenario_id: int) -> Path | None:
        """
        Find the most recent checkpoint to use as weights for scenario_id.

        For scenario N, looks for the final checkpoint from scenario N-1.
        The final checkpoint is named: qmix_{n}agents_{total_episodes}episodes.pt

        Returns None for Scenario 1 (no prior training exists).
        """
        if scenario_id == 1:
            return None

        source_id = scenario_id - 1
        source_cfg = load_config(source_id)
        n_agents  = source_cfg.agents.count
        total_eps = source_cfg.training.total_episodes

        # Look for the exact final checkpoint first
        final_ckpt = self.models_dir / f"qmix_{n_agents}agents_{total_eps}episodes.pt"
        if final_ckpt.exists():
            return final_ckpt

        # Fall back to any checkpoint from the source scenario (most recent)
        candidates = sorted(
            self.models_dir.glob(f"qmix_{n_agents}agents_*episodes.pt")
        )
        if candidates:
            print(
                f"  [Curriculum] Final checkpoint not found for S{source_id}. "
                f"Using most recent: {candidates[-1].name}"
            )
            return candidates[-1]

        print(
            f"  [Curriculum] WARNING: No checkpoint found for Scenario {source_id}. "
            f"Scenario {scenario_id} will start from random initialisation."
        )
        return None

    def run_scenario(self, scenario_id: int) -> dict:
        """
        Run one scenario of the curriculum.

        Automatically finds and loads the transfer checkpoint from
        the previous scenario if available.

        Parameters
        ----------
        scenario_id : int
            Which scenario to train (1, 2, or 3).

        Returns
        -------
        dict
            Final training metrics for this scenario.
        """
        cfg = load_config(scenario_id)
        transfer_ckpt = self.get_transfer_checkpoint(scenario_id)

        print(f"\n{'='*60}")
        print(f"  SCENARIO {scenario_id}: {cfg.scenario.name}")
        print(f"  Agents: {cfg.agents.count}  |  "
              f"Grid: {cfg.grid.width}×{cfg.grid.height}  |  "
              f"Episodes: {cfg.training.total_episodes:,}")
        if transfer_ckpt:
            print(f"  Transfer weights: {transfer_ckpt.name}")
        else:
            print(f"  Transfer weights: None (random init)")
        print(f"{'='*60}\n")

        trainer = QMIXTrainer(
            cfg=cfg,
            device=self.device,
            run_name=f"qmix_scenario_{scenario_id}_{cfg.scenario.name.replace(' ', '_')}",
            checkpoint_dir=self.models_dir,
        )

        final_metrics = trainer.train(
            transfer_checkpoint=transfer_ckpt,
            mlflow_experiment="WarehouseRL_Curriculum",
        )

        # Save scenario summary to logs
        summary_path = self.logs_dir / f"scenario_{scenario_id}_summary.json"
        with summary_path.open("w") as f:
            json.dump({
                "scenario_id":   scenario_id,
                "scenario_name": cfg.scenario.name,
                "agents":        cfg.agents.count,
                "grid":          f"{cfg.grid.width}x{cfg.grid.height}",
                "episodes":      cfg.training.total_episodes,
                "transfer_from": str(transfer_ckpt) if transfer_ckpt else None,
                "final_metrics": final_metrics,
            }, f, indent=2)

        print(f"\n[OK] Scenario {scenario_id} complete. Summary: {summary_path}")
        return final_metrics

    def run_full_curriculum(self) -> dict[int, dict]:
        """
        Run all three scenarios in sequence.
        Scenario 1 → Scenario 2 (transfer) → Scenario 3 (transfer).

        Total episodes: 100K + 300K + 500K = 900K.

        Returns
        -------
        dict[scenario_id → final_metrics]
        """
        print("\n" + "="*60)
        print("  WarehouseRL — Full Curriculum Training")
        print("  900,000 total episodes across 3 scenarios")
        print("  Device:", self.device)
        print("="*60)

        all_metrics = {}
        for sid in self.SCENARIO_ORDER:
            metrics = self.run_scenario(sid)
            all_metrics[sid] = metrics

        # Write aggregate summary for the overview panel hero metric
        total_packages = sum(
            m["packages_delivered"] * load_config(sid).training.total_episodes
            for sid, m in all_metrics.items()
        )
        aggregate = {
            "total_episodes":        900_000,
            "total_packages_approx": int(total_packages),
            "per_scenario":          {str(k): v for k, v in all_metrics.items()},
        }
        agg_path = self.logs_dir / "curriculum_aggregate.json"
        with agg_path.open("w") as f:
            json.dump(aggregate, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  CURRICULUM COMPLETE")
        print(f"  Estimated total packages delivered: {int(total_packages):,}")
        print(f"  Aggregate summary: {agg_path}")
        print(f"{'='*60}\n")

        return all_metrics

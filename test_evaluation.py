"""
test_evaluation.py — Step 4 test suite.

Tests:
  - Evaluator runs deterministic episodes without errors
  - BehaviorDetector parses trajectories and outputs correct JSON & PNG heatmap
  - EpisodeRecorder generates valid trajectory JSON and MP4 video file
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path
import pytest

from src.environment.config_loader import load_config
from src.evaluation.evaluator import Evaluator
from src.evaluation.behavior_detector import BehaviorDetector
from src.evaluation.recorder import EpisodeRecorder


@pytest.fixture
def smoke_checkpoint():
    """Ensure a checkpoint exists for Scenario 1 tests."""
    ckpt_path = Path("models/qmix_4agents_500episodes.pt")
    if not ckpt_path.exists():
        # Fallback to dummy training if not created
        from src.training.trainer import QMIXTrainer
        cfg = load_config(1)
        cfg.training.total_episodes = 10
        trainer = QMIXTrainer(cfg=cfg, device="cpu", checkpoint_dir=Path("models"))
        trainer.train()
    return ckpt_path


def test_evaluator_run(smoke_checkpoint):
    cfg = load_config(1)
    evaluator = Evaluator(cfg=cfg, checkpoint_path=smoke_checkpoint, device="cpu")
    metrics, trajectories = evaluator.evaluate(n_episodes=5)

    assert "mean_reward" in metrics
    assert "mean_deliveries" in metrics
    assert len(trajectories) == 5
    assert "frames" in trajectories[0]


def test_behavior_detector(smoke_checkpoint):
    cfg = load_config(1)
    evaluator = Evaluator(cfg=cfg, checkpoint_path=smoke_checkpoint, device="cpu")
    _, trajectories = evaluator.evaluate(n_episodes=5)

    detector = BehaviorDetector(cfg=cfg, output_dir=Path("logs"))
    results = detector.analyze_trajectories(trajectories, scenario_id=1)

    assert "behaviors" in results
    assert "lane_formation" in results["behaviors"]
    assert Path(results["heatmap_path"]).exists()
    assert Path("logs/scenario_1_behaviors.json").exists()


def test_episode_recorder(smoke_checkpoint):
    recorder = EpisodeRecorder(
        episodes_dir=Path("episodes"),
        videos_dir=Path("videos"),
        device="cpu",
    )
    json_path, mp4_path = recorder.record_stage(
        scenario_id=1,
        checkpoint_path=smoke_checkpoint,
        stage="final",
    )

    assert json_path.exists()
    assert mp4_path.exists()

    with json_path.open() as f:
        data = json.load(f)
        assert data["scenario"] == 1
        assert data["stage"] == "final"
        assert len(data["frames"]) > 0

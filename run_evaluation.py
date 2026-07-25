"""
run_evaluation.py
-----------------
Entry-point script for running evaluation, emergent behavior detection,
and episode trajectory/video recording on trained model checkpoints.

Usage examples:

  # Evaluate a scenario checkpoint (e.g. Scenario 1)
  python run_evaluation.py --scenario 1 --checkpoint models/qmix_4agents_500episodes.pt

  # Run full evaluation + behavior detection + video export
  python run_evaluation.py --scenario 1 --checkpoint models/qmix_4agents_500episodes.pt --record
"""

import argparse
from pathlib import Path

import torch

from src.environment.config_loader import load_config
from src.evaluation.evaluator import Evaluator
from src.evaluation.behavior_detector import BehaviorDetector
from src.evaluation.recorder import EpisodeRecorder


def parse_args():
    p = argparse.ArgumentParser(
        description="WarehouseRL Evaluation & Behavior Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Scenario ID to evaluate.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of deterministic evaluation episodes to run.",
    )
    p.add_argument(
        "--record",
        action="store_true",
        help="If set, exports trajectory JSON and MP4 video recordings.",
    )
    p.add_argument(
        "--stage",
        type=str,
        default="final",
        choices=["episode_1", "25pct", "50pct", "75pct", "final"],
        help="Stage label for recorded artifacts.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device ('cpu' or 'cuda').",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"Error: Checkpoint file '{ckpt_path}' does not exist.")
        return

    cfg = load_config(args.scenario)

    print("\n" + "=" * 60)
    print(f"  WarehouseRL Evaluation — Scenario {args.scenario}: {cfg.scenario.name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Episodes:   {args.eval_episodes} (deterministic, epsilon=0.0)")
    print("=" * 60 + "\n")

    # 1. Run Evaluator
    evaluator = Evaluator(cfg=cfg, checkpoint_path=ckpt_path, device=args.device)
    metrics, trajectories = evaluator.evaluate(n_episodes=args.eval_episodes)

    print("=== Evaluation Metrics ===")
    print(f"  Mean Reward:             {metrics['mean_reward']:.2f}")
    print(f"  Packages Delivered:      {metrics['mean_deliveries']:.2f} per episode")
    print(f"  Collision Rate:          {metrics['collision_rate']:.4f}")
    print(f"  Deadlock Frequency:      {metrics['deadlock_frequency']:.4f}")
    print(f"  Battery Depletion Evts:  {metrics['mean_battery_depletion']:.2f}")
    print()

    # 2. Run Behavior Detector
    print("=== Emergent Behavior Analysis ===")
    detector = BehaviorDetector(cfg=cfg)
    behavior_results = detector.analyze_trajectories(trajectories, scenario_id=args.scenario)

    for b_name, b_data in behavior_results["behaviors"].items():
        status = "DETECTED [✓]" if b_data["detected"] else "Not Detected [x]"
        print(f"  - {b_name:20s}: {status} (score: {b_data['score']:.3f})")
    print(f"  Heatmap saved to: {behavior_results['heatmap_path']}")
    print()

    # 3. Export Recordings if requested
    if args.record:
        print("=== Recording Stage Artifacts ===")
        recorder = EpisodeRecorder(device=args.device)
        json_path, mp4_path = recorder.record_stage(
            scenario_id=args.scenario,
            checkpoint_path=ckpt_path,
            stage=args.stage,
        )
        print(f"  [✓] Artifacts saved for stage '{args.stage}'")

    print("\n=== Evaluation Pipeline Complete ===")


if __name__ == "__main__":
    main()

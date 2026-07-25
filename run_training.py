"""
run_training.py
---------------
Entry-point script for WarehouseRL training.

Usage examples:

  # Run the full 900K episode curriculum (all 3 scenarios)
  python run_training.py --mode curriculum

  # Train a single scenario only
  python run_training.py --mode single --scenario 1

  # Resume from a checkpoint (e.g., after interruption)
  python run_training.py --mode single --scenario 2 --transfer models/qmix_4agents_100000episodes.pt

  # Smoke test: 500 episodes of Scenario 1 to verify training works
  python run_training.py --mode smoke

Run from the project root with the venv active:
  .venv_wrl\\Scripts\\activate
  python run_training.py --mode smoke
"""

import argparse
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="WarehouseRL QMIX Training Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["curriculum", "single", "smoke"],
        default="smoke",
        help=(
            "curriculum: run all 3 scenarios in sequence (900K eps total). "
            "single: run one scenario. "
            "smoke: 500 eps of Scenario 1 to verify everything works."
        ),
    )
    p.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Which scenario to train (only used with --mode single).",
    )
    p.add_argument(
        "--transfer",
        type=str,
        default=None,
        help="Path to checkpoint to load weights from (optional).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. Defaults to 'cuda' if available.",
    )
    p.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  WarehouseRL Training — {args.mode.upper()} mode")
    print(f"  Device: {args.device}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:   {vram:.1f} GB")
    print(f"{'='*60}")

    if args.mode == "curriculum":
        from src.training.curriculum import CurriculumManager
        mgr = CurriculumManager(
            models_dir=Path(args.models_dir),
            device=args.device,
        )
        mgr.run_full_curriculum()

    elif args.mode == "single":
        from src.environment.config_loader import load_config
        from src.training.trainer import QMIXTrainer

        cfg = load_config(args.scenario)
        transfer = Path(args.transfer) if args.transfer else None

        trainer = QMIXTrainer(
            cfg=cfg,
            device=args.device,
            checkpoint_dir=Path(args.models_dir),
        )
        metrics = trainer.train(transfer_checkpoint=transfer)
        print("\nFinal metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    elif args.mode == "smoke":
        # Quick smoke test: 500 episodes of Scenario 1
        # Overrides total_episodes so it finishes in ~30 seconds
        from src.environment.config_loader import load_config
        from src.training.trainer import QMIXTrainer

        print("\n[Smoke] Running 500-episode test of Scenario 1...")
        cfg = load_config(1)
        cfg.training.total_episodes       = 500
        cfg.training.eval_interval        = 100
        cfg.training.checkpoint_interval  = 500
        cfg.training.buffer_capacity      = 200
        cfg.training.batch_size           = 16

        trainer = QMIXTrainer(
            cfg=cfg,
            device=args.device,
            run_name="smoke_test",
            checkpoint_dir=Path(args.models_dir),
        )
        metrics = trainer.train()

        print("\n[Smoke] Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print("\n[Smoke] Training pipeline VERIFIED.")


if __name__ == "__main__":
    main()

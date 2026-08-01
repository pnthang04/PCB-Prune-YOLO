"""Fine-tune a safely produced pruned checkpoint."""

import argparse
from pathlib import Path

from pcb_prune_yolo.training.trainer import train_pruned


def main() -> None:
    """Fine-tune with a configurable small learning rate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/data/deeppcb.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="outputs/finetune")
    parser.add_argument("--name", default="pruned")
    args = parser.parse_args()
    train_pruned(
        {
            "model": str(args.model),
            "data": str(args.data),
            "epochs": args.epochs,
            "lr0": args.lr0,
            "device": args.device,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "patience": args.patience,
            "workers": args.workers,
            "seed": 42,
            "amp": True,
            "deterministic": True,
            "project": args.project,
            "name": args.name,
        }
    )


if __name__ == "__main__":
    main()

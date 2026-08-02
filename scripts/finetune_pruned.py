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
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="outputs/finetune")
    parser.add_argument("--name", default="pruned")
    parser.add_argument(
        "--distill-model",
        type=Path,
        default=None,
        help="Teacher checkpoint for Ultralytics' native knowledge distillation",
    )
    parser.add_argument(
        "--dis",
        type=float,
        default=6.0,
        help="Distillation loss weight (Ultralytics default), used only with --distill-model",
    )
    args = parser.parse_args()
    config = {
        "model": str(args.model),
        "data": str(args.data),
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
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
    if args.distill_model is not None:
        config["distill_model"] = str(args.distill_model)
        config["dis"] = args.dis
    train_pruned(config)


if __name__ == "__main__":
    main()

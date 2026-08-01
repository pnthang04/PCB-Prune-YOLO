"""Train the baseline YOLOv8n model."""

import argparse
from pathlib import Path

from pcb_prune_yolo.config import load_config
from pcb_prune_yolo.training.trainer import train


def main() -> None:
    """Load YAML, apply explicit CLI overrides, and train."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train/yolov8n_baseline.yaml"))
    for name, kind in (
        ("data", str),
        ("epochs", int),
        ("patience", int),
        ("imgsz", int),
        ("batch", int),
        ("device", str),
        ("workers", int),
        ("seed", int),
        ("fraction", float),
        ("project", str),
        ("name", str),
    ):
        parser.add_argument(f"--{name}", type=kind)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short 1-epoch DDP pipeline check on 10%% of the training set",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config.update(
        {
            key: value
            for key, value in vars(args).items()
            if key not in {"config", "smoke"} and value is not None
        }
    )
    if args.smoke:
        config.update(
            {
                "epochs": 1,
                "patience": 1,
                "fraction": args.fraction if args.fraction is not None else 0.1,
                "val": False,
                "plots": False,
                "save": False,
                "name": "smoke",
            }
        )
    train(config)


if __name__ == "__main__":
    main()

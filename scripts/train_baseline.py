"""Train the baseline YOLOv8n model."""

import argparse
from pathlib import Path

from pcb_prune_yolo.config import load_config, save_config
from pcb_prune_yolo.training.trainer import train


def main() -> None:
    """Load YAML, apply explicit CLI overrides, and train."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train/yolov8n_baseline.yaml"))
    for name, kind in (("data", str), ("epochs", int), ("imgsz", int), ("batch", int), ("device", str), ("workers", int), ("seed", int), ("project", str), ("name", str)):
        parser.add_argument(f"--{name}", type=kind)
    args = parser.parse_args()
    config = load_config(args.config)
    config.update({key: value for key, value in vars(args).items() if key != "config" and value is not None})
    output = Path(config.get("project", "outputs/train")) / str(config.get("name", "baseline"))
    save_config(config, output / "used_config.yaml")
    train(config)


if __name__ == "__main__":
    main()


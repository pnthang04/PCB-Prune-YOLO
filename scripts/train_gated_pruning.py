"""Train DiariZen-style Hard-Concrete gates for YOLO structured pruning."""

import argparse
from pathlib import Path

from pcb_prune_yolo.config import load_config
from pcb_prune_yolo.training.trainer import train_gated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/prune/gated_p10.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.name is not None:
        config["name"] = args.name
    train_gated(config)


if __name__ == "__main__":
    main()

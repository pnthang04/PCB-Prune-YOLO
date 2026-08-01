"""Evaluate a YOLO checkpoint and save standard metrics."""

import argparse
from pathlib import Path

from pcb_prune_yolo.benchmarking.report import write_report
from pcb_prune_yolo.training.evaluator import evaluate


def main() -> None:
    """Run Ultralytics validation and write JSON/CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/data/deeppcb.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    write_report(evaluate(args.checkpoint, args.data, args.split, args.device), args.output, "metrics")


if __name__ == "__main__":
    main()


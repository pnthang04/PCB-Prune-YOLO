"""Run a short TensorRT-oriented INT8 QAT experiment from a fixed checkpoint."""

import argparse
from pathlib import Path

from pcb_prune_yolo.config import load_config
from pcb_prune_yolo.training.trainer import train_qat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qat/p30_int8_qat.yaml"))
    args = parser.parse_args()
    train_qat(load_config(args.config))


if __name__ == "__main__":
    main()

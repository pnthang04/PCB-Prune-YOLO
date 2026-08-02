"""Run DepGraph group-level sparse training from the unpruned baseline."""

import argparse
from pathlib import Path

from pcb_prune_yolo.config import load_config
from pcb_prune_yolo.training.trainer import train_sparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/prune/depgraph_sparse.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--name", type=str)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("epochs", "batch", "device", "fraction", "name"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.smoke:
        config.update(
            {
                "epochs": args.epochs or 2,
                "fraction": args.fraction or 0.1,
                "name": args.name or "depgraph_sparse_smoke",
                "plots": False,
            }
        )
        if int(config["epochs"]) > 3:
            raise ValueError("Sparse smoke test bị giới hạn tối đa 3 epoch")
    train_sparse(config)


if __name__ == "__main__":
    main()

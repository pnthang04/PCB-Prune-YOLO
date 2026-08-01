"""Analyze or eventually prune a YOLOv8 checkpoint with DepGraph."""

import argparse
import json
from pathlib import Path

import torch

from pcb_prune_yolo.config import load_config, save_config
from pcb_prune_yolo.pruning.dependency_pruner import YOLODepGraphPruner
from pcb_prune_yolo.utils.device import resolve_device


def main() -> None:
    """Build dependency graph in dry-run mode; guard real pruning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/prune/depgraph.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.checkpoint:
        config["checkpoint"] = str(args.checkpoint)
    if args.dry_run is not None:
        config["dry_run"] = args.dry_run
    from ultralytics import YOLO

    device = resolve_device(str(config["device"]))
    yolo = YOLO(str(config["checkpoint"]))
    model = yolo.model.to(device).eval()
    example = torch.randn(1, 3, int(config["imgsz"]), int(config["imgsz"]), device=device)
    pruner = YOLODepGraphPruner(model, example, float(config["pruning_ratio"]), **config)
    before = pruner.analyze_model()
    print(f"Parameters trước pruning: {before['parameters']:,}; MACs: TODO")
    pruner.build_dependency_graph()
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "used_config.yaml")
    (output / "report.json").write_text(json.dumps({"before": before, "after": None, "macs": "TODO"}, indent=2), encoding="utf-8")
    if not config["dry_run"]:
        pruner.prune()
    print("Dry-run hoàn tất; không channel nào bị xóa." if config["dry_run"] else "Pruning hoàn tất.")


if __name__ == "__main__":
    main()


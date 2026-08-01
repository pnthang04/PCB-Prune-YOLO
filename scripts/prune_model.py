"""Dry-run or structurally prune a YOLOv8 checkpoint with DepGraph."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from pcb_prune_yolo.config import load_config, save_config
from pcb_prune_yolo.pruning.dependency_pruner import YOLODepGraphPruner
from pcb_prune_yolo.utils.device import resolve_device


def main() -> None:
    """Trace safely in dry-run mode or create and verify a pruned checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/prune/depgraph.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pruning-ratio", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--round-to", type=int)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("checkpoint", "pruning_ratio", "output", "round_to", "dry_run"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value

    from ultralytics import YOLO

    device = resolve_device(str(config["device"]))
    yolo = YOLO(str(config["checkpoint"]))
    model = yolo.model.to(device).eval()
    example = torch.randn(1, 3, int(config["imgsz"]), int(config["imgsz"]), device=device)
    pruning_ratio = float(config["pruning_ratio"])
    pruner = YOLODepGraphPruner(
        model,
        example,
        pruning_ratio,
        importance=str(config["importance"]),
        iterative_steps=int(config["iterative_steps"]),
        round_to=int(config["round_to"]) if config.get("round_to") else None,
        global_pruning=bool(config["global_pruning"]),
    )

    tag = f"p{round(pruning_ratio * 100):02d}"
    output = Path(config["output"]) / tag
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "used_config.yaml")

    before = pruner.analyze_model()
    forward_before = pruner.validate_forward()
    pruner.build_dependency_graph()
    report: dict[str, object] = {
        "dry_run": bool(config["dry_run"]),
        "pruning_ratio": pruning_ratio,
        "dependency_groups": pruner.group_count,
        "replaced_c2f": pruner.replaced_c2f,
        "protected_layers": pruner.protected_layer_names(),
        "before": before,
        "forward_before": forward_before,
        "after": None,
        "forward_after": None,
        "checkpoint": None,
        "reload_check": None,
    }

    if not config["dry_run"]:
        pruner.prune()
        report["after"] = pruner.analyze_model()
        if report["after"]["parameters"] >= before["parameters"]:  # type: ignore[index]
            raise RuntimeError("Pruning không làm giảm parameter; từ chối lưu checkpoint")
        report["forward_after"] = pruner.validate_forward()
        checkpoint = output / "pruned.pt"
        pruner.save_pruned_model(checkpoint)
        report["checkpoint"] = str(checkpoint)
        verification = subprocess.run(
            [
                sys.executable,
                "scripts/verify_pruned_model.py",
                "--checkpoint",
                str(checkpoint),
                "--device",
                str(device),
                "--imgsz",
                str(config["imgsz"]),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report["reload_check"] = json.loads(verification.stdout)

    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Apply one HALP structural milestone selected by the Stage 2 dry-run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch_pruning as tp
from ultralytics import YOLO

from pcb_prune_yolo.halp.stage2 import audit_backbone_lut
from pcb_prune_yolo.pruning.dependency_pruner import YOLODepGraphPruner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/train/baseline/weights/best.pt"))
    parser.add_argument("--plan", type=Path, default=Path("outputs/halp/stage2/dry_run.json"))
    parser.add_argument("--lut", type=Path, default=Path("outputs/halp/lut/t4_fp16_backbone.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/halp/stage3_m05"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("model_mutated") or plan.get("status") != "dry_run_only":
        raise ValueError("Expected an unmodified Stage 2 dry-run plan")
    device = torch.device(args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    example = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    pruner = YOLODepGraphPruner(model, example, pruning_ratio=plan["target_reduction"], round_to=None)
    before = pruner.analyze_model()
    applied = []
    protected = [module for _, module in pruner.protected_modules]

    for row in plan["groups"]:
        idxs = row.get("pruned_indices", [])
        if not idxs:
            continue
        modules = dict(model.named_modules())
        root_name = row["root_name"]
        root = modules.get(root_name)
        if not isinstance(root, torch.nn.Conv2d):
            raise RuntimeError(f"Missing pruning root {root_name}")
        if root.out_channels != row["channels"]:
            raise RuntimeError(
                f"{root_name} width changed before its turn: {root.out_channels} != {row['channels']}"
            )
        graph = tp.DependencyGraph().build_dependency(model, example_inputs=example)
        group = graph.get_pruning_group(root, tp.prune_conv_out_channels, idxs=idxs)
        if not graph.check_pruning_group(group):
            raise RuntimeError(f"Unsafe DepGraph group for {root_name}")
        protected_ids = {id(module) for module in protected}
        for item in group:
            if id(item.dep.target.module) in protected_ids and item.dep.handler.__name__ == "prune_out_channels":
                raise RuntimeError(f"Plan would change protected output through {root_name}")
        group.prune()
        applied.append({"root_name": root_name, "pruned_channels": len(idxs), "indices": idxs})

    model.eval()
    forward = pruner.validate_forward()
    after = pruner.analyze_model()
    if int(after["parameters"]) >= int(before["parameters"]):
        raise RuntimeError("Structural milestone did not reduce parameters")
    lut = json.loads(args.lut.read_text(encoding="utf-8"))
    audit = audit_backbone_lut(model, lut)
    report = {
        "status": "audit_only" if args.audit_only else "complete",
        "source_checkpoint": str(args.checkpoint),
        "stage2_plan": str(args.plan),
        "target_reduction": plan["target_reduction"],
        "applied_groups": applied,
        "protected_layers": pruner.protected_layer_names(),
        "before": before,
        "after": after,
        "forward": forward,
        "lut_audit": audit,
        "checkpoint": None,
        "reload_check": None,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.audit_only:
        print(json.dumps({"after": after, "lut_audit": audit}, indent=2))
        return
    if not audit["exact"]:
        raise RuntimeError(
            f"Refusing to save: {len(audit['missing_pairs'])} exact post-pruning LUT pairs are missing"
        )
    checkpoint = args.output / "pruned.pt"
    pruner.save_pruned_model(checkpoint)
    verification = subprocess.run(
        [sys.executable, "scripts/verify_pruned_model.py", "--checkpoint", str(checkpoint),
         "--device", args.device, "--imgsz", str(args.imgsz)],
        check=True, capture_output=True, text=True,
    )
    report["checkpoint"] = str(checkpoint)
    report["reload_check"] = json.loads(verification.stdout)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

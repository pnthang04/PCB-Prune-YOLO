"""Run HALP Stage 2 as a non-mutating backbone pruning dry-run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

from pcb_prune_yolo.halp.stage2 import (
    PrefixOption,
    exact_lut_index,
    latency_group_sizes,
    multiple_choice_knapsack,
    original_lut_name,
    taylor_bn_term,
    write_stage2_outputs,
)
from pcb_prune_yolo.pruning.dependency_pruner import YOLODepGraphPruner


def model_index(name: str) -> int:
    return int(name.split(".")[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/train/baseline/weights/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("configs/data/deeppcb.yaml"))
    parser.add_argument("--lut", type=Path, default=Path("outputs/halp/lut/t4_fp16_backbone.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/halp/stage2"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--saliency-batches", type=int, default=8)
    parser.add_argument("--target-reduction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.target_reduction < 1:
        raise ValueError("target-reduction must be in (0, 1)")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    torch.manual_seed(args.seed)
    model = YOLO(str(args.checkpoint)).model.to(device)
    example = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    dep = YOLODepGraphPruner(model, example, pruning_ratio=args.target_reduction, round_to=None)
    graph = dep.build_dependency_graph()
    module_names = {module: name for name, module in model.named_modules()}

    trainer = DetectionTrainer(
        overrides={
            "model": str(args.checkpoint), "data": str(args.data), "imgsz": args.imgsz,
            "batch": args.batch, "device": str(args.device), "workers": args.workers,
            "epochs": 1, "plots": False, "save": False, "val": False, "amp": False,
            "seed": args.seed, "optimizer": "AdamW",
        }
    )
    trainer.model = model
    trainer.set_model_attributes()
    trainer.stride = max(int(model.stride.max()), 32)
    loader = trainer.get_dataloader(trainer.data["train"], args.batch, rank=-1, mode="train")
    totals: dict[str, torch.Tensor] = {}
    model.train().requires_grad_(True)
    used_batches = 0
    for raw in loader:
        model.zero_grad(set_to_none=True)
        batch = trainer.preprocess_batch(raw)
        loss, _ = model(batch)
        loss.sum().backward()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d) and module.weight.grad is not None:
                value = taylor_bn_term(module).cpu()
                totals[name] = totals.get(name, torch.zeros_like(value)) + value
        used_batches += 1
        if used_batches >= args.saliency_batches:
            break
    if used_batches != args.saliency_batches:
        raise RuntimeError(f"Only collected {used_batches} saliency batches")
    totals = {name: value / used_batches for name, value in totals.items()}

    lut = json.loads(args.lut.read_text(encoding="utf-8"))
    index = exact_lut_index(lut)
    steps = latency_group_sizes(lut)
    groups_report = []
    choices: list[list[PrefixOption]] = []
    choice_rows: list[int] = []
    missing: list[dict] = []
    all_groups = list(graph.get_all_groups(ignored_layers=[m for _, m in dep.protected_modules]))
    for group in all_groups:
        root = group[0].dep.target.module
        root_name = module_names.get(root, "")
        if not isinstance(root, torch.nn.Conv2d) or not root_name or model_index(root_name) > 9:
            continue
        # Aggregate every BN channel coupled to the root by DepGraph, matching
        # HALP's group Taylor importance rather than ranking the root BN alone.
        score = torch.zeros(root.out_channels)
        dependency_bn_terms = 0
        for item in group:
            target = item.dep.target.module
            target_name = module_names.get(target, "")
            term = totals.get(target_name)
            if not isinstance(target, torch.nn.BatchNorm2d) or term is None:
                continue
            for root_idx, target_idx in zip(item.root_idxs, item.idxs):
                if root_idx < root.out_channels and target_idx < term.numel():
                    score[root_idx] += term[target_idx]
            dependency_bn_terms += 1
        if dependency_bn_terms == 0:
            score = None
        else:
            score = score.abs()
        lut_name = original_lut_name(root_name)
        group_size = steps.get(lut_name)
        row = {
            "root_name": root_name, "channels": root.out_channels,
            "group_size": group_size, "importance_mean": float(score.mean()) if score is not None else None,
            "selected_keep_channels": None, "selected_latency_ms": None, "status": "eligible",
            "dependency_bn_terms": dependency_bn_terms,
        }
        if score is None or group_size is None:
            row["status"] = "protected_missing_saliency" if score is None else "protected_no_latency_cliff"
            groups_report.append(row)
            continue
        native_key = (lut_name, root.in_channels, root.out_channels)
        if native_key not in index:
            row["status"] = "protected_missing_native_lut"
            missing.append({"layer": lut_name, "cin": root.in_channels, "cout": root.out_channels})
            groups_report.append(row)
            continue
        order = torch.argsort(score, descending=True)
        layer_options = []
        for keep in range(root.out_channels, max(group_size, 1) - 1, -group_size):
            key = (lut_name, root.in_channels, keep)
            if key not in index:
                missing.append({"layer": lut_name, "cin": root.in_channels, "cout": keep})
                continue
            importance = float(score[order[:keep]].sum())
            layer_options.append(PrefixOption(keep, index[key], importance))
        if not layer_options:
            row["status"] = "protected_missing_candidate_lut"
            groups_report.append(row)
            continue
        choices.append(layer_options)
        choice_rows.append(len(groups_report))
        groups_report.append(row)

    dense_latency = sum(max(layer, key=lambda option: option.keep_channels).latency_ms for layer in choices)
    budget = dense_latency * (1 - args.target_reduction)
    selected, kept_importance = multiple_choice_knapsack(choices, budget) if choices else ([], 0.0)
    for row_index, option in zip(choice_rows, selected):
        groups_report[row_index]["selected_keep_channels"] = option.keep_channels
        groups_report[row_index]["selected_latency_ms"] = option.latency_ms
        root_name = groups_report[row_index]["root_name"]
        root = dict(model.named_modules())[root_name]
        group = next(g for g in all_groups if g[0].dep.target.module is root)
        score = torch.zeros(root.out_channels)
        for item in group:
            target = item.dep.target.module
            term = totals.get(module_names.get(target, ""))
            if not isinstance(target, torch.nn.BatchNorm2d) or term is None:
                continue
            for root_idx, target_idx in zip(item.root_idxs, item.idxs):
                if root_idx < root.out_channels and target_idx < term.numel():
                    score[root_idx] += term[target_idx]
        prune_count = root.out_channels - option.keep_channels
        groups_report[row_index]["pruned_indices"] = (
            torch.argsort(score.abs())[:prune_count].tolist() if prune_count else []
        )
    model.eval()
    report = {
        "schema_version": 1,
        "status": "dry_run_only",
        "checkpoint": str(args.checkpoint),
        "input_shape": [1, 3, args.imgsz, args.imgsz],
        "saliency": {"method": "abs(gamma*dL_dgamma + beta*dL_dbeta)", "batches": used_batches},
        "scope": "backbone roots model.0-model.9; Detect/DFL fixed outputs protected",
        "protected_layers": dep.protected_layer_names(),
        "c2f_lut_mapping": "converted cv0/cv1 branches map to the original C2f cv1 operator surface",
        "cost_model": "paper-style current-state root Cout cost; downstream Cin is recomputed after each future milestone",
        "target_reduction": args.target_reduction,
        "eligible_dense_latency_ms": dense_latency,
        "latency_budget_ms": budget,
        "selected_latency_ms": sum(option.latency_ms for option in selected),
        "kept_importance": kept_importance,
        "missing_exact_lut_pairs": missing,
        "groups": groups_report,
        "forward_validation": dep.validate_forward(),
        "model_mutated": False,
    }
    write_stage2_outputs(report, args.output)
    print(json.dumps({k: report[k] for k in ("status", "selected_latency_ms", "latency_budget_ms")}, indent=2))


if __name__ == "__main__":
    main()

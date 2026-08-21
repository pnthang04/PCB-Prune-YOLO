"""Turn learned Hard-Concrete masks into physical DepGraph channel pruning."""

import argparse
from pathlib import Path

import torch

from pcb_prune_yolo.pruning.dependency_pruner import YOLODepGraphPruner
from pcb_prune_yolo.pruning.gated_groups import (
    build_gated_registry,
    extract_gate_state,
    remove_embedded_gates,
)


def _student_model(checkpoint: Path) -> torch.nn.Module:
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = value.get("model") or value.get("ema")
    if model is None:
        raise ValueError("Checkpoint không chứa model/ema")
    return getattr(model, "student_model", model).float().eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--target-sparsity", type=float, default=0.10)
    parser.add_argument("--min-channels", type=int, default=8)
    parser.add_argument("--round-to", type=int, default=None)
    args = parser.parse_args()

    model = _student_model(args.checkpoint)
    gate_state = extract_gate_state(model)
    if not gate_state:
        raise RuntimeError("Checkpoint không chứa Hard-Concrete gate đã học")
    remove_embedded_gates(model)
    example = torch.randn(1, 3, args.imgsz, args.imgsz)
    wrapper = YOLODepGraphPruner(
        model=model,
        example_input=example,
        pruning_ratio=args.target_sparsity,
        iterative_steps=1,
        round_to=None,
        global_pruning=False,
    )
    registry = build_gated_registry(
        wrapper, min_channels=args.min_channels, round_to=args.round_to
    )
    missing = []
    for group in registry.groups:
        logits = gate_state.get(group.gate_owner_name)
        if logits is None or logits.shape != group.gate.log_alpha.shape:
            missing.append(group.gate_owner_name)
            continue
        group.gate.log_alpha.data.copy_(logits)
    if missing:
        raise RuntimeError(f"Không ánh xạ được gate vào DepGraph roots: {missing}")

    selected = registry.materialize()
    wrapper.validate_forward()
    wrapper.save_pruned_model(args.output)
    pruned_channels = sum(len(indices) for indices in selected.values())
    print(f"Saved {args.output} after physically pruning {pruned_channels} channels")


if __name__ == "__main__":
    main()

"""HALP Stage 2: Taylor saliency, latency groups, and a dry-run selector."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def original_lut_name(name: str) -> str:
    """Map an explicit C2f input branch to its pre-conversion LUT operator."""
    parts = name.split(".")
    if len(parts) > 3 and parts[1] in {"2", "4", "6", "8"} and parts[2] == "cv0":
        parts[2] = "cv1"
    return ".".join(parts)


def taylor_bn_term(module: torch.nn.BatchNorm2d) -> torch.Tensor:
    """Return HALP's first-order BN Taylor term for one minibatch."""
    if module.weight.grad is None or module.bias.grad is None:
        raise RuntimeError("BatchNorm gradients are unavailable")
    return (module.weight * module.weight.grad + module.bias * module.bias.grad).abs().detach()


def exact_lut_index(payload: dict[str, Any]) -> dict[tuple[str, int, int], float]:
    """Index successful mean-latency records without interpolation."""
    return {
        (r["layer_name"], int(r["input_channels"]), int(r["output_channels"])): float(
            r["mean_latency_ms"]
        )
        for r in payload["records"]
        if r["status"] == "success"
    }


def latency_group_sizes(payload: dict[str, Any]) -> dict[str, int]:
    """Return only measured, valid staircase group sizes."""
    return {
        row["layer_name"]: int(row["proposed_group_size"])
        for row in payload.get("latency_steps", [])
        if row.get("proposed_group_size")
    }


@dataclass(frozen=True)
class PrefixOption:
    """One legal kept prefix for an ordered channel group."""

    keep_channels: int
    latency_ms: float
    importance: float


def multiple_choice_knapsack(
    options: list[list[PrefixOption]], budget_ms: float, resolution_us: int = 1
) -> tuple[list[PrefixOption], float]:
    """Maximize importance with one prefix per layer under a latency budget.

    Prefix options encode HALP's preceding-group constraint: a later group can
    only exist when every more-important group in that layer is retained.
    """
    if budget_ms <= 0 or resolution_us <= 0:
        raise ValueError("budget and resolution must be positive")
    capacity = int(math.floor(budget_ms * 1000 / resolution_us))
    states: dict[int, tuple[float, list[PrefixOption]]] = {0: (0.0, [])}
    for layer_options in options:
        if not layer_options:
            raise ValueError("every layer needs at least one prefix option")
        updated: dict[int, tuple[float, list[PrefixOption]]] = {}
        for used, (value, chosen) in states.items():
            for option in layer_options:
                weight = int(math.ceil(option.latency_ms * 1000 / resolution_us))
                total = used + weight
                if total > capacity:
                    continue
                candidate = (value + option.importance, chosen + [option])
                if total not in updated or candidate[0] > updated[total][0]:
                    updated[total] = candidate
        states = updated
        if not states:
            raise RuntimeError("No feasible augmented-knapsack state")
    _, (value, selected) = max(states.items(), key=lambda item: item[1][0])
    return selected, value


def write_stage2_outputs(report: dict[str, Any], output: Path) -> None:
    """Write the reproducible dry-run plan as JSON and a flat group CSV."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "dry_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = report.get("groups", [])
    with (output / "groups.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "root_name",
            "channels",
            "group_size",
            "importance_mean",
            "dependency_bn_terms",
            "selected_keep_channels",
            "selected_latency_ms",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

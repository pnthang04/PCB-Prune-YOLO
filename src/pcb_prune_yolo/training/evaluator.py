"""Ultralytics evaluation result extraction."""

from pathlib import Path
from typing import Any


def evaluate(checkpoint: Path, data: Path, split: str, device: str) -> dict[str, float]:
    """Evaluate a checkpoint and return standard detection metrics."""
    from ultralytics import YOLO

    metrics: Any = YOLO(str(checkpoint)).val(data=str(data), split=split, device=device)
    return {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr), "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map)}


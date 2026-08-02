"""Baseline evaluation and per-class metric extraction."""

from pathlib import Path
from typing import Any


def evaluate(checkpoint: Path, data: Path, split: str, device: str) -> dict[str, Any]:
    """Evaluate one checkpoint on val or test and return detection metrics."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint}")
    from ultralytics import YOLO

    model = YOLO(str(checkpoint), task="detect")
    metrics: Any = model.val(
        data=str(data), split=split, device=None if device == "auto" else device
    )
    overall = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }
    per_class = []
    for index, class_id in enumerate(metrics.box.ap_class_index.tolist()):
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": model.names[int(class_id)],
                "precision": float(metrics.box.p[index]),
                "recall": float(metrics.box.r[index]),
                "mAP50": float(metrics.box.ap50[index]),
                "mAP50-95": float(metrics.box.maps[int(class_id)]),
            }
        )
    return {
        "split": split,
        "overall": overall,
        "per_class": per_class,
        "pipeline_speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
        "measurement_scope": "validation_pipeline_includes_preprocess_inference_and_postprocess_nms",
    }

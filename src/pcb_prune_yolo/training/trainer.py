"""Ultralytics training wrapper."""

from pathlib import Path
from typing import Any


def train(config: dict[str, Any]) -> Any:
    """Train a YOLO model using an explicit configuration mapping."""
    from ultralytics import YOLO

    options = dict(config)
    model_name = options.pop("model")
    if "project" in options:
        options["project"] = str(Path(options["project"]).resolve())
    if options.get("device") == "auto":
        options["device"] = None
    model = YOLO(model_name)
    return model.train(**options)


def train_pruned(config: dict[str, Any]) -> Any:
    """Fine-tune an in-memory changed architecture without rebuilding from YAML."""
    from ultralytics import YOLO

    options = dict(config)
    model_path = str(options.pop("model"))
    if "," in str(options.get("device", "")):
        raise ValueError("Fine-tune kiến trúc pruned hiện hỗ trợ một GPU trong mỗi process")
    if "project" in options:
        options["project"] = str(Path(options["project"]).resolve())
    yolo = YOLO(model_path)
    overrides = {
        **options,
        "model": model_path,
        "task": "detect",
        "mode": "train",
    }
    trainer = yolo._smart_load("trainer")(overrides=overrides, _callbacks=yolo.callbacks)
    trainer.model = yolo.model
    trainer.train()
    return getattr(trainer.validator, "metrics", None)

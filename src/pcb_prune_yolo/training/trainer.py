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

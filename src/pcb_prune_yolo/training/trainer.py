"""Ultralytics training wrapper."""

from typing import Any


def train(config: dict[str, Any]) -> Any:
    """Train a YOLO model using an explicit configuration mapping."""
    from ultralytics import YOLO

    options = dict(config)
    model_name = options.pop("model")
    model = YOLO(model_name)
    return model.train(**options)


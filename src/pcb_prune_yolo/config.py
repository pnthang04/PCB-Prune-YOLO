"""YAML configuration helpers."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from *path*."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Config phải là YAML mapping: {config_path}")
    return value


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write a configuration mapping as YAML."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)


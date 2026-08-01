"""Simple model size metrics."""

from pathlib import Path

import torch


def parameter_count(model: torch.nn.Module) -> int:
    """Count all model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def file_size_mb(path: Path) -> float:
    """Return file size in MiB."""
    return path.stat().st_size / (1024 * 1024)


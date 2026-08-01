"""Path validation helpers."""

from pathlib import Path


def existing_path(value: str) -> Path:
    """Return an existing path or raise ValueError for argparse."""
    path = Path(value)
    if not path.exists():
        raise ValueError(f"Không tồn tại: {path}")
    return path


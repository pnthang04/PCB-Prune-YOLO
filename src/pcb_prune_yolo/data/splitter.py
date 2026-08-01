"""Deterministic dataset splitting."""

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def split_items(items: Sequence[T], val_ratio: float, seed: int) -> tuple[list[T], list[T]]:
    """Return deterministic train/validation lists."""
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio phải thuộc [0, 1)")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    val_count = round(len(shuffled) * val_ratio)
    return shuffled[val_count:], shuffled[:val_count]

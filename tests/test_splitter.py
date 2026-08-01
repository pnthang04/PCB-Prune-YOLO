from pathlib import Path

from pcb_prune_yolo.data.splitter import split_items


def test_split_is_reproducible() -> None:
    items = [Path(str(index)) for index in range(10)]
    assert split_items(items, 0.2, 42) == split_items(items, 0.2, 42)


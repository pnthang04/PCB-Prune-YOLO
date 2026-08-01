from pathlib import Path

import pytest

from pcb_prune_yolo.config import load_config
from pcb_prune_yolo.data.validator import validate_label


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 42\n", encoding="utf-8")
    assert load_config(path) == {"seed": 42}


@pytest.mark.parametrize("line", ["6 0.5 0.5 0.1 0.1", "0 1.1 0.5 0.1 0.1"])
def test_invalid_label(tmp_path: Path, line: str) -> None:
    path = tmp_path / "label.txt"
    path.write_text(line, encoding="utf-8")
    assert validate_label(path)


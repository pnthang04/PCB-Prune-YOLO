from pathlib import Path

from pcb_prune_yolo.data.validator import validate_label


def test_valid_label_counts_class(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("5 0.5 0.5 0.2 0.1\n", encoding="utf-8")

    errors, counts = validate_label(label)

    assert errors == []
    assert counts[5] == 1


def test_label_requires_positive_size(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0 0.1\n", encoding="utf-8")

    errors, _ = validate_label(label)

    assert any("width/height" in error for error in errors)

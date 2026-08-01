import pytest

from pcb_prune_yolo.data.converter import bbox_to_yolo


def test_bbox_to_yolo() -> None:
    assert bbox_to_yolo(10, 20, 30, 60, 100, 100) == pytest.approx((0.2, 0.4, 0.2, 0.4))


def test_bbox_outside_image() -> None:
    with pytest.raises(ValueError):
        bbox_to_yolo(-1, 0, 10, 10, 100, 100)


def test_bbox_exceeds_right_edge() -> None:
    with pytest.raises(ValueError):
        bbox_to_yolo(90, 10, 101, 20, 100, 100)

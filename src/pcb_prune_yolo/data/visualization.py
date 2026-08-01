"""Render YOLO annotations without a GUI."""

from pathlib import Path

import cv2

CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]


def render_annotation(image_path: Path, label_path: Path, output_path: Path) -> None:
    """Draw one YOLO label file over its image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")
    height, width = image.shape[:2]
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        class_id, xc, yc, box_width, box_height = raw.split()
        cid = int(class_id)
        xc, yc, box_width, box_height = map(float, (xc, yc, box_width, box_height))
        x1, y1 = int((xc - box_width / 2) * width), int((yc - box_height / 2) * height)
        x2, y2 = int((xc + box_width / 2) * width), int((yc + box_height / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, CLASS_NAMES[cid], (x1, max(15, y1 - 4)), 0, 0.5, (0, 0, 255), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Không ghi được ảnh: {output_path}")

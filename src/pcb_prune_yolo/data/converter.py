"""Convert DeepPCB bounding boxes to YOLO labels."""

from pathlib import Path


def bbox_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    """Convert pixel corner coordinates to normalized YOLO coordinates."""
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("Kích thước ảnh và bounding box phải dương")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError("Bounding box nằm ngoài ảnh")
    values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)
    return values


def convert_annotation(source: Path, destination: Path, image_size: tuple[int, int], class_offset: int = 1) -> dict[int, int]:
    """Convert one comma/space-separated DeepPCB annotation file."""
    width, height = image_size
    lines: list[str] = []
    counts: dict[int, int] = {}
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.replace(",", " ").split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{source}:{line_number}: cần 5 trường")
        x1, y1, x2, y2 = map(float, parts[:4])
        class_id = int(parts[4]) - class_offset
        if class_id not in range(6):
            raise ValueError(f"{source}:{line_number}: class ID không hợp lệ: {class_id}")
        box = bbox_to_yolo(x1, y1, x2, y2, width, height)
        lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in box))
        counts[class_id] = counts.get(class_id, 0) + 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return counts

"""Validate YOLO detection datasets."""

from pathlib import Path


def validate_label(path: Path, class_count: int = 6) -> list[str]:
    """Return validation errors for one YOLO label file."""
    errors: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        prefix = f"{path}:{number}"
        if len(parts) != 5:
            errors.append(f"{prefix}: cần đúng 5 trường")
            continue
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{prefix}: trường không phải số")
            continue
        if class_id not in range(class_count):
            errors.append(f"{prefix}: class ID không hợp lệ")
        if any(value < 0 or value > 1 for value in coordinates):
            errors.append(f"{prefix}: tọa độ ngoài [0, 1]")
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            errors.append(f"{prefix}: width/height phải lớn hơn 0")
    return errors


def validate_dataset(images_dir: Path, labels_dir: Path) -> list[str]:
    """Validate image-label pairing and all label contents."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [path for path in images_dir.rglob("*") if path.suffix.lower() in extensions]
    errors: list[str] = []
    for image in images:
        relative = image.relative_to(images_dir).with_suffix(".txt")
        label = labels_dir / relative
        if not label.is_file():
            errors.append(f"Thiếu label cho ảnh: {image}")
        else:
            errors.extend(validate_label(label))
    return errors


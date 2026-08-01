"""Validate processed YOLO detection datasets."""

import hashlib
import math
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def file_sha256(path: Path) -> str:
    """Hash an image without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_label(path: Path, class_count: int = 6) -> tuple[list[str], Counter[int]]:
    """Return errors and class counts for one YOLO label file."""
    errors: list[str] = []
    counts: Counter[int] = Counter()
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
        else:
            counts[class_id] += 1
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in coordinates):
            errors.append(f"{prefix}: tọa độ ngoài [0, 1]")
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            errors.append(f"{prefix}: width/height phải lớn hơn 0")
    return errors, counts


def validate_split(
    root: Path, split: str
) -> tuple[list[str], int, Counter[int], dict[str, Path]]:
    """Validate image-label pairing and annotations for one split."""
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    image_keys = {str(path.relative_to(images_dir).with_suffix("")) for path in images}
    label_keys = {str(path.relative_to(labels_dir).with_suffix("")) for path in labels_dir.rglob("*.txt")}
    errors = [f"Thiếu label: {key}" for key in sorted(image_keys - label_keys)]
    errors.extend(f"Thiếu ảnh: {key}" for key in sorted(label_keys - image_keys))
    counts: Counter[int] = Counter()
    for key in sorted(image_keys & label_keys):
        label_errors, label_counts = validate_label(labels_dir / f"{key}.txt")
        errors.extend(label_errors)
        counts.update(label_counts)
    hashes = {file_sha256(path): path for path in images}
    if len(hashes) != len(images):
        errors.append(f"Có ảnh trùng nội dung trong split {split}")
    return errors, len(images), counts, hashes


def validate_dataset(root: Path) -> tuple[list[str], dict[str, tuple[int, Counter[int]]]]:
    """Validate all splits and ensure that no image key appears in two splits."""
    errors: list[str] = []
    statistics: dict[str, tuple[int, Counter[int]]] = {}
    hashes_by_split: dict[str, dict[str, Path]] = {}
    for split in ("train", "val", "test"):
        split_errors, image_count, counts, hashes = validate_split(root, split)
        errors.extend(split_errors)
        statistics[split] = (image_count, counts)
        hashes_by_split[split] = hashes
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = hashes_by_split[left].keys() & hashes_by_split[right].keys()
        if overlap:
            samples = [str(hashes_by_split[left][digest]) for digest in sorted(overlap)[:10]]
            errors.append(f"Ảnh trùng giữa {left}/{right}: {', '.join(samples)}")
    return errors, statistics

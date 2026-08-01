"""Prepare the official dataset splits for YOLO without changing raw data."""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import cv2

from pcb_prune_yolo.data.converter import convert_annotation
from pcb_prune_yolo.data.splitter import split_items

CLASS_NAMES = ("open", "short", "mousebite", "spur", "copper", "pin-hole")


def read_manifest(dataset_root: Path, name: str) -> list[tuple[Path, Path]]:
    """Resolve tested images and annotations listed by an official manifest."""
    manifest = dataset_root / f"{name}.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"Thiếu split chính thức: {manifest}")
    pairs: list[tuple[Path, Path]] = []
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split()
        if len(parts) != 2:
            raise ValueError(f"{manifest}:{number}: cần đường dẫn ảnh và annotation")
        listed_image, label = dataset_root / parts[0], dataset_root / parts[1]
        tested_image = listed_image.with_name(f"{listed_image.stem}_test{listed_image.suffix}")
        if not tested_image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Thiếu dữ liệu tại {manifest}:{number}")
        pairs.append((tested_image, label))
    return pairs


def copy_split(
    pairs: list[tuple[Path, Path]], dataset_root: Path, output: Path, split: str
) -> Counter[int]:
    """Copy tested images and convert their labels for one split."""
    counts: Counter[int] = Counter()
    for image_path, source_label in pairs:
        relative = image_path.relative_to(dataset_root)
        target_image = output / "images" / split / relative
        target_label = output / "labels" / split / relative.with_suffix(".txt")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        height, width = image.shape[:2]
        counts.update(convert_annotation(source_label, target_label, (width, height), class_offset=1))
    return counts


def main() -> None:
    """Create train/val from trainval and preserve the official test split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/DeepPCB/PCBData"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/deeppcb"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trainval = read_manifest(args.raw, "trainval")
    official_test = read_manifest(args.raw, "test")
    train, val = split_items(trainval, args.val_ratio, args.seed)
    for split, pairs in (("train", train), ("val", val), ("test", official_test)):
        counts = copy_split(pairs, args.raw, args.output, split)
        print(f"{split}: {len(pairs)} images, {sum(counts.values())} bounding boxes")
        for class_id, class_name in enumerate(CLASS_NAMES):
            print(f"  {class_id} {class_name}: {counts[class_id]}")


if __name__ == "__main__":
    main()

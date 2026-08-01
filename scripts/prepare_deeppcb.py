"""Prepare a DeepPCB-like image/annotation pair collection for YOLO."""

import argparse
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from pcb_prune_yolo.data.converter import convert_annotation
from pcb_prune_yolo.data.splitter import split_items


def main() -> None:
    """Parse CLI options and create a non-destructive YOLO dataset copy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-offset", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in extensions)
    if not images:
        parser.error(f"Không tìm thấy ảnh trong {args.images}")
    train, val = split_items(images, args.val_ratio, args.seed)
    counts: Counter[int] = Counter()
    converted = 0
    for split, paths in (("train", train), ("val", val)):
        for image_path in paths:
            relative = image_path.relative_to(args.images)
            source_label = args.labels / relative.with_suffix(".txt")
            if not source_label.is_file():
                raise FileNotFoundError(f"Thiếu annotation: {source_label}")
            target_image = args.output / "images" / split / relative
            target_label = args.output / "labels" / split / relative.with_suffix(".txt")
            target_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_image)
            with Image.open(image_path) as image:
                counts.update(convert_annotation(source_label, target_label, image.size, args.class_offset))
            converted += 1
    print(f"Ảnh: {converted}; bounding box: {sum(counts.values())}")
    for class_id, name in enumerate(("open", "short", "mousebite", "spur", "copper", "pin-hole")):
        print(f"  {class_id} {name}: {counts[class_id]}")


if __name__ == "__main__":
    main()


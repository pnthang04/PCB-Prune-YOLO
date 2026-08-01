"""Save random annotated dataset previews."""

import argparse
import random
from pathlib import Path

from pcb_prune_yolo.data.visualization import render_annotation


def main() -> None:
    """Render random samples without opening a GUI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_preview"))
    args = parser.parse_args()
    image_dir = args.root / "images" / args.split
    images = [path for path in image_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    for image in random.Random(args.seed).sample(images, min(args.count, len(images))):
        relative = image.relative_to(image_dir)
        label = args.root / "labels" / args.split / relative.with_suffix(".txt")
        render_annotation(image, label, args.output / relative.with_suffix(".jpg"))
    print(f"Đã lưu preview vào {args.output}")


if __name__ == "__main__":
    main()


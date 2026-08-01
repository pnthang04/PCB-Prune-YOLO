"""Save random annotated dataset previews."""

import argparse
import random
from pathlib import Path

from pcb_prune_yolo.data.visualization import render_annotation


def main() -> None:
    """Render random samples without opening a GUI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed/deeppcb"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_preview"))
    args = parser.parse_args()
    image_dir = args.root / "images" / args.split
    images = sorted(
        path
        for path in image_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        parser.error(f"Không tìm thấy ảnh trong {image_dir}")
    for image in random.Random(args.seed).sample(images, min(args.count, len(images))):
        relative = image.relative_to(image_dir)
        label = args.root / "labels" / args.split / relative.with_suffix(".txt")
        render_annotation(image, label, args.output / relative.with_suffix(".jpg"))
    print(f"Saved previews to {args.output}")


if __name__ == "__main__":
    main()

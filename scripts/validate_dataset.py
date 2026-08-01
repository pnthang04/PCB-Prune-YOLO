"""Validate all processed dataset splits and print statistics."""

import argparse
from pathlib import Path

from pcb_prune_yolo.data.validator import validate_dataset

CLASS_NAMES = ("open", "short", "mousebite", "spur", "copper", "pin-hole")


def main() -> None:
    """Print validation errors, image counts and class box counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed/deeppcb"))
    args = parser.parse_args()
    errors, statistics = validate_dataset(args.root)
    for split, (image_count, counts) in statistics.items():
        print(f"{split}: {image_count} images, {sum(counts.values())} bounding boxes")
        for class_id, class_name in enumerate(CLASS_NAMES):
            print(f"  {class_id} {class_name}: {counts[class_id]}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("Dataset is valid; train, val and test do not overlap.")


if __name__ == "__main__":
    main()

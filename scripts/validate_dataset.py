"""Validate a processed YOLO dataset."""

import argparse
from pathlib import Path

from pcb_prune_yolo.data.validator import validate_dataset


def main() -> None:
    """Print errors and exit nonzero for invalid data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()
    errors = validate_dataset(args.root / "images" / args.split, args.root / "labels" / args.split)
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"Dataset hợp lệ: {args.root} ({args.split})")


if __name__ == "__main__":
    main()


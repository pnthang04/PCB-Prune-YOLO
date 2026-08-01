"""JSON and CSV report output."""

import csv
import json
from pathlib import Path
from typing import Any


def write_report(values: dict[str, Any], output_dir: Path, stem: str) -> None:
    """Write a flat report to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=values.keys())
        writer.writeheader()
        writer.writerow(values)


"""JSON and CSV report output."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_report(values: dict[str, Any], output_dir: Path, stem: str) -> None:
    """Write a report to JSON and a tabular CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.json").write_text(
        json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if "overall" in values:
        rows = [{"scope": "overall", **values["overall"]}]
        rows.extend({"scope": "class", **row} for row in values.get("per_class", []))
    else:
        rows = [values]
    pd.DataFrame(rows).to_csv(output_dir / f"{stem}.csv", index=False)

"""Profile only exact post-pruning LUT pairs reported missing by Stage 3."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from pcb_prune_yolo.halp.lut import TensorRTConvProfiler, validate_lut, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lut", type=Path, default=Path("outputs/halp/lut/t4_fp16_backbone.json"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/halp/lut_stage3"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--workspace-gib", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    payload = json.loads(args.lut.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))["lut_audit"]
    layer_metadata = {row["layer_name"]: row for row in payload["profiled_layers"]}
    requests = {
        (row["lut_layer"], int(row["input_channels"]), int(row["output_channels"]))
        for row in audit["missing_pairs"]
    }
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    profiler = TensorRTConvProfiler(device, args.workspace_gib)
    records = copy.deepcopy(payload["records"])
    for number, (name, cin, cout) in enumerate(sorted(requests), start=1):
        layer = layer_metadata[name]
        print(f"[{number}/{len(requests)}] {name}: Cin={cin}, Cout={cout}", flush=True)
        result = None
        for attempt in range(1, args.max_attempts + 1):
            candidate = profiler.profile(layer, cin, cout, args.warmup, args.iterations, repeat=True)
            error = candidate.get("reproducibility_relative_error", 0.0)
            if error <= 0.20:
                result = candidate
                break
            print(f"  retry {attempt}/{args.max_attempts}: relative error={error:.2%}", flush=True)
        if result is None:
            raise RuntimeError(f"Unstable TensorRT measurement for {name}, Cin={cin}, Cout={cout}")
        records.append(
            {
                "layer_name": name,
                "layer_type": "Conv2d",
                "input_channels": cin,
                "output_channels": cout,
                "height": int(layer["height"]),
                "width": int(layer["width"]),
                "output_height": int(layer["output_height"]),
                "output_width": int(layer["output_width"]),
                "kernel": int(layer["kernel"]),
                "stride": int(layer["stride"]),
                "groups": int(layer["groups"]),
                "precision": "fp16",
                "warmup_iterations": args.warmup,
                "benchmark_iterations": args.iterations,
                "status": "success",
                "error": None,
                "native_input_channels": int(layer["native_input_channels"]),
                "native_output_channels": int(layer["native_output_channels"]),
                "stage3_targeted_refinement": True,
                **result,
            }
        )
    refined = copy.deepcopy(payload)
    refined["schema_version"] = 2
    refined["records"] = records
    refined["stage3_refinement"] = {
        "source_lut": str(args.lut),
        "source_audit": str(args.audit),
        "requested_pairs": len(requests),
        "warmup_iterations": args.warmup,
        "benchmark_iterations": args.iterations,
    }
    validate_lut(refined)
    args.output.mkdir(parents=True)
    write_json(refined, args.output / "t4_fp16_backbone_stage3.json")
    write_csv(records, args.output / "t4_fp16_backbone_stage3.csv")


if __name__ == "__main__":
    main()

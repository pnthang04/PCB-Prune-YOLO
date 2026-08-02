"""Profile a two-dimensional TensorRT FP16 convolution LUT for HALP Stage 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pcb_prune_yolo.halp.lut import (
    TensorRTConvProfiler,
    analyze_latency_steps,
    discover_backbone_convolutions,
    environment_metadata,
    measurement_pairs,
    signature,
    validate_lut,
    write_csv,
    write_json,
)


def main() -> None:
    """Discover the YOLO backbone, profile unique operators, and analyze cliffs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/train/baseline/weights/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/halp/lut"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--workspace-gib", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-signatures", type=int, help="Smoke-test only; limits unique operator signatures")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("HALP TensorRT LUT requires CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    lut_path = args.output / "t4_fp16_backbone.json"
    csv_path = args.output / "t4_fp16_backbone.csv"
    environment_path = args.output / "environment.json"
    steps_path = args.output / "latency_steps.json"
    steps_csv_path = args.output / "latency_steps.csv"
    existing = None
    if lut_path.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite {lut_path}; pass --resume")
        existing = json.loads(lut_path.read_text(encoding="utf-8"))
    elif any(path.exists() for path in (csv_path, environment_path, steps_path, steps_csv_path)):
        raise FileExistsError("Refusing to overwrite partial HALP LUT outputs")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    layers, excluded = discover_backbone_convolutions(args.checkpoint, args.imgsz, device)
    environment = environment_metadata(device)
    environment.update(
        {
            "checkpoint": str(args.checkpoint),
            "model_input_shape": [1, 3, args.imgsz, args.imgsz],
            "warmup_iterations": args.warmup,
            "benchmark_iterations": args.iterations,
            "sampling": "dense native-Cin Cout sweep by 8 plus sparse 4x4 Cin-Cout calibration grid",
            "halp_commit": "dfee297d55d1638b968359e7ffff878be846ec02",
        }
    )
    write_json(environment, environment_path)
    completed = {
        (
            record["layer_name"],
            int(record["input_channels"]),
            int(record["output_channels"]),
        )
        for record in (existing or {}).get("records", [])
    }
    records = list((existing or {}).get("records", []))
    signature_layers: dict[tuple[int, ...], list[dict]] = {}
    for layer in layers:
        signature_layers.setdefault(signature(layer), []).append(layer)
    selected = list(signature_layers.items())
    if args.max_signatures is not None:
        selected = selected[: args.max_signatures]
        selected_names = {layer["layer_name"] for _, group in selected for layer in group}
        layers = [layer for layer in layers if layer["layer_name"] in selected_names]
    profiler = TensorRTConvProfiler(device, args.workspace_gib)
    for signature_index, (_, aliases) in enumerate(selected, start=1):
        representative = aliases[0]
        pairs = measurement_pairs(
            int(representative["native_input_channels"]),
            int(representative["native_output_channels"]),
            int(representative["groups"]),
        )
        print(
            f"[{signature_index}/{len(selected)}] {representative['layer_name']}: "
            f"{len(pairs)} Cin-Cout configurations, {len(aliases)} layer alias(es)",
            flush=True,
        )
        for cin, cout in pairs:
            missing_aliases = [
                layer
                for layer in aliases
                if (layer["layer_name"], cin, cout) not in completed
            ]
            if not missing_aliases:
                continue
            base = {
                "layer_type": "Conv2d",
                "input_channels": cin,
                "output_channels": cout,
                "height": int(representative["height"]),
                "width": int(representative["width"]),
                "kernel": int(representative["kernel"]),
                "stride": int(representative["stride"]),
                "groups": int(representative["groups"]),
                "precision": "fp16",
                "warmup_iterations": args.warmup,
                "benchmark_iterations": args.iterations,
                "status": "success",
                "error": None,
            }
            try:
                base.update(
                    profiler.profile(
                        representative,
                        cin,
                        cout,
                        args.warmup,
                        args.iterations,
                        repeat=(
                            cin == representative["native_input_channels"]
                            and cout == representative["native_output_channels"]
                        ),
                    )
                )
            except Exception as error:  # retain failed configurations in the LUT
                base.update(
                    {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "mean_latency_ms": None,
                        "median_latency_ms": None,
                        "p95_latency_ms": None,
                    }
                )
            for layer in missing_aliases:
                record = dict(base)
                record["layer_name"] = layer["layer_name"]
                record["native_input_channels"] = layer["native_input_channels"]
                record["native_output_channels"] = layer["native_output_channels"]
                records.append(record)
                completed.add((layer["layer_name"], cin, cout))
        partial = {
            "schema_version": 1,
            "environment": environment,
            "profiled_layers": layers,
            "excluded_layers": excluded,
            "records": records,
            "latency_steps": [],
        }
        write_json(partial, lut_path)
    payload = {
        "schema_version": 1,
        "environment": environment,
        "profiled_layers": layers,
        "excluded_layers": excluded,
        "records": records,
    }
    payload["latency_steps"] = analyze_latency_steps(payload)
    validate_lut(payload)
    write_json(payload, lut_path)
    write_csv(records, csv_path)
    write_json(payload["latency_steps"], steps_path)
    flat_steps = [
        {key: value for key, value in step.items() if key != "transitions"}
        for step in payload["latency_steps"]
    ]
    write_csv(flat_steps, steps_csv_path)
    print(
        json.dumps(
            {
                "profiled_layers": len(layers),
                "unique_signatures": len(selected),
                "records": len(records),
                "successful_records": sum(record["status"] == "success" for record in records),
                "failed_records": sum(record["status"] != "success" for record in records),
            }
        )
    )


if __name__ == "__main__":
    main()

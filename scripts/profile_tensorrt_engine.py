"""Profile a metadata-wrapped static TensorRT engine without rebuilding it."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import tensorrt as trt

from benchmark_tensorrt_runtime import load_engine


def _find_trtexec() -> Path:
    candidates = [
        shutil.which("trtexec"),
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    for candidate in Path("/tmp").glob("*/extracted/usr/bin/trtexec"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Không tìm thấy trtexec")


def _layer_counts(layer_info: object) -> dict[str, int]:
    layers = layer_info.get("Layers", []) if isinstance(layer_info, dict) else layer_info
    counts: Counter[str] = Counter()
    for layer in layers if isinstance(layers, list) else []:
        if isinstance(layer, dict):
            layer_type = str(layer.get("LayerType", "Unknown"))
            name = str(layer.get("Name", "")).lower()
        else:
            layer_type = ""
            name = str(layer).lower()
        counts["layers"] += 1
        if "convolution" in layer_type.lower() or "/conv" in name:
            counts["convolution"] += 1
        if "reformat" in layer_type.lower() or "reformat" in name:
            counts["reformat"] += 1
        if "pointwise" in layer_type.lower() or "pointwise" in name or "pwn(" in name:
            counts["pointwise"] += 1
        if "concatenation" in layer_type.lower() or "concat" in name:
            counts["concat"] += 1
        if "copy" in layer_type.lower() or "copy" in name:
            counts["copy"] += 1
    return dict(counts)


def _kernel_launches(stats_csv: Path) -> int | None:
    if not stats_csv.is_file():
        return None
    lines = [line for line in stats_csv.read_text(encoding="utf-8").splitlines() if line]
    try:
        header_index = next(i for i, line in enumerate(lines) if "Instances" in line)
    except StopIteration:
        return None
    rows = csv.DictReader(lines[header_index:])
    total = 0
    for row in rows:
        value = row.get("Instances")
        if value:
            total += int(value.replace(",", ""))
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--nsys", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Từ chối ghi đè {args.output}")
    args.output.mkdir(parents=True)

    metadata, plan = load_engine(args.engine)
    raw_plan = args.output / "model.plan"
    raw_plan.write_bytes(plan)
    trtexec = _find_trtexec()
    trt_libs = Path(trt.__file__).resolve().parent.parent / "tensorrt_libs"
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        part
        for part in (str(trt_libs), environment.get("LD_LIBRARY_PATH", ""))
        if part
    )
    profile = args.output / "profile.json"
    times = args.output / "times.json"
    layers = args.output / "layers.json"
    command = [
        str(trtexec),
        f"--loadEngine={raw_plan}",
        "--warmUp=0",
        "--duration=0",
        f"--iterations={args.iterations}",
        "--avgRuns=1",
        "--noDataTransfers",
        "--dumpProfile",
        "--separateProfileRun",
        "--profilingVerbosity=detailed",
        f"--exportProfile={profile}",
        f"--exportTimes={times}",
        f"--exportLayerInfo={layers}",
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, env=environment
    )
    (args.output / "trtexec.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )

    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("TensorRT engine deserialization failed")
    inspector = engine.create_engine_inspector()
    inspector_json = json.loads(
        inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    )
    (args.output / "inspector.json").write_text(
        json.dumps(inspector_json, indent=2), encoding="utf-8"
    )

    stats_csv = args.output / "nsys_cuda_gpu_kern_sum.csv"
    if args.nsys:
        nsys = shutil.which("nsys")
        if nsys is None:
            raise FileNotFoundError("Không tìm thấy nsys")
        trace_prefix = args.output / "kernel_trace"
        trace_command = [
            nsys,
            "profile",
            "--force-overwrite=true",
            "--trace=cuda",
            f"--output={trace_prefix}",
            str(trtexec),
            f"--loadEngine={raw_plan}",
            "--warmUp=0",
            "--duration=0",
            f"--iterations={args.iterations}",
            "--avgRuns=1",
            "--noDataTransfers",
        ]
        subprocess.run(
            trace_command, check=True, capture_output=True, text=True, env=environment
        )
        stats = subprocess.run(
            [
                nsys,
                "stats",
                "--report=cuda_gpu_kern_sum",
                "--format=csv",
                "--force-export=true",
                str(trace_prefix.with_suffix(".nsys-rep")),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stats_csv.write_text(stats.stdout, encoding="utf-8")

    layer_info = json.loads(layers.read_text(encoding="utf-8"))
    result = {
        "engine": str(args.engine),
        "input_shape": metadata.get("imgsz", [640, 640]),
        "iterations": args.iterations,
        "cuda_graph": False,
        "data_transfers": False,
        "layer_counts": _layer_counts(layer_info),
        "kernel_launches": _kernel_launches(stats_csv),
        "kernel_launches_per_iteration": (
            _kernel_launches(stats_csv) / args.iterations
            if _kernel_launches(stats_csv) is not None
            else None
        ),
        "tensorrt": trt.__version__,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    raw_plan.unlink()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

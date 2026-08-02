"""TensorRT convolution LUT discovery, validation, and staircase analysis."""

from __future__ import annotations

import csv
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REQUIRED_RECORD_FIELDS = {
    "layer_name",
    "layer_type",
    "input_channels",
    "output_channels",
    "height",
    "width",
    "kernel",
    "stride",
    "groups",
    "precision",
    "mean_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "warmup_iterations",
    "benchmark_iterations",
    "status",
}


def load_lut(path: Path) -> dict[str, Any]:
    """Load a JSON LUT and reject a non-object root."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("HALP LUT root must be a JSON object")
    return payload


def validate_lut(payload: dict[str, Any], reproducibility_tolerance: float = 0.20) -> None:
    """Validate schema, channel legality, timing, group sizes, and repeatability."""
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("HALP LUT must contain non-empty records")
    for index, record in enumerate(records):
        missing = REQUIRED_RECORD_FIELDS - set(record)
        if missing:
            raise ValueError(f"record {index} missing fields: {sorted(missing)}")
        cin = int(record["input_channels"])
        cout = int(record["output_channels"])
        groups = int(record["groups"])
        if cin <= 0 or cout <= 0 or groups <= 0 or cin % groups:
            raise ValueError(f"record {index} has invalid channels/groups")
        if record["status"] == "success":
            for field in ("mean_latency_ms", "median_latency_ms", "p95_latency_ms"):
                if not np.isfinite(record[field]) or float(record[field]) <= 0:
                    raise ValueError(f"record {index} has invalid {field}")
            error = record.get("reproducibility_relative_error")
            if error is not None and float(error) > reproducibility_tolerance:
                raise ValueError(f"record {index} exceeds reproducibility tolerance")
    for step in payload.get("latency_steps", []):
        group_size = step.get("proposed_group_size")
        if group_size is not None and not 0 < int(group_size) <= int(step["native_output_channels"]):
            raise ValueError("latency group size exceeds native channel count")


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def latency_summary(values: list[float]) -> dict[str, float]:
    """Summarize positive latency observations."""
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("latency samples must be finite and positive")
    return {
        "mean_latency_ms": float(statistics.fmean(values)),
        "median_latency_ms": float(statistics.median(values)),
        "p95_latency_ms": _percentile(values, 95),
    }


def candidate_channels(channels: int) -> list[int]:
    """Return a sparse, aligned 2D calibration grid including the native width."""
    if channels <= 0:
        raise ValueError("channels must be positive")
    if channels < 8:
        return [channels]
    values = {channels}
    for ratio in (0.25, 0.50, 0.75):
        values.add(max(8, min(channels, int(channels * ratio) // 8 * 8)))
    return sorted(values)


def measurement_pairs(cin: int, cout: int, groups: int) -> list[tuple[int, int]]:
    """Build dense-Cin Cout sweep plus an explicit sparse two-dimensional grid."""
    if groups <= 0:
        raise ValueError("groups must be positive")
    output_sweep = range(8, cout + 1, 8) if cout >= 8 and cout % 8 == 0 else [cout]
    pairs = {(cin, candidate) for candidate in output_sweep}
    pairs.update(
        (candidate_in, candidate_out)
        for candidate_in in candidate_channels(cin)
        for candidate_out in candidate_channels(cout)
    )
    return sorted(
        (candidate_in, candidate_out)
        for candidate_in, candidate_out in pairs
        if candidate_in > 0
        and candidate_out > 0
        and candidate_in <= cin
        and candidate_out <= cout
        and candidate_in % groups == 0
    )


def discover_backbone_convolutions(
    checkpoint: Path, imgsz: int, device: torch.device, backbone_end: int = 9
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Trace leaf Conv2d input/output shapes and split backbone from protected head."""
    from ultralytics import YOLO

    model = YOLO(str(checkpoint)).model.to(device).eval()
    discovered: list[dict[str, Any]] = []
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue

        def capture(
            current: torch.nn.Conv2d,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            layer_name: str = name,
        ) -> None:
            model_index = int(layer_name.split(".")[1])
            discovered.append(
                {
                    "layer_name": layer_name,
                    "layer_type": "Conv2d",
                    "model_index": model_index,
                    "native_input_channels": current.in_channels,
                    "native_output_channels": current.out_channels,
                    "height": int(inputs[0].shape[-2]),
                    "width": int(inputs[0].shape[-1]),
                    "output_height": int(output.shape[-2]),
                    "output_width": int(output.shape[-1]),
                    "kernel": int(current.kernel_size[0]),
                    "stride": int(current.stride[0]),
                    "padding": int(current.padding[0]),
                    "dilation": int(current.dilation[0]),
                    "groups": int(current.groups),
                }
            )

        hooks.append(module.register_forward_hook(capture))
    with torch.inference_mode():
        model(torch.randn(1, 3, imgsz, imgsz, device=device))
    for hook in hooks:
        hook.remove()
    included = [layer for layer in discovered if layer["model_index"] <= backbone_end]
    excluded = [
        {
            "layer_name": layer["layer_name"],
            "reason": "model.10+ neck/head excluded from backbone-only HALP Stage 1",
        }
        for layer in discovered
        if layer["model_index"] > backbone_end
    ]
    return included, excluded


def signature(layer: dict[str, Any]) -> tuple[int, ...]:
    """Return the operator signature that determines a reusable measurement surface."""
    return tuple(
        int(layer[key])
        for key in (
            "native_input_channels",
            "native_output_channels",
            "height",
            "width",
            "kernel",
            "stride",
            "padding",
            "dilation",
            "groups",
        )
    )


class TensorRTConvProfiler:
    """Build and benchmark static FP16 TensorRT convolution engines."""

    def __init__(self, device: torch.device, workspace_gib: float = 1.0) -> None:
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.builder = trt.Builder(self.logger)
        self.device = device
        self.workspace_bytes = int(workspace_gib * 1024**3)
        self.timing_cache: bytes | None = None

    def profile(
        self,
        layer: dict[str, Any],
        cin: int,
        cout: int,
        warmup: int,
        iterations: int,
        repeat: bool,
    ) -> dict[str, Any]:
        """Build outside timing, then benchmark with reused buffers and synchronization."""
        trt = self.trt
        started = time.perf_counter()
        network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        input_tensor = network.add_input(
            "input", trt.float16, (1, cin, int(layer["height"]), int(layer["width"]))
        )
        weight = np.zeros(
            (cout, cin // int(layer["groups"]), int(layer["kernel"]), int(layer["kernel"])),
            dtype=np.float32,
        )
        convolution = network.add_convolution_nd(
            input_tensor, cout, (int(layer["kernel"]), int(layer["kernel"])), weight
        )
        convolution.name = "profiled_conv"
        convolution.stride_nd = (int(layer["stride"]), int(layer["stride"]))
        convolution.padding_nd = (int(layer["padding"]), int(layer["padding"]))
        convolution.dilation_nd = (int(layer["dilation"]), int(layer["dilation"]))
        convolution.num_groups = int(layer["groups"])
        convolution.precision = trt.float16
        convolution.set_output_type(0, trt.float16)
        output_tensor = convolution.get_output(0)
        output_tensor.name = "output"
        network.mark_output(output_tensor)
        config = self.builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_bytes)
        cache = config.create_timing_cache(self.timing_cache or b"")
        config.set_timing_cache(cache, False)
        serialized = self.builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT returned no serialized engine")
        self.timing_cache = bytes(config.get_timing_cache().serialize())
        engine = trt.Runtime(self.logger).deserialize_cuda_engine(serialized)
        context = engine.create_execution_context()
        output_shape = tuple(context.get_tensor_shape("output"))
        image = torch.empty(
            (1, cin, int(layer["height"]), int(layer["width"])),
            dtype=torch.float16,
            device=self.device,
        ).normal_()
        output = torch.empty(output_shape, dtype=torch.float16, device=self.device)
        context.set_tensor_address("input", image.data_ptr())
        context.set_tensor_address("output", output.data_ptr())
        stream = torch.cuda.Stream(device=self.device)

        def measure(count: int) -> list[float]:
            values = []
            for _ in range(count):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT execute_async_v3 failed")
                end.record(stream)
                end.synchronize()
                values.append(float(start.elapsed_time(end)))
            return values

        for _ in range(warmup):
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT warm-up execution failed")
        stream.synchronize()
        values = measure(iterations)
        summary = latency_summary(values)
        if repeat:
            repeated = latency_summary(measure(iterations))
            summary["repeat_median_latency_ms"] = repeated["median_latency_ms"]
            summary["reproducibility_relative_error"] = abs(
                summary["median_latency_ms"] - repeated["median_latency_ms"]
            ) / summary["median_latency_ms"]
        inspector = json.loads(
            engine.create_engine_inspector().get_engine_information(
                trt.LayerInformationFormat.JSON
            )
        )
        tactic = next(
            (item for item in inspector.get("Layers", []) if item.get("Name") == "profiled_conv"),
            {},
        )
        summary.update(
            {
                "profile_wall_time_seconds": time.perf_counter() - started,
                "output_height": int(output_shape[-2]),
                "output_width": int(output_shape[-1]),
                "tactic_name": tactic.get("TacticName"),
                "tactic_value": tactic.get("TacticValue"),
                "tensorrt_layer_type": tactic.get("LayerType"),
            }
        )
        return summary


def analyze_latency_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect plateaus/cliffs on the dense-input Cout sweep for each layer."""
    results = []
    layers = {layer["layer_name"]: layer for layer in payload["profiled_layers"]}
    for layer_name, layer in layers.items():
        points = [
            record
            for record in payload["records"]
            if record["layer_name"] == layer_name
            and record["status"] == "success"
            and record["input_channels"] == layer["native_input_channels"]
        ]
        points.sort(key=lambda item: item["output_channels"], reverse=True)
        transitions = []
        positive_drops = []
        for high, low in zip(points, points[1:]):
            drop = float(high["median_latency_ms"] - low["median_latency_ms"])
            if drop > 0:
                positive_drops.append(drop)
            transitions.append(
                {
                    "from_output_channels": int(high["output_channels"]),
                    "to_output_channels": int(low["output_channels"]),
                    "channel_reduction": int(high["output_channels"] - low["output_channels"]),
                    "latency_drop_ms": drop,
                }
            )
        native_latency = float(points[0]["median_latency_ms"]) if points else 0.0
        noise_floor = max(0.0005, native_latency * 0.01)
        robust_drop = statistics.median(positive_drops) if positive_drops else 0.0
        cliff_threshold = max(noise_floor * 2.0, robust_drop * 1.5)
        cliffs = []
        plateaus = []
        for transition in transitions:
            if transition["latency_drop_ms"] >= cliff_threshold:
                transition["classification"] = "cliff"
                cliffs.append(transition)
            elif abs(transition["latency_drop_ms"]) <= noise_floor:
                transition["classification"] = "plateau"
                plateaus.append(transition)
            else:
                transition["classification"] = "slope_or_noise"
        cliff_positions = [item["to_output_channels"] for item in cliffs]
        cliff_widths = [
            abs(first - second)
            for first, second in zip(cliff_positions, cliff_positions[1:])
            if first != second
        ]
        proposed = None
        if cliff_widths:
            center = statistics.median(cliff_widths)
            proposed = min(cliff_widths, key=lambda value: (abs(value - center), value))
        if proposed is not None:
            proposed = min(proposed, int(layer["native_output_channels"]))
        results.append(
            {
                "layer_name": layer_name,
                "native_input_channels": int(layer["native_input_channels"]),
                "native_output_channels": int(layer["native_output_channels"]),
                "noise_floor_ms": noise_floor,
                "cliff_threshold_ms": cliff_threshold,
                "cliff_count": len(cliffs),
                "plateau_count": len(plateaus),
                "proposed_group_size": proposed,
                "status": "resolved" if proposed is not None else "insufficient_cliffs",
                "transitions": transitions,
            }
        )
    return results


def environment_metadata(device: torch.device) -> dict[str, Any]:
    """Capture the runtime that makes a TensorRT LUT device-specific."""
    import tensorrt
    import ultralytics

    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": properties.name,
        "gpu_compute_capability": list(properties.major_minor) if hasattr(properties, "major_minor") else [properties.major, properties.minor],
        "gpu_total_memory_mib": properties.total_memory / 1024**2,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tensorrt_version": tensorrt.__version__,
        "ultralytics_version": ultralytics.__version__,
        "precision": "fp16",
        "batch_size": 1,
        "dynamic_shapes": False,
    }


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Write heterogeneous dictionaries to CSV using their union of keys."""
    fields = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_json(payload: Any, path: Path) -> None:
    """Write stable, human-readable JSON."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

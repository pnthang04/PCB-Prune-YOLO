"""Benchmark a static TensorRT engine with explicit reusable runtime resources."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression


TORCH_DTYPES = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


def load_engine(path: Path) -> tuple[dict, bytes]:
    """Read an Ultralytics length-prefixed engine without modifying it."""
    with path.open("rb") as handle:
        size_data = handle.read(4)
        if len(size_data) != 4:
            raise ValueError("Invalid Ultralytics engine header")
        metadata_size = struct.unpack("<I", size_data)[0]
        metadata = json.loads(handle.read(metadata_size))
        plan = handle.read()
    return metadata, plan


def summarize(samples: list[float]) -> dict[str, float]:
    mean = statistics.fmean(samples)
    return {
        "mean_latency_ms": mean,
        "median_latency_ms": statistics.median(samples),
        "p95_latency_ms": float(np.percentile(samples, 95)),
        "fps": 1000 / mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("data/processed/deeppcb/images/val"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    metadata, plan = load_engine(args.engine)
    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("TensorRT engine deserialization failed")
    context = engine.create_execution_context()
    stream = torch.cuda.Stream(device=device)
    tensors: dict[str, torch.Tensor] = {}
    input_name = ""
    output_names = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = TORCH_DTYPES[engine.get_tensor_dtype(name)]
        tensors[name] = torch.empty(shape, dtype=dtype, device=device)
        context.set_tensor_address(name, tensors[name].data_ptr())
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_names.append(name)
    if not input_name or len(output_names) != 1:
        raise RuntimeError("Expected one static input and one output")
    input_tensor = tensors[input_name]
    output_tensor = tensors[output_names[0]]
    host_input = torch.empty(input_tensor.shape, dtype=input_tensor.dtype, pin_memory=True)

    graph = None
    with torch.cuda.stream(stream):
        input_tensor.zero_()
        for _ in range(3):
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT warm-up enqueue failed")
        stream.synchronize()
        if args.cuda_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT graph capture enqueue failed")
    stream.synchronize()

    def inference(include_h2d: bool) -> None:
        with torch.cuda.stream(stream):
            if include_h2d:
                input_tensor.copy_(host_input, non_blocking=True)
            if graph is not None:
                graph.replay()
            else:
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT enqueue failed")

    for _ in range(args.warmup):
        inference(include_h2d=False)
    stream.synchronize()
    forward_samples = []
    for _ in range(args.iterations):
        stream.synchronize()
        started = time.perf_counter()
        inference(include_h2d=False)
        stream.synchronize()
        forward_samples.append((time.perf_counter() - started) * 1000)

    paths = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in {".jpg", ".png"})
    images = [cv2.imread(str(path)) for path in paths]
    if not images or any(image is None for image in images):
        raise RuntimeError("Unable to preload validation images")
    letterbox = LetterBox(new_shape=input_tensor.shape[-2:], auto=False, stride=32)

    def preprocess(image: np.ndarray) -> None:
        resized = letterbox(image=image)
        array = np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1))[None]
        source = torch.from_numpy(array)
        host_input.copy_(source)
        host_input.div_(255)

    def e2e(image: np.ndarray) -> None:
        preprocess(image)
        inference(include_h2d=True)
        with torch.cuda.stream(stream):
            non_max_suppression(output_tensor, args.conf, args.iou, nc=len(metadata["names"]))

    for index in range(args.warmup):
        e2e(images[index % len(images)])
    stream.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    e2e_samples = []
    for index in range(args.iterations):
        stream.synchronize()
        started = time.perf_counter()
        e2e(images[index % len(images)])
        stream.synchronize()
        e2e_samples.append((time.perf_counter() - started) * 1000)

    report = {
        "engine": str(args.engine),
        "cuda_graph": args.cuda_graph,
        "static_shape": list(input_tensor.shape),
        "input_dtype": str(input_tensor.dtype),
        "output_shape": list(output_tensor.shape),
        "context_reused": True,
        "stream_reused": True,
        "gpu_buffers_reused": True,
        "pinned_host_buffer": True,
        "h2d": "non_blocking copy on reused CUDA stream",
        "warmup_iterations": args.warmup,
        "benchmark_iterations": args.iterations,
        "forward_scope": "TensorRT enqueue only; CUDA Graph captures inference launches only",
        "e2e_scope": "CPU letterbox/normalize + pinned copy + async H2D + TensorRT + NMS; disk excluded",
        "forward": summarize(forward_samples),
        "e2e": summarize(e2e_samples),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "engine_size_mb": args.engine.stat().st_size / 1024**2,
        "gpu": torch.cuda.get_device_name(device),
        "tensorrt": trt.__version__,
    }
    args.output.mkdir(parents=True)
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    flat = {"engine": report["engine"], "cuda_graph": args.cuda_graph,
            **{f"forward_{k}": v for k, v in report["forward"].items()},
            **{f"e2e_{k}": v for k, v in report["e2e"].items()},
            "peak_gpu_memory_mb": report["peak_gpu_memory_mb"],
            "engine_size_mb": report["engine_size_mb"]}
    with (args.output / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat.keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerow(flat)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

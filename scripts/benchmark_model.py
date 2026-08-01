"""Benchmark a YOLO model on CPU or CUDA."""

import argparse
from pathlib import Path

import torch

from pcb_prune_yolo.benchmarking.complexity import file_size_mb, parameter_count
from pcb_prune_yolo.benchmarking.latency import benchmark_latency
from pcb_prune_yolo.benchmarking.report import write_report
from pcb_prune_yolo.config import load_config
from pcb_prune_yolo.utils.device import resolve_device


def main() -> None:
    """Load once, benchmark batch size one, and write JSON/CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark/default.yaml"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    config.update({key: str(value) for key, value in vars(args).items() if key != "config" and value is not None})
    from ultralytics import YOLO

    checkpoint = Path(config["model"])
    device = resolve_device(config["device"])
    module = YOLO(str(checkpoint)).model.to(device).eval()
    image = torch.randn(1, 3, int(config["imgsz"]), int(config["imgsz"]), device=device)
    with torch.inference_mode():
        values = benchmark_latency(lambda: module(image), int(config["warmup_iterations"]), int(config["benchmark_iterations"]), device)
    values.update({"parameters": parameter_count(module), "model_size_mb": file_size_mb(checkpoint), "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None})
    write_report(values, Path(config["output"]), "benchmark")


if __name__ == "__main__":
    main()


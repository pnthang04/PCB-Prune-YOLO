"""Benchmark TensorRT preprocessing, H2D, inference, and NMS without disk I/O."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("data/processed/deeppcb/images/val"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    paths = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in {".jpg", ".png"})
    images = [cv2.imread(str(path)) for path in paths]
    if not images or any(image is None for image in images):
        raise RuntimeError("Unable to preload validation images")

    yolo = YOLO(str(args.engine), task="detect")
    yolo.predict(images[0], device=args.device, imgsz=args.imgsz, verbose=False)
    predictor = yolo.predictor

    def pipeline(image: np.ndarray) -> None:
        tensor = predictor.preprocess([image])
        predictions = predictor.inference(tensor)
        predictor.postprocess(predictions, tensor, [image])

    for index in range(args.warmup):
        pipeline(images[index % len(images)])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for index in range(args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        pipeline(images[index % len(images)])
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    report = {
        "engine": str(args.engine),
        "measurement_scope": "preprocess_H2D_TensorRT_forward_NMS_excludes_disk_io",
        "preloaded_images": len(images),
        "warmup_iterations": args.warmup,
        "benchmark_iterations": args.iterations,
        "mean_latency_ms": statistics.fmean(samples),
        "median_latency_ms": statistics.median(samples),
        "p95_latency_ms": float(np.percentile(samples, 95)),
        "fps": 1000 / statistics.fmean(samples),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "engine_size_mb": args.engine.stat().st_size / 1024**2,
        "execution_context_reused": True,
        "disk_io_timed": False,
        "device": torch.cuda.get_device_name(),
    }
    args.output.mkdir(parents=True)
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=report.keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerow(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

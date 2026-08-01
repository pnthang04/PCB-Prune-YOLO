"""Inference latency measurement."""

import statistics
import time
from collections.abc import Callable

import torch


def benchmark_latency(infer: Callable[[], object], warmup: int, iterations: int, device: torch.device) -> dict[str, float]:
    """Measure latency excluding model loading."""
    if warmup < 0 or iterations <= 0:
        raise ValueError("Số lần warm-up/benchmark không hợp lệ")
    for _ in range(warmup):
        infer()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        infer()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    mean = statistics.fmean(samples)
    return {"mean_latency_ms": mean, "median_latency_ms": statistics.median(samples), "p95_latency_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], "fps": 1000 / mean}


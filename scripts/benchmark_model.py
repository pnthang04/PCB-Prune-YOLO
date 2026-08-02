"""Benchmark a YOLO model on CPU or CUDA."""

import argparse
import platform
from pathlib import Path

import torch

from pcb_prune_yolo.benchmarking.complexity import file_size_mb, model_complexity
from pcb_prune_yolo.benchmarking.latency import benchmark_latency
from pcb_prune_yolo.benchmarking.report import write_report
from pcb_prune_yolo.config import load_config
from pcb_prune_yolo.utils.device import resolve_device


def main() -> None:
    """Load once, benchmark batch size one, and write JSON/CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark/default.yaml"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source-model", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--benchmark-iterations", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    config.update(
        {key: value for key, value in vars(args).items() if key != "config" and value is not None}
    )
    from ultralytics import YOLO

    checkpoint = Path(config["model"])
    device = resolve_device(config["device"])
    image = torch.randn(1, 3, int(config["imgsz"]), int(config["imgsz"]), device=device)
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if checkpoint.suffix == ".engine":
        from ultralytics.nn.autobackend import AutoBackend

        module = AutoBackend(str(checkpoint), device=device, fp16=True)
        image = image.half()
        source_model = config.get("source_model")
        if not source_model:
            raise ValueError("Benchmark TensorRT cần --source-model để xác minh params/MACs")
        source = YOLO(str(source_model)).model.to(device).eval()
        values = model_complexity(source, image.float())
        del source
        values["source_model"] = str(source_model)
        values["measurement_scope"] = "pure_engine_forward_excludes_preprocess_and_nms"
    else:
        module = YOLO(str(checkpoint)).model.to(device).eval()
        values = model_complexity(module, image)
        values["measurement_scope"] = "pure_model_forward_excludes_preprocess_and_nms"
    with torch.inference_mode():
        values.update(
            benchmark_latency(
                lambda: module(image),
                int(config["warmup_iterations"]),
                int(config["benchmark_iterations"]),
                device,
            )
        )
    values.update(
        {
            "model": str(checkpoint),
            "model_size_mb": file_size_mb(checkpoint),
            "batch_size": 1,
            "imgsz": int(config["imgsz"]),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_memory_mb": (
                torch.cuda.get_device_properties(device).total_memory / 1024**2
                if device.type == "cuda"
                else None
            ),
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else None
            ),
            "peak_gpu_memory_measurement": (
                "torch_cuda_allocator; TensorRT execution-context allocation is logged separately"
                if checkpoint.suffix == ".engine"
                else "torch_cuda_allocator"
            ),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    )
    import ultralytics

    values["ultralytics_version"] = ultralytics.__version__
    if checkpoint.suffix == ".engine":
        import tensorrt

        values["tensorrt_version"] = tensorrt.__version__
    write_report(values, Path(config["output"]), "benchmark")


if __name__ == "__main__":
    main()

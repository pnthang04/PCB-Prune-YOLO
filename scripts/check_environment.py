"""Print and optionally validate the runtime accelerator configuration."""

import argparse
import platform
from importlib.metadata import PackageNotFoundError, version

import torch


def package_version(name: str) -> str:
    """Return an installed package version or a clear missing marker."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    """Print Python, package, CUDA, GPU and VRAM information."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpus", type=int, default=0)
    args = parser.parse_args()

    print(f"Python: {platform.python_version()}")
    for name in ("torch", "ultralytics", "opencv-python", "PyYAML", "pandas"):
        print(f"{name}: {package_version(name)}")
    print(f"PyTorch CUDA build: {torch.version.cuda or 'none (CPU build)'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"cuDNN: {torch.backends.cudnn.version() or 'unavailable'}")
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")
    for index in range(gpu_count):
        properties = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {properties.name}")
        print(f"  capability: {properties.major}.{properties.minor}")
        print(f"  VRAM: {properties.total_memory / 1024**3:.2f} GiB")
    if gpu_count < args.require_gpus:
        raise SystemExit(f"Need {args.require_gpus} GPUs, but PyTorch sees {gpu_count}.")


if __name__ == "__main__":
    main()

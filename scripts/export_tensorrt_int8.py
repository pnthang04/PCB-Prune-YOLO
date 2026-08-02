"""Export a YOLO checkpoint to a static TensorRT INT8 PTQ engine.

Calibration images are selected deterministically from the training split only.
The exact manifest and calibration cache are preserved with the engine.
"""

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from pcb_prune_yolo.benchmarking.complexity import model_complexity
from pcb_prune_yolo.benchmarking.report import write_report
from pcb_prune_yolo.utils.device import resolve_device


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _calibration_images(dataset_root: Path, count: int, seed: int) -> list[Path]:
    train_root = dataset_root / "images" / "train"
    images = sorted(path.resolve() for path in train_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"Không tìm thấy ảnh calibration trong {train_root}")
    if count > len(images):
        raise ValueError(f"calibration-count={count} vượt quá {len(images)} ảnh train")
    chosen = random.Random(seed).sample(images, count)
    return sorted(chosen)


def main() -> None:
    """Build a non-overwriting INT8 engine and record complete provenance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/data/deeppcb.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/deployment_optimization/int8_ptq"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace", type=float, default=4.0)
    parser.add_argument("--calibration-count", type=int, default=500)
    parser.add_argument("--calibration-seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    data_file = args.data.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not data_file.is_file():
        raise FileNotFoundError(data_file)

    data = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    configured_root = Path(data["path"])
    dataset_root = (data_file.parent / configured_root).resolve() if not configured_root.is_absolute() else configured_root
    if not dataset_root.is_dir():
        # Project data YAML paths are conventionally relative to the repository root.
        dataset_root = (Path.cwd() / configured_root).resolve()
    images = _calibration_images(dataset_root, args.calibration_count, args.calibration_seed)
    train_root = (dataset_root / "images" / "train").resolve()
    if any(train_root not in image.parents for image in images):
        raise RuntimeError("Calibration manifest chứa ảnh ngoài training split")

    output_dir = args.output / args.name
    engine = output_dir / "model.engine"
    report = output_dir / "export.json"
    manifest = output_dir / "calibration_images.txt"
    calibration_yaml = output_dir / "calibration_data.yaml"
    cache = output_dir / "calibration.cache"
    for path in (engine, report, manifest, calibration_yaml, cache):
        if path.exists():
            raise FileExistsError(f"Từ chối ghi đè artifact: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest.write_text("".join(f"{path}\n" for path in images), encoding="utf-8")
    calibration_config = {
        "path": str(dataset_root),
        "train": str(manifest.resolve()),
        "val": str(manifest.resolve()),
        "names": data["names"],
    }
    calibration_yaml.write_text(yaml.safe_dump(calibration_config, sort_keys=False), encoding="utf-8")

    from ultralytics import YOLO, __version__ as ultralytics_version

    device = resolve_device(args.device)
    yolo = YOLO(str(checkpoint))
    module = yolo.model.to(device).eval()
    image = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        raw = module(image)
    prediction = raw[0] if isinstance(raw, (tuple, list)) else raw
    class_count = len(module.names)
    expected = [1, 4 + class_count, 8400]
    if not isinstance(prediction, torch.Tensor) or list(prediction.shape) != expected:
        raise RuntimeError(f"Source output không hợp lệ: {getattr(prediction, 'shape', None)}")
    complexity = model_complexity(module, image)

    with tempfile.TemporaryDirectory(prefix="pcb_int8_export_") as temp:
        temp_dir = Path(temp)
        temp_checkpoint = temp_dir / "source.pt"
        shutil.copy2(checkpoint, temp_checkpoint)
        temp_yolo = YOLO(str(temp_checkpoint))
        exported = Path(
            temp_yolo.export(
                format="engine",
                imgsz=args.imgsz,
                quantize=8,
                dynamic=False,
                batch=1,
                device=str(device),
                workspace=args.workspace,
                simplify=False,
                nms=False,
                data=str(calibration_yaml.resolve()),
                split="val",
            )
        )
        if not exported.is_file():
            raise RuntimeError(f"Không tìm thấy engine đã export: {exported}")
        shutil.move(str(exported), engine)
        generated_cache = temp_checkpoint.with_suffix(".cache")
        if not generated_cache.is_file():
            raise RuntimeError("TensorRT không tạo calibration cache")
        shutil.move(str(generated_cache), cache)

    import tensorrt as trt

    values = {
        "name": args.name,
        "source_checkpoint": str(args.checkpoint),
        "engine": str(engine),
        "input_shape": [1, 3, args.imgsz, args.imgsz],
        "output_shape_before_export": list(prediction.shape),
        "class_count": class_count,
        "class_names": list(module.names.values()),
        "precision": "int8_ptq",
        "dynamic": False,
        "batch_size": 1,
        "calibration_split": "train",
        "calibration_count": len(images),
        "calibration_seed": args.calibration_seed,
        "calibration_manifest": str(manifest),
        "calibration_cache": str(cache),
        "calibration_algorithm": "MINMAX_CALIBRATION (Ultralytics 8.4.115 TensorRT exporter)",
        **complexity,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tensorrt_version": trt.__version__,
        "ultralytics_version": ultralytics_version,
    }
    write_report(values, output_dir, "export")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()

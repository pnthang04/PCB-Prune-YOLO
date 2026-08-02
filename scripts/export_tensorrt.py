"""Export a verified YOLO checkpoint to a static TensorRT FP16 engine."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from pcb_prune_yolo.benchmarking.complexity import model_complexity
from pcb_prune_yolo.benchmarking.report import write_report
from pcb_prune_yolo.utils.device import resolve_device


def main() -> None:
    """Verify the source object, export it once, and preserve provenance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/tensorrt_fp16"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace", type=float, default=4.0)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    output_dir = args.output / args.name
    engine = output_dir / "model.engine"
    report = output_dir / "export.json"
    generated_engine = args.checkpoint.with_suffix(".engine")
    for path in (engine, report, generated_engine):
        if path.exists():
            raise FileExistsError(f"Từ chối ghi đè artifact: {path}")

    from ultralytics import YOLO, __version__ as ultralytics_version

    device = resolve_device(args.device)
    yolo = YOLO(str(args.checkpoint))
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

    exported = Path(
        yolo.export(
            format="engine",
            imgsz=args.imgsz,
            half=True,
            dynamic=False,
            batch=1,
            device=str(device),
            workspace=args.workspace,
            simplify=False,
            nms=False,
        )
    )
    if exported.resolve() != generated_engine.resolve() or not exported.is_file():
        raise RuntimeError(f"Ultralytics trả về engine ngoài dự kiến: {exported}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), engine)

    import tensorrt as trt

    values = {
        "name": args.name,
        "source_checkpoint": str(args.checkpoint),
        "engine": str(engine),
        "input_shape": [1, 3, args.imgsz, args.imgsz],
        "output_shape_before_export": list(prediction.shape),
        "class_count": class_count,
        "class_names": list(module.names.values()),
        "precision": "fp16",
        "dynamic": False,
        "batch_size": 1,
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

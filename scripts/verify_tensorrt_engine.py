"""Load a TensorRT engine in a new process and verify raw inference."""

import argparse
import json
from pathlib import Path

import torch

from pcb_prune_yolo.utils.device import resolve_device


def main() -> None:
    """Verify static input, output shape, classes, and FP16 engine execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from ultralytics.nn.autobackend import AutoBackend

    device = resolve_device(args.device)
    backend = AutoBackend(str(args.engine), device=device, fp16=True)
    image = torch.randn(1, 3, args.imgsz, args.imgsz, device=device, dtype=torch.float16)
    with torch.inference_mode():
        output = backend(image)
    prediction = output[0] if isinstance(output, (tuple, list)) else output
    class_count = len(backend.names)
    expected = [1, 4 + class_count, 8400]
    if not isinstance(prediction, torch.Tensor) or list(prediction.shape) != expected:
        raise RuntimeError(f"Engine output không hợp lệ: {getattr(prediction, 'shape', None)}")
    print(
        json.dumps(
            {
                "loaded": True,
                "input_shape": list(image.shape),
                "input_dtype": str(image.dtype),
                "prediction_shape": list(prediction.shape),
                "class_count": class_count,
                "class_names": list(backend.names.values()),
                "device": str(device),
            }
        )
    )


if __name__ == "__main__":
    main()

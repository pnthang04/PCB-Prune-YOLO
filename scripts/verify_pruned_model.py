"""Load a changed YOLO architecture in a new process and verify inference."""

import argparse
import json
from pathlib import Path

import torch

from pcb_prune_yolo.utils.device import resolve_device


def main() -> None:
    """Load an Ultralytics checkpoint and check decoded output and class metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from ultralytics import YOLO

    device = resolve_device(args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    image = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        output = model(image)
    prediction = output[0] if isinstance(output, (tuple, list)) else output
    class_count = len(model.names)
    if not isinstance(prediction, torch.Tensor) or list(prediction.shape[:2]) != [1, 4 + class_count]:
        raise RuntimeError(f"Output checkpoint không hợp lệ: {getattr(prediction, 'shape', None)}")
    print(
        json.dumps(
            {
                "loaded": True,
                "class_count": class_count,
                "class_names": list(model.names.values()),
                "prediction_shape": list(prediction.shape),
                "device": str(device),
            }
        )
    )


if __name__ == "__main__":
    main()

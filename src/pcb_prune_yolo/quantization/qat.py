"""NVIDIA ModelOpt INT8 QAT preparation for the fixed P30 architecture."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _dataset_root(data_file: Path) -> Path:
    from pcb_prune_yolo.config import load_config

    data = load_config(data_file)
    configured = Path(data["path"])
    candidates = (
        configured,
        data_file.parent / configured,
        Path.cwd() / configured,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    raise FileNotFoundError(f"Không tìm thấy dataset root từ {data_file}: {configured}")


def _select_train_images(data_file: Path, count: int, seed: int) -> list[Path]:
    train_root = _dataset_root(data_file) / "images" / "train"
    images = sorted(path.resolve() for path in train_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if count < 1 or count > len(images):
        raise ValueError(f"calibration_images phải trong [1,{len(images)}], nhận {count}")
    return sorted(random.Random(seed).sample(images, count))


def _preprocess(path: Path, imgsz: int) -> torch.Tensor:
    from ultralytics.data.augment import LetterBox

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Không đọc được ảnh calibration: {path}")
    image = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image)
    image = image[:, :, ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)


def _quantizer_stats(model: torch.nn.Module) -> dict[str, Any]:
    rows = []
    for name, module in model.named_modules():
        if type(module).__name__ != "TensorQuantizer":
            continue
        rows.append(
            {
                "name": name,
                "enabled": bool(module.is_enabled),
                "num_bits": int(module.num_bits),
                "axis": module.axis,
                "has_amax": getattr(module, "amax", None) is not None,
            }
        )
    return {
        "quantizer_count": len(rows),
        "enabled_quantizer_count": sum(row["enabled"] for row in rows),
        "quantizers": rows,
    }


def prepare_int8_qat(
    model: torch.nn.Module,
    data_file: Path,
    quantization: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    imgsz: int,
) -> dict[str, Any]:
    """Insert fake quantizers and calibrate them using train images only."""
    import modelopt
    import modelopt.torch.quantization as mtq

    if quantization["backend"] != "nvidia-modelopt":
        raise ValueError(f"Backend QAT chưa hỗ trợ: {quantization['backend']}")
    if quantization["preset"] != "INT8_DEFAULT_CFG":
        raise ValueError(f"Preset QAT chưa hỗ trợ: {quantization['preset']}")

    output_dir.mkdir(parents=True, exist_ok=False)
    images = _select_train_images(
        data_file.resolve(),
        int(quantization["calibration_images"]),
        int(quantization["calibration_seed"]),
    )
    manifest = output_dir / "calibration_images.txt"
    manifest.write_text("".join(f"{path}\n" for path in images), encoding="utf-8")

    config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    exclusions = list(quantization["excluded_modules"])
    for name in exclusions:
        config["quant_cfg"].append(
            {"quantizer_name": f"{name}.*_quantizer", "enable": False}
        )

    model = model.to(device).eval()

    def forward_loop(module: torch.nn.Module) -> None:
        with torch.inference_mode():
            for path in images:
                module(_preprocess(path, imgsz).unsqueeze(0).to(device))

    mtq.quantize(model, config, forward_loop)
    stats = _quantizer_stats(model)
    stats.update(
        {
            "backend": "nvidia-modelopt",
            "modelopt_version": modelopt.__version__,
            "preset": "INT8_DEFAULT_CFG",
            "calibration_algorithm": quantization["calibration_algorithm"],
            "calibration_split": "train",
            "calibration_count": len(images),
            "calibration_manifest": str(manifest),
            "excluded_modules": exclusions,
        }
    )
    (output_dir / "quantization_prepare.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats

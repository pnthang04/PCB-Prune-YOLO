"""PyTorch device selection."""

import torch


def resolve_device(value: str) -> torch.device:
    """Resolve 'auto', CPU or CUDA device strings."""
    requested = "cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng không khả dụng")
    return device


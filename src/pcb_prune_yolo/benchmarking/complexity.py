"""Simple model size metrics."""

from pathlib import Path

import torch


def parameter_count(model: torch.nn.Module) -> int:
    """Count all model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def file_size_mb(path: Path) -> float:
    """Return file size in MiB."""
    return path.stat().st_size / (1024 * 1024)


def model_complexity(
    model: torch.nn.Module, example_input: torch.Tensor
) -> dict[str, int | float | None]:
    """Count parameters and MACs for one forward pass when supported."""
    parameters = parameter_count(model)
    try:
        import torch_pruning as tp

        macs, counted_parameters = tp.utils.count_ops_and_params(model, example_input)
        return {
            "parameters": parameters,
            "counted_parameters": int(counted_parameters),
            "macs": int(macs),
            "gmacs": float(macs / 1e9),
            "flops_estimate": int(2 * macs),
            "gflops_estimate": float(2 * macs / 1e9),
        }
    except Exception as error:  # pragma: no cover - model-specific fallback
        return {
            "parameters": parameters,
            "counted_parameters": None,
            "macs": None,
            "gmacs": None,
            "flops_estimate": None,
            "gflops_estimate": None,
            "complexity_error": f"{type(error).__name__}: {error}",
        }

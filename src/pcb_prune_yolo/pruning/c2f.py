"""Pruning-friendly equivalent of the Ultralytics C2f block."""

import copy

import torch
from torch import nn
from ultralytics.nn.modules import Bottleneck, C2f, Conv


class PrunableC2f(nn.Module):
    """C2f with two explicit input branches instead of a channel split."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two explicit branches and concatenate bottleneck outputs."""
        values = [self.cv0(x), self.cv1(x)]
        values.extend(block(values[-1]) for block in self.m)
        return self.cv2(torch.cat(values, 1))


def _uses_shortcut(block: Bottleneck) -> bool:
    return bool(getattr(block, "add", False))


def _transfer_c2f(source: C2f, target: PrunableC2f) -> None:
    """Transfer a C2f exactly by splitting its first convolution in half."""
    target.cv2 = source.cv2
    target.m = source.m
    source_state = source.state_dict()
    target_state = target.state_dict()
    half = source_state["cv1.conv.weight"].shape[0] // 2
    target_state["cv0.conv.weight"] = source_state["cv1.conv.weight"][:half]
    target_state["cv1.conv.weight"] = source_state["cv1.conv.weight"][half:]
    for key in ("weight", "bias", "running_mean", "running_var"):
        value = source_state[f"cv1.bn.{key}"]
        target_state[f"cv0.bn.{key}"] = value[:half]
        target_state[f"cv1.bn.{key}"] = value[half:]
    for key, value in source_state.items():
        if not key.startswith("cv1."):
            target_state[key] = value
    target.load_state_dict(target_state)
    for branch in (target.cv0, target.cv1):
        branch.bn.eps = source.cv1.bn.eps
        branch.bn.momentum = source.cv1.bn.momentum
        branch.act = copy.deepcopy(source.cv1.act)
    for attribute in ("i", "f", "type", "np"):
        if hasattr(source, attribute):
            setattr(target, attribute, getattr(source, attribute))


def replace_c2f(module: nn.Module) -> list[str]:
    """Replace every C2f recursively and return the replaced module names."""
    replaced: list[str] = []
    for name, child in list(module.named_children()):
        if isinstance(child, C2f):
            output_channels = child.cv2.conv.out_channels
            replacement = PrunableC2f(
                child.cv1.conv.in_channels,
                output_channels,
                n=len(child.m),
                shortcut=_uses_shortcut(child.m[0]),
                g=child.m[0].cv2.conv.groups,
                e=child.c / output_channels,
            )
            _transfer_c2f(child, replacement)
            reference = child.cv1.conv.weight
            replacement.to(device=reference.device, dtype=reference.dtype)
            replacement.train(child.training)
            setattr(module, name, replacement)
            replaced.append(name)
        else:
            for nested in replace_c2f(child):
                replaced.append(f"{name}.{nested}")
    return replaced

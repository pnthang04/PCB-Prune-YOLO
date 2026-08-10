"""Hard-Concrete gates adapted from the official DiariZen implementation.

Reference: BUTSpeechFIT/DiariZen commit 844f5555, MIT licensed.
"""

import math

import torch
from torch import nn


class HardConcrete(nn.Module):
    """Learn a stochastic structured mask with a differentiable expected L0 norm."""

    def __init__(
        self,
        channels: int,
        init_drop_rate: float = 0.01,
        temperature: float = 2 / 3,
        limit_l: float = -0.1,
        limit_r: float = 1.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels phải dương")
        self.channels = channels
        self.temperature = temperature
        self.limit_l = limit_l
        self.limit_r = limit_r
        self.eps = eps
        self.bias = -temperature * math.log(-limit_l / limit_r)
        mean = math.log(1 - init_drop_rate) - math.log(init_drop_rate)
        self.log_alpha = nn.Parameter(torch.empty(channels).normal_(mean, 0.01))

    def expected_mask(self) -> torch.Tensor:
        """Return each channel's non-zero probability."""
        return torch.sigmoid(self.log_alpha + self.bias)

    def l0_norm(self) -> torch.Tensor:
        """Return the differentiable expected retained-channel count."""
        return self.expected_mask().sum()

    def deterministic_mask(self) -> torch.Tensor:
        """Compile a fixed mask exactly as DiariZen does in evaluation mode."""
        zero_count = round(self.channels - float(self.l0_norm().detach()))
        mask = torch.sigmoid(self.log_alpha / self.temperature * 0.8)
        if zero_count:
            indices = torch.topk(mask, k=zero_count, largest=False).indices
            mask = mask.clone()
            mask[indices] = 0
        return mask

    def forward(self) -> torch.Tensor:
        if not self.training:
            return self.deterministic_mask()
        uniform = self.log_alpha.new_empty(self.channels).uniform_(self.eps, 1 - self.eps)
        sample = torch.sigmoid((torch.logit(uniform) + self.log_alpha) / self.temperature)
        stretched = sample * (self.limit_r - self.limit_l) + self.limit_l
        return stretched.clamp(0, 1)


class ChannelGateHook:
    """Picklable forward hook applying a module-owned channel gate."""

    def __call__(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        gate = module.hard_concrete_gate()
        return output * gate.view(1, -1, 1, 1)


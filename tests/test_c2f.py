"""Tests for the pruning-friendly C2f conversion."""

import torch
from torch import nn
from ultralytics.nn.modules import C2f

from pcb_prune_yolo.pruning.c2f import PrunableC2f, replace_c2f


def test_replace_c2f_preserves_eval_output() -> None:
    torch.manual_seed(42)
    model = nn.Sequential(C2f(8, 16, n=1)).eval()
    image = torch.randn(1, 8, 16, 16)
    with torch.no_grad():
        expected = model(image)
    replaced = replace_c2f(model)
    with torch.no_grad():
        actual = model(image)
    assert replaced == ["0"]
    assert isinstance(model[0], PrunableC2f)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

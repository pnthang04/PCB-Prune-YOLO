import torch

from pcb_prune_yolo.halp.stage2 import (
    PrefixOption,
    multiple_choice_knapsack,
    original_lut_name,
    taylor_bn_term,
)


def test_original_lut_name_only_maps_converted_backbone_input_branch():
    assert original_lut_name("model.2.cv0.conv") == "model.2.cv1.conv"
    assert original_lut_name("model.2.cv1.conv") == "model.2.cv1.conv"
    assert original_lut_name("model.12.cv0.conv") == "model.12.cv0.conv"


def test_taylor_bn_term():
    bn = torch.nn.BatchNorm2d(2)
    bn.weight.grad = torch.tensor([2.0, -1.0])
    bn.bias.grad = torch.tensor([3.0, 4.0])
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([0.5, 2.0]))
        bn.bias.copy_(torch.tensor([1.0, -1.0]))
    assert torch.equal(taylor_bn_term(bn), torch.tensor([4.0, 6.0]))


def test_multiple_choice_knapsack_enforces_one_prefix_per_layer():
    layers = [
        [PrefixOption(8, 1.0, 8.0), PrefixOption(4, 0.5, 5.0)],
        [PrefixOption(8, 1.0, 7.0), PrefixOption(4, 0.5, 6.0)],
    ]
    selected, value = multiple_choice_knapsack(layers, budget_ms=1.5)
    assert [option.keep_channels for option in selected] == [8, 4]
    assert value == 14.0

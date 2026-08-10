import torch
from torch import nn

import torch_pruning as tp

from pcb_prune_yolo.pruning.gated_groups import GatedGroupRegistry
from pcb_prune_yolo.pruning.hard_concrete import HardConcrete


def test_hard_concrete_expected_l0_and_deterministic_mask() -> None:
    gate = HardConcrete(8)
    gate.log_alpha.data[:3] = -20
    gate.log_alpha.data[3:] = 20
    gate.eval()
    mask = gate()
    assert mask.eq(0).sum() == 3
    assert gate.l0_norm().requires_grad


def test_depgraph_gates_receive_gradients_and_materialize() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.SiLU(),
        nn.Conv2d(16, 16, 1),
    )
    example = torch.randn(1, 3, 16, 16)
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    registry = GatedGroupRegistry(model, graph, list(graph.get_all_groups()), min_channels=4)
    model.train()
    loss = model(example).square().mean() + registry.expected_sparsity()
    loss.backward()
    assert all(parameter.grad is not None for parameter in registry.gate_parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in registry.gate_parameters())

    for group in registry.groups:
        group.gate.log_alpha.data[:4] = -20
        group.gate.log_alpha.data[4:] = 20
    model.eval()
    selected = registry.materialize()
    output = model(example)
    assert all(len(indices) == 4 for indices in selected.values())
    assert output.shape == (1, 12, 16, 16)

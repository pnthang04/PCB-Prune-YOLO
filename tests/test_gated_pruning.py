import torch
from torch import nn

import torch_pruning as tp
from ultralytics.utils.torch_utils import EarlyStopping

from pcb_prune_yolo.pruning.gated_groups import GatedGroupRegistry
from pcb_prune_yolo.pruning.hard_concrete import HardConcrete
from pcb_prune_yolo.training.trainer import _MinHoldEpochsStopper


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


def _round_to_registry(min_channels: int, round_to: int | None, channels: int = 8) -> GatedGroupRegistry:
    model = nn.Sequential(
        nn.Conv2d(3, channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.SiLU(),
        nn.Conv2d(channels, channels, 1),
    )
    example = torch.randn(1, 3, 16, 16)
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    return GatedGroupRegistry(
        model, graph, list(graph.get_all_groups()), min_channels=min_channels, round_to=round_to
    )


def test_round_to_rounds_remaining_width_down_and_prefers_low_confidence_channels() -> None:
    registry = _round_to_registry(min_channels=2, round_to=4)
    group = registry.groups[0]
    # Gate drops channels 0-2; among the 5 kept channels, channel 3 has the
    # lowest log_alpha (closest to the gate's own drop threshold).
    group.gate.log_alpha.data[:3] = -20
    for offset, value in enumerate([1.0, 5.0, 10.0, 15.0, 20.0]):
        group.gate.log_alpha.data[3 + offset] = value

    indices = registry.selected_indices()[group.root_name]

    # Remaining width 5 rounds down to the nearest multiple of 4 -> 4 pruned.
    assert len(indices) == 4
    assert set(indices) == {0, 1, 2, 3}


def test_round_to_never_restores_a_gate_dropped_channel() -> None:
    registry = _round_to_registry(min_channels=1, round_to=8)
    group = registry.groups[0]
    group.gate.log_alpha.data[0] = -20
    group.gate.log_alpha.data[1:] = 20

    indices = registry.selected_indices()[group.root_name]

    # round_to=8 forces the remaining width toward a multiple of 8 (clamped
    # to min_channels=1 here since channels=8), pruning far more than the
    # gate alone selected, but the originally gate-dropped channel must
    # still be among them.
    assert 0 in indices
    assert len(indices) > 1


def test_round_to_respects_min_channels_floor() -> None:
    registry = _round_to_registry(min_channels=4, round_to=8)
    group = registry.groups[0]
    group.gate.log_alpha.data[:3] = -20
    group.gate.log_alpha.data[3:] = 20

    indices = registry.selected_indices()[group.root_name]

    # Naive rounding of remaining=5 down to a multiple of 8 would go to 0;
    # the min_channels=4 floor must win instead.
    assert len(indices) == 4
    assert group.gate.channels - len(indices) == 4


def test_round_to_none_matches_existing_behavior() -> None:
    registry = _round_to_registry(min_channels=4, round_to=None)
    group = registry.groups[0]
    group.gate.log_alpha.data[:4] = -20
    group.gate.log_alpha.data[4:] = 20

    indices = registry.selected_indices()[group.root_name]

    assert set(indices) == {0, 1, 2, 3}


def _small_gated_model() -> nn.Sequential:
    # conv1 stays 16x16 (k=3, pad=1, stride=1); conv2 downsamples to 8x8 (k=1, stride=2).
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.SiLU(),
        nn.Conv2d(16, 16, 1, stride=2),
    )


def test_mac_cost_scales_by_spatial_size_unlike_param_cost() -> None:
    example = torch.randn(1, 3, 16, 16)

    # Independent model/graph per registry: GatedGroupRegistry attaches a gate
    # submodule to each owner, so a second registry on the same instance would
    # find every group already gated and raise "no eligible group" instead.
    param_model = _small_gated_model()
    param_graph = tp.DependencyGraph().build_dependency(param_model, example_inputs=example)
    param_registry = GatedGroupRegistry(
        param_model, param_graph, list(param_graph.get_all_groups()), min_channels=4, cost_type="params"
    )

    mac_model = _small_gated_model()
    mac_graph = tp.DependencyGraph().build_dependency(mac_model, example_inputs=example)
    mac_registry = GatedGroupRegistry(
        mac_model,
        mac_graph,
        list(mac_graph.get_all_groups()),
        min_channels=4,
        cost_type="macs",
        example_input=example,
    )

    # DepGraph enumerates one group per candidate root, in no guaranteed order;
    # select the one actually rooted at conv1 (Sequential index "0") rather
    # than assuming groups[0].
    param_conv1_group = next(g for g in param_registry.groups if g.root_name == "0")
    mac_conv1_group = next(g for g in mac_registry.groups if g.root_name == "0")

    # conv1-out (27) + BN (2) + conv2-in (16) = 45 params per channel, identical for all 16.
    assert torch.allclose(param_conv1_group.costs, torch.full((16,), 45.0))
    # conv1-out (27*16*16=6912) + BN (0) + conv2-in (16*8*8=1024) = 7936 MACs per channel.
    assert torch.allclose(mac_conv1_group.costs, torch.full((16,), 7936.0))


def test_gated_group_registry_rejects_unknown_cost_type() -> None:
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.Conv2d(16, 16, 1))
    example = torch.randn(1, 3, 8, 8)
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    groups = list(graph.get_all_groups())
    try:
        GatedGroupRegistry(model, graph, groups, min_channels=4, cost_type="latency")
    except ValueError:
        return
    raise AssertionError("cost_type không hợp lệ phải bị từ chối")


def test_gated_group_registry_requires_example_input_for_macs() -> None:
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.Conv2d(16, 16, 1))
    example = torch.randn(1, 3, 8, 8)
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    groups = list(graph.get_all_groups())
    try:
        GatedGroupRegistry(model, graph, groups, min_channels=4, cost_type="macs")
    except ValueError:
        return
    raise AssertionError("cost_type='macs' thiếu example_input phải bị từ chối")


def test_min_hold_epochs_stopper_suppresses_stop_before_threshold() -> None:
    stopper = _MinHoldEpochsStopper(EarlyStopping(patience=5), min_hold_epochs=20)
    # Fitness never improves past epoch 1, so the wrapped EarlyStopping alone
    # would signal stop at epoch 7 (patience=5); the wrapper must suppress
    # that until min_hold_epochs even though it keeps calling the real stopper.
    for epoch in range(1, 15):
        assert stopper(epoch, fitness=1.0) is False
    assert stopper.best_epoch == 1
    assert stopper.possible_stop is True


def test_min_hold_epochs_stopper_allows_stop_after_threshold() -> None:
    stopper = _MinHoldEpochsStopper(EarlyStopping(patience=5), min_hold_epochs=20)
    for epoch in range(1, 20):
        assert stopper(epoch, fitness=1.0) is False
    assert stopper(20, fitness=1.0) is True


def test_min_hold_epochs_stopper_never_stops_while_still_improving() -> None:
    stopper = _MinHoldEpochsStopper(EarlyStopping(patience=3), min_hold_epochs=5)
    for epoch in range(1, 10):
        assert stopper(epoch, fitness=float(epoch)) is False

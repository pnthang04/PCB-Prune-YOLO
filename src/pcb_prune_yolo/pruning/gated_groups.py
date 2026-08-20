"""DepGraph-backed learnable channel gates and physical materialization."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from pcb_prune_yolo.pruning.hard_concrete import ChannelGateHook, HardConcrete


@dataclass
class GatedGroup:
    root_name: str
    gate_owner_name: str
    root: nn.Module
    owner: nn.Module
    group: Any
    costs: torch.Tensor
    hook: Any

    @property
    def gate(self) -> HardConcrete:
        return self.owner.hard_concrete_gate


class GatedGroupRegistry:
    """Attach one learned gate to each eligible DepGraph Conv output group."""

    def __init__(
        self,
        model: nn.Module,
        graph: Any,
        groups: list[Any],
        protected: list[nn.Module] | None = None,
        init_drop_rate: float = 0.01,
        min_channels: int = 8,
        cost_type: str = "params",
        example_input: torch.Tensor | None = None,
    ) -> None:
        if cost_type not in {"params", "macs"}:
            raise ValueError(f"cost_type không hỗ trợ: {cost_type}")
        if cost_type == "macs" and example_input is None:
            raise ValueError("cost_type='macs' cần example_input để đo kích thước không gian")
        self.model = model
        self.graph = graph
        self.original_cost = self._total_cost(model, cost_type, example_input)
        self.min_channels = min_channels
        self.cost_type = cost_type
        self._spatial_sizes = (
            _capture_conv_output_sizes(model, example_input) if cost_type == "macs" else {}
        )
        protected_ids = {id(module) for module in protected or []}
        names = dict(model.named_modules())
        reverse_names = {id(module): name for name, module in names.items()}
        self.groups: list[GatedGroup] = []
        for group in groups:
            root = group[0].dep.target.module
            if not isinstance(root, nn.Conv2d) or id(root) in protected_ids:
                continue
            channels = root.out_channels
            if channels <= min_channels:
                continue
            root_name = reverse_names[id(root)]
            owner_name = root_name.removesuffix(".conv")
            owner = names.get(owner_name, root)
            if hasattr(owner, "hard_concrete_gate"):
                continue
            owner.add_module(
                "hard_concrete_gate",
                HardConcrete(channels, init_drop_rate=init_drop_rate),
            )
            hook = owner.register_forward_hook(ChannelGateHook())
            costs = self._channel_costs(group, channels, root.weight.device)
            self.groups.append(GatedGroup(root_name, owner_name, root, owner, group, costs, hook))
        if not self.groups:
            raise RuntimeError("Không tìm thấy DepGraph group phù hợp để gắn gate")

    def _channel_costs(self, group: Any, channels: int, device: torch.device) -> torch.Tensor:
        """Estimate unique removal cost for every root channel, in params or MACs."""
        costs = torch.zeros(channels, device=device)
        seen: set[tuple[int, str, int]] = set()
        for item in group.items:
            module = item.dep.target.module
            root_indices = item.root_idxs or item.idxs
            is_out = self.graph.is_out_channel_pruning_fn(item.dep.handler)
            dimension = "out" if is_out else "in"
            for index, root_index in zip(item.idxs, root_indices):
                key = (id(module), dimension, int(index))
                if key in seen or not 0 <= int(root_index) < channels:
                    continue
                seen.add(key)
                value = (
                    self._slice_macs(module, int(index), is_out)
                    if self.cost_type == "macs"
                    else self._slice_parameters(module, int(index), is_out)
                )
                costs[int(root_index)] += value
        return costs.clamp_min(1)

    @staticmethod
    def _slice_parameters(module: nn.Module, index: int, is_out: bool) -> int:
        if isinstance(module, nn.Conv2d):
            if is_out:
                return module.weight[index].numel() + int(module.bias is not None)
            if module.groups == 1:
                return module.weight[:, index].numel()
            return max(1, module.weight.numel() // module.in_channels)
        if isinstance(module, nn.BatchNorm2d):
            return int(module.weight is not None) + int(module.bias is not None)
        if isinstance(module, nn.Linear):
            if is_out:
                return module.weight[index].numel() + int(module.bias is not None)
            return module.weight[:, index].numel()
        return 0

    def _slice_macs(self, module: nn.Module, index: int, is_out: bool) -> int | float:
        """Multiply-accumulate cost of removing one channel index.

        Reuses the parameter-cost weight count (a conv kernel's weights are
        applied once per output pixel) and scales it by the module's own
        output spatial size. BatchNorm is elementwise and contributes
        negligible MACs next to its owning convolution, so it costs 0 here
        even though it counts toward parameter cost.
        """
        if isinstance(module, nn.BatchNorm2d):
            return 0
        weight_count = self._slice_parameters(module, index, is_out)
        if isinstance(module, nn.Conv2d):
            weight_count -= int(is_out and module.bias is not None)
            height, width = self._spatial_sizes[id(module)]
            return weight_count * height * width
        return weight_count

    @staticmethod
    def _total_cost(model: nn.Module, cost_type: str, example_input: torch.Tensor | None) -> float:
        """Return the whole-model cost that group costs must be a fraction of.

        Group costs and this total must share units, or expected_sparsity()
        is a meaningless ratio: MAC-cost channels summed against a
        parameter-count denominator inflated an observed run's expected
        sparsity to 0.64 against a target of 0.10.
        """
        if cost_type == "params":
            return sum(p.numel() for p in model.parameters())
        import torch_pruning as tp

        macs, _ = tp.utils.count_ops_and_params(model, example_input)
        return float(macs)

    def gate_parameters(self) -> list[nn.Parameter]:
        return [group.gate.log_alpha for group in self.groups]

    def expected_pruned_cost(self) -> torch.Tensor:
        return sum(
            ((1 - group.gate.expected_mask()) * group.costs).sum() for group in self.groups
        )

    def expected_sparsity(self) -> torch.Tensor:
        return self.expected_pruned_cost() / self.original_cost

    def report(self) -> dict[str, float | int]:
        return {
            "groups": len(self.groups),
            "gated_channels": sum(group.gate.channels for group in self.groups),
            "expected_sparsity": float(self.expected_sparsity().detach()),
        }

    def selected_indices(self) -> dict[str, list[int]]:
        selected = {}
        for group in self.groups:
            mask = group.gate.deterministic_mask()
            indices = mask.eq(0).nonzero().flatten().tolist()
            maximum = group.gate.channels - self.min_channels
            selected[group.root_name] = indices[:maximum]
        return selected

    def remove_gates(self) -> None:
        for group in self.groups:
            group.hook.remove()
            delattr(group.owner, "hard_concrete_gate")

    def materialize(self) -> dict[str, list[int]]:
        """Remove gates, then physically prune their learned indices through DepGraph."""
        selected = self.selected_indices()
        self.remove_gates()
        for group in self.groups:
            indices = selected[group.root_name]
            if indices:
                group.group.prune(indices)
        return selected


@torch.no_grad()
def _capture_conv_output_sizes(model: nn.Module, example_input: torch.Tensor) -> dict[int, tuple[int, int]]:
    """Run one forward pass and record each Conv2d's output (height, width) for MAC costing."""
    sizes: dict[int, tuple[int, int]] = {}

    def record(module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        sizes[id(module)] = (int(output.shape[-2]), int(output.shape[-1]))

    hooks = [
        module.register_forward_hook(record)
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    model(example_input.to(device))
    model.train(was_training)
    for hook in hooks:
        hook.remove()
    return sizes


def build_gated_registry(
    wrapper: Any,
    init_drop_rate: float = 0.01,
    min_channels: int = 8,
    cost_type: str = "params",
) -> GatedGroupRegistry:
    """Build gates from the project's already configured YOLO DepGraph wrapper."""
    graph = wrapper.build_dependency_graph()
    groups = list(graph.get_all_groups(ignored_layers=wrapper.ignored_root_modules()))
    return GatedGroupRegistry(
        wrapper.model,
        graph,
        groups,
        protected=wrapper.ignored_root_modules(),
        init_drop_rate=init_drop_rate,
        min_channels=min_channels,
        cost_type=cost_type,
        example_input=wrapper.example_input,
    )


def extract_gate_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy learned gate logits by their owning module name."""
    return {
        name: module.hard_concrete_gate.log_alpha.detach().cpu().clone()
        for name, module in model.named_modules()
        if hasattr(module, "hard_concrete_gate")
    }


def remove_embedded_gates(model: nn.Module) -> None:
    """Remove serialized gate hooks/modules before rebuilding DepGraph."""
    for module in model.modules():
        for hook_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, ChannelGateHook):
                del module._forward_hooks[hook_id]
        if hasattr(module, "hard_concrete_gate"):
            delattr(module, "hard_concrete_gate")

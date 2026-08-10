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
    ) -> None:
        self.model = model
        self.graph = graph
        self.original_parameters = sum(p.numel() for p in model.parameters())
        self.min_channels = min_channels
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
            costs = self._parameter_costs(group, channels, root.weight.device)
            self.groups.append(GatedGroup(root_name, owner_name, root, owner, group, costs, hook))
        if not self.groups:
            raise RuntimeError("Không tìm thấy DepGraph group phù hợp để gắn gate")

    def _parameter_costs(self, group: Any, channels: int, device: torch.device) -> torch.Tensor:
        """Estimate unique trainable parameter removal cost for every root channel."""
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
                value = self._slice_parameters(module, int(index), is_out)
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

    def gate_parameters(self) -> list[nn.Parameter]:
        return [group.gate.log_alpha for group in self.groups]

    def expected_pruned_parameters(self) -> torch.Tensor:
        return sum(
            ((1 - group.gate.expected_mask()) * group.costs).sum() for group in self.groups
        )

    def expected_sparsity(self) -> torch.Tensor:
        return self.expected_pruned_parameters() / self.original_parameters

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


def build_gated_registry(
    wrapper: Any,
    init_drop_rate: float = 0.01,
    min_channels: int = 8,
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

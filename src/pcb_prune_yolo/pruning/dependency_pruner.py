"""DepGraph integration for structurally pruning Ultralytics YOLO detectors."""

import copy
from pathlib import Path
from typing import Any

import torch

from pcb_prune_yolo.benchmarking.complexity import model_complexity
from pcb_prune_yolo.pruning.c2f import replace_c2f
from pcb_prune_yolo.pruning.importance import create_importance


class YOLODepGraphPruner:
    """Build and apply a DepGraph pruner while protecting the detection outputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        pruning_ratio: float,
        importance: str = "group_magnitude",
        iterative_steps: int = 1,
        round_to: int | None = 8,
        global_pruning: bool = False,
        hardware_policy: str = "standard",
        **_: Any,
    ) -> None:
        if not 0 <= pruning_ratio < 1:
            raise ValueError("pruning_ratio phải thuộc [0, 1)")
        self.model = model
        self.example_input = example_input
        self.pruning_ratio = pruning_ratio
        self.importance_name = importance
        self.iterative_steps = iterative_steps
        self.round_to = round_to
        self.global_pruning = global_pruning
        if hardware_policy not in {"standard", "block"}:
            raise ValueError(f"hardware_policy không hỗ trợ: {hardware_policy}")
        self.hardware_policy = hardware_policy
        self.graph: Any | None = None
        self.group_count = 0
        self.meta_pruner: Any | None = None
        self.replaced_c2f = replace_c2f(self.model)
        self.head_name, self.detection_head = self._find_detection_head()
        self.protected_modules = self._find_protected_modules()
        self.block_protected_modules = self._find_block_protected_modules()
        self.initial_channel_layout = self.channel_layout()

    def _find_detection_head(self) -> tuple[str, torch.nn.Module]:
        """Return the terminal Ultralytics Detect module without pinning its class."""
        candidates = [
            (name, module)
            for name, module in self.model.named_modules()
            if hasattr(module, "nc") and hasattr(module, "cv2") and hasattr(module, "cv3")
        ]
        if not candidates:
            raise RuntimeError("Không tìm thấy Ultralytics detection head để bảo vệ")
        return candidates[-1]

    def _find_protected_modules(self) -> list[tuple[str, torch.nn.Module]]:
        """Find fixed-width regression, classification and DFL output modules."""
        modules: list[tuple[str, torch.nn.Module]] = []
        for branch_name in ("cv2", "cv3"):
            branch = getattr(self.detection_head, branch_name, [])
            for index, sequence in enumerate(branch):
                if len(sequence):
                    modules.append(
                        (f"{self.head_name}.{branch_name}.{index}.{len(sequence) - 1}", sequence[-1])
                    )
        if hasattr(self.detection_head, "dfl"):
            modules.append((f"{self.head_name}.dfl.conv", self.detection_head.dfl.conv))
        return modules

    def protected_layer_names(self) -> list[str]:
        """List fixed-width output layers excluded as pruning roots."""
        return [name for name, _ in self.protected_modules]

    def _find_block_protected_modules(self) -> list[tuple[str, torch.nn.Module]]:
        """Protect C2f bottleneck outputs as roots for the block-level policy."""
        if self.hardware_policy != "block":
            return []
        protected: list[tuple[str, torch.nn.Module]] = []
        for name, module in self.model.named_modules():
            # Keep each Bottleneck output width intact so residual/C2f branch
            # boundaries stay regular, while its internal cv1 remains eligible.
            if ".m." in name and name.endswith(".cv2.conv") and isinstance(
                module, torch.nn.Conv2d
            ):
                protected.append((name, module))
        return protected

    def ignored_root_modules(self) -> list[torch.nn.Module]:
        """Return modules protected from serving as output-pruning roots."""
        return [
            module
            for _, module in [*self.protected_modules, *self.block_protected_modules]
        ]

    def channel_layout(self) -> dict[str, int]:
        """Return convolution output widths for alignment and mutation audits."""
        return {
            name: int(module.out_channels)
            for name, module in self.model.named_modules()
            if isinstance(module, torch.nn.Conv2d)
        }

    def channel_audit(self) -> dict[str, Any]:
        """Summarize changed convolution widths and hardware alignment."""
        current = self.channel_layout()
        changed = {
            name: {"before": before, "after": current[name]}
            for name, before in self.initial_channel_layout.items()
            if name in current and current[name] != before
        }
        prunable_widths = [
            width
            for name, width in current.items()
            if name not in self.protected_layer_names()
        ]
        return {
            "policy": self.hardware_policy,
            "round_to": self.round_to,
            "changed_convolution_outputs": len(changed),
            "changed_channels": changed,
            "all_prunable_outputs_aligned_to_8": all(v % 8 == 0 for v in prunable_widths),
            "aligned_to_8_count": sum(v % 8 == 0 for v in prunable_widths),
            "aligned_to_16_count": sum(v % 16 == 0 for v in prunable_widths),
            "audited_convolutions": len(prunable_widths),
            "block_protected_roots": [name for name, _ in self.block_protected_modules],
        }

    def build_dependency_graph(self) -> Any:
        """Trace dependencies and enumerate groups without changing any channels."""
        import torch_pruning as tp

        # Ultralytics strip_optimizer() stores checkpoints with requires_grad=False.
        # DepGraph relies on autograd edges, so restore tracing gradients in this
        # in-memory pruning model before both this graph and the high-level pruner.
        self.model.requires_grad_(True)
        self.graph = tp.DependencyGraph().build_dependency(
            self.model, example_inputs=self.example_input
        )
        # Materialize groups now so tracing/API errors fail during dry-run.
        groups = list(
            self.graph.get_all_groups(
                ignored_layers=self.ignored_root_modules()
            )
        )
        self.group_count = len(groups)
        if self.group_count == 0:
            raise RuntimeError("DepGraph không tìm thấy pruning group an toàn")
        return self.graph

    def analyze_model(self) -> dict[str, int | float | None]:
        """Return parameter and operation counts for a 640px forward pass."""
        return model_complexity(self.model, self.example_input)

    @torch.no_grad()
    def validate_forward(self) -> dict[str, Any]:
        """Check class metadata and the decoded detector tensor after a forward pass."""
        output = self.model(self.example_input)
        prediction = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(prediction, torch.Tensor) or prediction.ndim != 3:
            raise RuntimeError(f"YOLO output không hợp lệ: {type(prediction).__name__}")
        class_count = int(getattr(self.detection_head, "nc"))
        expected_channels = 4 + class_count
        if prediction.shape[1] != expected_channels:
            raise RuntimeError(
                f"Output có {prediction.shape[1]} channels, cần {expected_channels} cho {class_count} lớp"
            )
        if len(self.model.names) != class_count:
            raise RuntimeError("Số tên lớp không khớp detection head")
        return {
            "class_count": class_count,
            "class_names": list(self.model.names.values()),
            "prediction_shape": list(prediction.shape),
        }

    def prune(self) -> torch.nn.Module:
        """Apply one or more group-magnitude structured pruning steps."""
        import torch_pruning as tp

        self.meta_pruner = tp.pruner.BasePruner(
            self.model,
            self.example_input,
            importance=create_importance(self.importance_name),
            pruning_ratio=self.pruning_ratio,
            iterative_steps=self.iterative_steps,
            ignored_layers=self.ignored_root_modules(),
            round_to=self.round_to,
            global_pruning=self.global_pruning,
        )
        for _ in range(self.iterative_steps):
            self.meta_pruner.step()
        return self.model

    def create_sparse_pruner(self, reg: float, alpha: int = 4) -> Any:
        """Create the vendored DepGraph group-sparsity pruner without pruning yet."""
        import torch_pruning as tp

        self.meta_pruner = tp.pruner.GroupNormPruner(
            self.model,
            self.example_input,
            importance=create_importance(self.importance_name),
            reg=reg,
            alpha=alpha,
            pruning_ratio=self.pruning_ratio,
            iterative_steps=self.iterative_steps,
            ignored_layers=self.ignored_root_modules(),
            round_to=self.round_to,
            global_pruning=self.global_pruning,
        )
        self.group_count = len(self.meta_pruner._groups)
        if self.group_count == 0:
            raise RuntimeError("GroupNormPruner không tìm thấy pruning group an toàn")
        return self.meta_pruner

    def save_pruned_model(self, path: Path) -> None:
        """Save the complete changed architecture in an Ultralytics checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        saved_model = copy.deepcopy(self.model).cpu().float().eval()
        saved_model.zero_grad(set_to_none=True)
        torch.save(
            {
                "model": saved_model,
                "train_args": getattr(saved_model, "args", {}),
                "pruning_ratio": self.pruning_ratio,
                "torch_pruning_version": __import__("torch_pruning").__version__,
            },
            path,
        )

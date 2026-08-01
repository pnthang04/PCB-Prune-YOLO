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
        self.graph: Any | None = None
        self.group_count = 0
        self.meta_pruner: Any | None = None
        self.replaced_c2f = replace_c2f(self.model)
        self.head_name, self.detection_head = self._find_detection_head()
        self.protected_modules = self._find_protected_modules()

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

    def build_dependency_graph(self) -> Any:
        """Trace dependencies and enumerate groups without changing any channels."""
        import torch_pruning as tp

        self.graph = tp.DependencyGraph().build_dependency(
            self.model, example_inputs=self.example_input
        )
        # Materialize groups now so tracing/API errors fail during dry-run.
        groups = list(
            self.graph.get_all_groups(
                ignored_layers=[module for _, module in self.protected_modules]
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
            ignored_layers=[module for _, module in self.protected_modules],
            round_to=self.round_to,
            global_pruning=self.global_pruning,
        )
        for _ in range(self.iterative_steps):
            self.meta_pruner.step()
        return self.model

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

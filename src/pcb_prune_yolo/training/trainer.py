"""Ultralytics training wrapper."""

from pathlib import Path
from typing import Any

import json

import torch


def train(config: dict[str, Any]) -> Any:
    """Train a YOLO model using an explicit configuration mapping."""
    from ultralytics import YOLO

    options = dict(config)
    model_name = options.pop("model")
    if "project" in options:
        options["project"] = str(Path(options["project"]).resolve())
    if options.get("device") == "auto":
        options["device"] = None
    model = YOLO(model_name)
    return model.train(**options)


def train_pruned(config: dict[str, Any]) -> Any:
    """Fine-tune an in-memory changed architecture without rebuilding from YAML."""
    from ultralytics import YOLO

    options = dict(config)
    model_path = str(options.pop("model"))
    if "," in str(options.get("device", "")):
        raise ValueError("Fine-tune kiến trúc pruned hiện hỗ trợ một GPU trong mỗi process")
    if "project" in options:
        options["project"] = str(Path(options["project"]).resolve())
    yolo = YOLO(model_path)
    overrides = {
        **options,
        "model": model_path,
        "task": "detect",
        "mode": "train",
    }
    trainer = yolo._smart_load("trainer")(overrides=overrides, _callbacks=yolo.callbacks)
    trainer.model = yolo.model
    trainer.train()
    return getattr(trainer.validator, "metrics", None)


def train_sparse(config: dict[str, Any]) -> Any:
    """Sparse-train an unpruned YOLO checkpoint with DepGraph group regularization."""
    from ultralytics import YOLO

    from pcb_prune_yolo.config import save_config
    from pcb_prune_yolo.pruning.c2f import replace_c2f

    options = dict(config)
    model_path = str(options.pop("checkpoint"))
    sparse = options.pop("sparse")
    options.pop("provenance", None)
    if "," in str(options.get("device", "")):
        raise ValueError("Sparse trainer hiện hỗ trợ một GPU để giữ đúng model object và DepGraph")
    if "project" in options:
        options["project"] = str(Path(options["project"]).resolve())

    yolo = YOLO(model_path)
    replace_c2f(yolo.model)
    overrides = {**options, "model": model_path, "task": "detect", "mode": "train"}
    trainer = SparseDetectionTrainer(
        overrides=overrides,
        _callbacks=yolo.callbacks,
        sparse_config=sparse,
    )
    trainer.model = yolo.model
    save_config(config, trainer.save_dir / "used_config.yaml")
    trainer.train()
    return getattr(trainer.validator, "metrics", None)


def train_qat(config: dict[str, Any]) -> Any:
    """QAT-fine-tune the exact in-memory P30 object with train-only calibration."""
    from ultralytics import YOLO

    from pcb_prune_yolo.config import save_config
    from pcb_prune_yolo.quantization import prepare_int8_qat

    options = dict(config)
    model_path = str(options.pop("checkpoint"))
    quantization = options.pop("quantization")
    options.pop("provenance", None)
    if "," in str(options.get("device", "")):
        raise ValueError("QAT smoke hiện hỗ trợ một GPU để giữ đúng model object")
    if "project" in options:
        options["project"] = str(Path(options["project"]).resolve())

    yolo = YOLO(model_path)
    overrides = {**options, "model": model_path, "task": "detect", "mode": "train"}
    trainer = QATDetectionTrainer(overrides=overrides, _callbacks=yolo.callbacks)
    trainer.model = yolo.model
    prepare_int8_qat(
        model=trainer.model,
        data_file=Path(options["data"]),
        quantization=quantization,
        output_dir=trainer.save_dir / "quantization",
        device=torch.device("cuda:0"),
        imgsz=int(options["imgsz"]),
    )
    save_config(config, trainer.save_dir / "used_config.yaml")
    trainer.train()
    return getattr(trainer.validator, "metrics", None)


def train_gated(config: dict[str, Any]) -> Any:
    """Train Hard-Concrete gates with detection, native KD and an L0 constraint."""
    from ultralytics import YOLO

    from pcb_prune_yolo.config import save_config
    from pcb_prune_yolo.pruning.dependency_pruner import YOLODepGraphPruner
    from pcb_prune_yolo.pruning.gated_groups import build_gated_registry

    options = dict(config)
    model_path = str(options.pop("checkpoint"))
    gated = options.pop("gated")
    if "," in str(options.get("device", "")):
        raise ValueError("Gated pruning hiện hỗ trợ một GPU trong mỗi process")
    if "project" in options:
        options["project"] = str(Path(options["project"]).resolve())
    yolo = YOLO(model_path)
    device = torch.device("cpu")
    example = torch.randn(1, 3, int(options["imgsz"]), int(options["imgsz"]), device=device)
    wrapper = YOLODepGraphPruner(
        model=yolo.model,
        example_input=example,
        pruning_ratio=float(gated["target_sparsity"]),
        importance="group_magnitude",
        iterative_steps=1,
        round_to=None,
        global_pruning=False,
    )
    registry = build_gated_registry(
        wrapper,
        init_drop_rate=float(gated.get("init_drop_rate", 0.01)),
        min_channels=int(gated.get("min_channels", 8)),
    )
    overrides = {**options, "model": model_path, "task": "detect", "mode": "train"}
    trainer = GatedDetectionTrainer(
        overrides=overrides,
        _callbacks=yolo.callbacks,
        gated_registry=registry,
        gated_config=gated,
    )
    trainer.model = yolo.model
    save_config(config, trainer.save_dir / "used_config.yaml")
    trainer.train()
    return getattr(trainer.validator, "metrics", None)


class QATDetectionTrainerMixin:
    """Record smoke-test gradient, update and finite-loss diagnostics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._qat_optimizer_steps = 0
        self._qat_gradient_exists = False
        self._qat_gradient_finite = True
        self._qat_weight_updated = False
        super().__init__(*args, **kwargs)

    def optimizer_step(self) -> None:
        parameter = next((p for p in self.model.parameters() if p.requires_grad), None)
        before = parameter.detach().clone() if parameter is not None and not self._qat_weight_updated else None
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self._qat_gradient_exists = self._qat_gradient_exists or bool(gradients)
        self._qat_gradient_finite = self._qat_gradient_finite and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        super().optimizer_step()
        self._qat_optimizer_steps += 1
        if before is not None:
            self._qat_weight_updated = not torch.equal(before, parameter.detach())

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        super().save_metrics(metrics)
        numeric = [float(value) for key, value in metrics.items() if "loss" in key]
        report = {
            "epoch": self.epoch + 1,
            "optimizer_steps": self._qat_optimizer_steps,
            "gradient_exists": self._qat_gradient_exists,
            "gradient_finite": self._qat_gradient_finite,
            "weight_updated": self._qat_weight_updated,
            "loss_finite": all(torch.isfinite(torch.tensor(value)) for value in numeric),
            "loss_values": numeric,
        }
        path = self.save_dir / "qat_diagnostics.json"
        history = json.loads(path.read_text()) if path.exists() else []
        history.append(report)
        path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    def save_model(self) -> bool:
        """Persist dynamic quantizers with ModelOpt's supported state-dict format."""
        import modelopt.torch.opt as mto

        self.wdir.mkdir(parents=True, exist_ok=True)
        mto.save(self.model, self.last)
        if self.best_fitness == self.fitness:
            mto.save(self.model, self.best)
        metadata = {
            "epoch": self.epoch + 1,
            "fitness": float(self.fitness),
            "best_fitness": float(self.best_fitness),
            "format": "NVIDIA ModelOpt state + PyTorch state_dict",
            "source_checkpoint": str(self.args.model),
        }
        (self.wdir / "checkpoint_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return True

    def final_eval(self) -> None:
        """Skip Ultralytics checkpoint stripping; ModelOpt checkpoints need explicit restore."""


class SparseDetectionTrainerMixin:
    """Insert DepGraph's gradient regularizer at the optimizer boundary."""

    def __init__(self, *args: Any, sparse_config: dict[str, Any], **kwargs: Any) -> None:
        self.sparse_config = sparse_config
        self.sparse_pruner = None
        self.sparse_wrapper = None
        self._regularizer_grad_nonzero = False
        self._regularizer_grad_delta_l2 = 0.0
        self._measure_regularizer_grad = True
        self._regularizer_new_nonfinite = 0
        self._all_sparse_groups = []
        self._degenerate_sparse_groups = 0
        super().__init__(*args, **kwargs)

    def _setup_train(self) -> None:
        super()._setup_train()
        from pcb_prune_yolo.pruning.dependency_pruner import YOLODepGraphPruner

        model = self.model.module if hasattr(self.model, "module") else self.model
        example = torch.randn(1, 3, int(self.args.imgsz), int(self.args.imgsz), device=self.device)
        self.sparse_wrapper = YOLODepGraphPruner(
            model=model,
            example_input=example,
            pruning_ratio=float(self.sparse_config["pruning_ratio"]),
            importance="group_magnitude",
            iterative_steps=int(self.sparse_config["iterative_steps"]),
            round_to=self.sparse_config.get("round_to"),
            global_pruning=bool(self.sparse_config["global_pruning"]),
        )
        self.sparse_pruner = self.sparse_wrapper.create_sparse_pruner(
            reg=float(self.sparse_config["regularization_strength"]),
            alpha=int(self.sparse_config["alpha"]),
        )
        self._refresh_regularizer_groups()

    @torch.no_grad()
    def _refresh_regularizer_groups(self) -> None:
        """Skip groups for which TP 1.6.0's Eq. (5) denominator is zero."""
        self.sparse_pruner.update_regularizer()
        self._all_sparse_groups = list(self.sparse_pruner._groups)
        eligible = []
        for group in self._all_sparse_groups:
            importance = self.sparse_pruner.estimate_importance(group)
            if importance is None or not torch.isfinite(importance).all():
                continue
            norm = importance.detach().float().clamp_min(0).sqrt()
            if norm.numel() and float(norm.max() - norm.min()) > torch.finfo(norm.dtype).eps:
                eligible.append(group)
        self._degenerate_sparse_groups = len(self._all_sparse_groups) - len(eligible)
        self.sparse_pruner._groups = eligible

    def run_callbacks(self, event: str) -> None:
        if event == "on_train_epoch_start" and self.sparse_pruner is not None:
            self._refresh_regularizer_groups()
            self._regularizer_grad_nonzero = False
            self._regularizer_grad_delta_l2 = 0.0
            self._regularizer_new_nonfinite = 0
            self._measure_regularizer_grad = True
        super().run_callbacks(event)

    def optimizer_step(self) -> None:
        """Unscale, add sparse gradient, then clip and step the optimizer."""
        from ultralytics.utils.torch_utils import TORCH_2_0

        self.scaler.unscale_(self.optimizer)
        before = None
        if self._measure_regularizer_grad:
            before = {
                name: parameter.grad.detach().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.grad is not None
            }

        # Paper Eq. (5) uses alpha=4 as the exponent bound. In TP 1.6.0 this method's
        # alpha argument is the exponential base, so 2**4 yields gamma in [1, 16].
        self.sparse_pruner.regularize(
            self.model, alpha=2 ** int(self.sparse_config["alpha"])
        )
        if before is not None:
            delta_sq = 0.0
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None and name in before:
                    after = parameter.grad.detach()
                    finite_before = torch.isfinite(before[name])
                    finite_after = torch.isfinite(after)
                    self._regularizer_new_nonfinite += int((finite_before & ~finite_after).sum())
                    finite = finite_before & finite_after
                    delta_sq += float((after[finite] - before[name][finite]).square().sum())
            self._regularizer_grad_delta_l2 = delta_sq**0.5
            self._regularizer_grad_nonzero = self._regularizer_grad_delta_l2 > 0
            self._measure_regularizer_grad = False

        if self.device.type == "npu" and TORCH_2_0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0, foreach=False)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    @torch.no_grad()
    def _group_statistics(self) -> dict[str, float | int | bool]:
        values = []
        for group in self.sparse_pruner._groups:
            importance = self.sparse_pruner.estimate_importance(group)
            if importance is not None:
                norm = importance.detach().float().clamp_min(0).sqrt()
                values.append(norm[torch.isfinite(norm)].cpu())
        all_norms = torch.cat(values) if values else torch.empty(0)
        threshold = float(self.sparse_config["near_zero_threshold"])
        return {
            "sparse/groups": len(self._all_sparse_groups),
            "sparse/regularized_groups": len(self.sparse_pruner._groups),
            "sparse/degenerate_groups_skipped": self._degenerate_sparse_groups,
            "sparse/channels": int(all_norms.numel()),
            "sparse/norm_min": float(all_norms.min()) if all_norms.numel() else float("nan"),
            "sparse/norm_mean": float(all_norms.mean()) if all_norms.numel() else float("nan"),
            "sparse/norm_median": float(all_norms.median()) if all_norms.numel() else float("nan"),
            "sparse/norm_max": float(all_norms.max()) if all_norms.numel() else float("nan"),
            "sparse/near_zero_fraction": (
                float((all_norms <= threshold).float().mean()) if all_norms.numel() else float("nan")
            ),
            "sparse/regularizer_grad_nonzero": self._regularizer_grad_nonzero,
            "sparse/regularizer_grad_delta_l2": self._regularizer_grad_delta_l2,
            "sparse/regularizer_new_nonfinite": self._regularizer_new_nonfinite,
        }

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        sparse_metrics = self._group_statistics()
        merged = {**metrics, **sparse_metrics}
        super().save_metrics(merged)
        report_path = self.save_dir / "sparse_metrics.json"
        history = json.loads(report_path.read_text()) if report_path.exists() else []
        history.append({"epoch": self.epoch + 1, **{k: _json_value(v) for k, v in merged.items()}})
        report_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


class GatedDetectionTrainerMixin:
    """Add DiariZen's augmented-Lagrangian L0 constraint at optimizer step."""

    def __init__(
        self,
        *args: Any,
        gated_registry: Any,
        gated_config: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.gated_registry = gated_registry
        self.gated_config = gated_config
        self.lambda1 = None
        self.lambda2 = None
        self._gate_loss = torch.tensor(0.0)
        self._gate_target = 0.0
        super().__init__(*args, **kwargs)

    def _setup_train(self) -> None:
        super()._setup_train()
        for group in self.gated_registry.groups:
            group.costs = group.costs.to(group.gate.log_alpha.device)

    def build_optimizer(self, *args: Any, **kwargs: Any) -> Any:
        optimizer = super().build_optimizer(*args, **kwargs)
        gate_parameters = self.gated_registry.gate_parameters()
        gate_ids = {id(parameter) for parameter in gate_parameters}
        for group in optimizer.param_groups:
            group["params"] = [parameter for parameter in group["params"] if id(parameter) not in gate_ids]
        device = gate_parameters[0].device
        self.lambda1 = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.lambda2 = torch.nn.Parameter(torch.tensor(0.0, device=device))
        reg_lr = float(self.gated_config.get("reg_lr", 0.02))
        optimizer.add_param_group(
            {"params": gate_parameters, "lr": reg_lr, "weight_decay": 0.0, "param_group": "gates"}
        )
        optimizer.add_param_group(
            {
                "params": [self.lambda1, self.lambda2],
                "lr": -reg_lr,
                "weight_decay": 0.0,
                "param_group": "gate_lambda",
            }
        )
        return optimizer

    def _scheduled_target(self) -> float:
        pretrain = int(self.gated_config.get("pretrain_epochs", 0))
        warmup = max(1, int(self.gated_config.get("sparsity_warmup_epochs", 5)))
        progress = min(1.0, max(0.0, (self.epoch + 1 - pretrain) / warmup))
        return float(self.gated_config["target_sparsity"]) * progress

    def optimizer_step(self) -> None:
        self._gate_target = self._scheduled_target()
        expected = self.gated_registry.expected_sparsity()
        difference = expected - self._gate_target
        self._gate_loss = self.lambda1 * difference + self.lambda2 * difference.square()
        self.scaler.scale(self._gate_loss).backward()
        super().optimizer_step()

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        merged = {
            **metrics,
            "gated/target_sparsity": self._gate_target,
            "gated/expected_sparsity": float(self.gated_registry.expected_sparsity().detach()),
            "gated/sparsity_loss": float(self._gate_loss.detach()),
            "gated/lambda1": float(self.lambda1.detach()),
            "gated/lambda2": float(self.lambda2.detach()),
        }
        super().save_metrics(merged)
        path = self.save_dir / "gated_metrics.json"
        history = json.loads(path.read_text()) if path.exists() else []
        history.append({"epoch": self.epoch + 1, **{k: _json_value(v) for k, v in merged.items()}})
        path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    if hasattr(value, "item"):
        return value.item()
    return value


def _sparse_trainer_class() -> type:
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    return type("SparseDetectionTrainer", (SparseDetectionTrainerMixin, DetectionTrainer), {})


SparseDetectionTrainer = _sparse_trainer_class()


def _qat_trainer_class() -> type:
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    return type("QATDetectionTrainer", (QATDetectionTrainerMixin, DetectionTrainer), {})


QATDetectionTrainer = _qat_trainer_class()


def _gated_trainer_class() -> type:
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    return type("GatedDetectionTrainer", (GatedDetectionTrainerMixin, DetectionTrainer), {})


GatedDetectionTrainer = _gated_trainer_class()

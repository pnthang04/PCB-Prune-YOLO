"""Torch-Pruning importance selection."""


def create_importance(name: str):
    """Create a supported Torch-Pruning importance estimator."""
    import torch_pruning as tp

    if name == "group_magnitude":
        return tp.importance.GroupMagnitudeImportance(p=2)
    raise ValueError(f"Importance chưa được hỗ trợ: {name}")


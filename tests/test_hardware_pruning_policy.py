"""Unit tests for hardware-aware pruning policy helpers."""

from torch import nn

from pcb_prune_yolo.pruning.dependency_pruner import YOLODepGraphPruner


def test_channel_audit_reports_alignment_and_changes() -> None:
    pruner = object.__new__(YOLODepGraphPruner)
    pruner.model = nn.Sequential(nn.Conv2d(3, 16, 1), nn.Conv2d(16, 24, 1))
    pruner.hardware_policy = "standard"
    pruner.round_to = 8
    pruner.protected_modules = []
    pruner.block_protected_modules = []
    pruner.initial_channel_layout = {"0": 32, "1": 24}

    audit = pruner.channel_audit()

    assert audit["changed_convolution_outputs"] == 1
    assert audit["all_prunable_outputs_aligned_to_8"] is True
    assert audit["changed_channels"]["0"] == {"before": 32, "after": 16}


def test_ignored_roots_include_detection_and_block_modules() -> None:
    pruner = object.__new__(YOLODepGraphPruner)
    detection = nn.Conv2d(4, 4, 1)
    block = nn.Conv2d(4, 4, 1)
    pruner.protected_modules = [("detect", detection)]
    pruner.block_protected_modules = [("block", block)]

    assert pruner.ignored_root_modules() == [detection, block]

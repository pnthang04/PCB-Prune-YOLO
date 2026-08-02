"""Validation tests for HALP latency lookup tables."""

import json

import pytest

from pcb_prune_yolo.halp.lut import candidate_channels, load_lut, validate_lut


def valid_payload() -> dict:
    return {
        "records": [
            {
                "layer_name": "model.1.conv",
                "layer_type": "Conv2d",
                "input_channels": 16,
                "output_channels": 24,
                "height": 320,
                "width": 320,
                "kernel": 3,
                "stride": 2,
                "groups": 1,
                "precision": "fp16",
                "mean_latency_ms": 0.1,
                "median_latency_ms": 0.09,
                "p95_latency_ms": 0.12,
                "warmup_iterations": 50,
                "benchmark_iterations": 200,
                "status": "success",
                "reproducibility_relative_error": 0.02,
            }
        ],
        "latency_steps": [
            {"native_output_channels": 32, "proposed_group_size": 8}
        ],
    }


def test_load_and_validate_lut(tmp_path):
    path = tmp_path / "lut.json"
    path.write_text(json.dumps(valid_payload()), encoding="utf-8")
    validate_lut(load_lut(path))


def test_missing_field_is_rejected():
    payload = valid_payload()
    del payload["records"][0]["kernel"]
    with pytest.raises(ValueError, match="missing fields"):
        validate_lut(payload)


@pytest.mark.parametrize("field", ["mean_latency_ms", "median_latency_ms", "p95_latency_ms"])
def test_nonpositive_latency_is_rejected(field):
    payload = valid_payload()
    payload["records"][0][field] = 0
    with pytest.raises(ValueError, match=field):
        validate_lut(payload)


def test_invalid_channels_and_group_size_are_rejected():
    payload = valid_payload()
    payload["records"][0]["input_channels"] = 15
    payload["records"][0]["groups"] = 8
    with pytest.raises(ValueError, match="invalid channels"):
        validate_lut(payload)
    payload = valid_payload()
    payload["latency_steps"][0]["proposed_group_size"] = 64
    with pytest.raises(ValueError, match="group size"):
        validate_lut(payload)


def test_repeatability_threshold_is_enforced():
    payload = valid_payload()
    payload["records"][0]["reproducibility_relative_error"] = 0.21
    with pytest.raises(ValueError, match="reproducibility"):
        validate_lut(payload)


def test_candidate_grid_is_valid_and_includes_native_width():
    values = candidate_channels(70)
    assert values[-1] == 70
    assert all(0 < value <= 70 for value in values)
    assert all(value == 70 or value % 8 == 0 for value in values)

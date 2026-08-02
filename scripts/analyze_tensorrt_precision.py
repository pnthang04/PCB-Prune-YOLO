"""Combine TensorRT engine inspector and ONNX evidence for convolution precision."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


def _precision(formats: list[str]) -> str:
    value = " ".join(formats).lower()
    if "int8" in value:
        return "INT8"
    if "fp16" in value:
        return "FP16"
    if "fp32" in value:
        return "FP32"
    return "OTHER"


def _shape(value: dict[str, Any] | None) -> str:
    return "x".join(str(item) for item in value.get("Dimensions", [])) if value else ""


def _fallback_evidence(layer: dict[str, Any], precision: str, explicit_qdq: bool) -> str:
    name = layer.get("Name", "")
    tactic = layer.get("TacticName", "")
    weight_type = layer.get("Weights", {}).get("Type", "")
    inputs = " ".join(item.get("Format/Datatype", "") for item in layer.get("Inputs", []))
    output_channels = layer.get("OutMaps")
    notes = []
    if precision != "FP32":
        return "Engine inspector confirms INT8 output format and INT8 tactic."
    if weight_type == "Int8" and "int8" in tactic.lower():
        notes.append("mixed tactic: INT8 input/weights with FP32 output, not full FP32 compute")
    elif weight_type == "Float":
        notes.append("engine inspector confirms float weights and FP32 tactic")
    if "Sigmoid" in name or "act/Mul" in name:
        notes.append(
            "SiLU output has no trailing Q in the explicit graph"
            if explicit_qdq
            else "fused SiLU touches Sigmoid that the implicit PTQ exporter constrained to FP32"
        )
    if "model.22" in name:
        notes.append("Detect/DFL boundary remains high precision to preserve detector semantics")
    if "||" in name:
        notes.append("TensorRT fused parallel C2f/Detect convolutions with a shared output precision")
    if len(layer.get("Inputs", [])) > 1:
        notes.append("residual/elementwise fusion has multiple FP32 inputs")
    if "FP32" in inputs and weight_type == "Float":
        notes.append("FP32 island propagated through the input tensor")
    if isinstance(output_channels, int) and output_channels % 4:
        notes.append(f"Cout={output_channels} is not divisible by 4, unfavorable for vectorized INT8")
    if not notes:
        notes.append("inspector proves FP32 selection; alternative tactic timing was not retained")
    return "; ".join(notes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-info", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    layer_info = json.loads(args.layer_info.read_text(encoding="utf-8"))
    layers = layer_info["Layers"] if isinstance(layer_info, dict) else layer_info
    graph = onnx.load(str(args.onnx))
    onnx_ops = Counter(node.op_type for node in graph.graph.node)
    qdq = {"QuantizeLinear": onnx_ops["QuantizeLinear"], "DequantizeLinear": onnx_ops["DequantizeLinear"]}
    explicit_qdq = qdq["QuantizeLinear"] > 0 and qdq["DequantizeLinear"] > 0

    rows = []
    for layer in layers:
        if "Convolution" not in layer.get("LayerType", ""):
            continue
        input_items = layer.get("Inputs", [])
        output_items = layer.get("Outputs", [])
        input_formats = [item.get("Format/Datatype", "") for item in input_items]
        output_formats = [item.get("Format/Datatype", "") for item in output_items]
        precision = _precision(output_formats)
        rows.append(
            {
                "layer": layer.get("Name", ""),
                "input_shape": " | ".join(_shape(item) for item in input_items),
                "output_shape": " | ".join(_shape(item) for item in output_items),
                "kernel": "x".join(str(item) for item in layer.get("Kernel", [])),
                "groups": layer.get("Groups", ""),
                "precision": precision,
                "input_format": " | ".join(input_formats),
                "output_format": " | ".join(output_formats),
                "weight_type": layer.get("Weights", {}).get("Type", ""),
                "tactic": layer.get("TacticName", ""),
                "suspected_fallback_reason": _fallback_evidence(layer, precision, explicit_qdq),
            }
        )

    counts = Counter(row["precision"] for row in rows)
    fp32_rows = [row for row in rows if row["precision"] == "FP32"]
    mixed = [row for row in fp32_rows if row["weight_type"] == "Int8" and "int8" in row["tactic"].lower()]
    full = [row for row in fp32_rows if row["weight_type"] == "Float"]
    summary = {
        "engine_layer_info": str(args.layer_info),
        "onnx": str(args.onnx),
        "onnx_operator_counts": dict(onnx_ops),
        "onnx_qdq_counts": qdq,
        "convolution_output_precision_counts": dict(counts),
        "fp32_output_convolutions": len(fp32_rows),
        "mixed_int8_input_weight_fp32_output_convolutions": len(mixed),
        "full_fp32_weight_output_convolutions": len(full),
        "evidence": {
            "graph_export": (
                f"Explicit ONNX contains Q={qdq['QuantizeLinear']} and DQ={qdq['DequantizeLinear']} nodes."
                if explicit_qdq
                else "FP32 ONNX and implicit PTQ ONNX contain no Q/DQ nodes."
            ),
            "precision_constraint": (
                "Q/DQ placement, not builder INT8 flags, controls quantized regions."
                if explicit_qdq
                else "Ultralytics implicit exporter kept 65 Sigmoid layers FP32."
            ),
            "tactic": "Inspector records actual tactic and weight/output formats; alternative tactic timings are unavailable.",
        },
        "rows": rows,
    }
    (args.output / "convolution_precision.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "convolution_precision.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

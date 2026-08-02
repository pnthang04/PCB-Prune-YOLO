"""Restore a ModelOpt QAT checkpoint and export explicit-Q/DQ ONNX and TensorRT."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import onnx
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--qat-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace", type=float, default=4.0)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    onnx_path = args.output / "model_qdq.onnx"
    engine_path = args.output / "model_qdq.engine"

    import modelopt
    import modelopt.torch.opt as mto
    from ultralytics import YOLO, __version__ as ultralytics_version

    device = torch.device(args.device)
    model = YOLO(str(args.base_checkpoint)).model
    model = mto.restore(model, args.qat_checkpoint).to(device).eval()
    head = model.model[-1]
    head.export = True
    head.format = "onnx"
    head.dynamic = False
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        prediction = model(dummy)
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    expected = [1, 4 + len(model.names), 8400]
    if not isinstance(prediction, torch.Tensor) or list(prediction.shape) != expected:
        raise RuntimeError(f"QAT output không hợp lệ trước export: {getattr(prediction, 'shape', None)}")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["images"],
        output_names=["output0"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    q_count = sum(node.op_type == "QuantizeLinear" for node in graph.graph.node)
    dq_count = sum(node.op_type == "DequantizeLinear" for node in graph.graph.node)
    if q_count < 1 or dq_count < 1:
        raise RuntimeError(f"ONNX không giữ explicit Q/DQ: Q={q_count}, DQ={dq_count}")

    metadata = {
        "description": "P30 direct ModelOpt INT8 QAT smoke explicit-Q/DQ",
        "author": "PCB-Prune-YOLO",
        "date": datetime.now().astimezone().isoformat(),
        "version": ultralytics_version,
        "license": "AGPL-3.0",
        "docs": "docs/P30_INT8_QAT_SMOKE_REPORT.md",
        "stride": int(model.stride.max()),
        "task": "detect",
        "head": "Detect",
        "batch": 1,
        "imgsz": [args.imgsz, args.imgsz],
        "names": model.names,
        "args": {
            "batch": 1,
            "quantize": 8,
            "dynamic": False,
            "workspace": args.workspace,
            "nms": False,
        },
        "channels": 3,
        "end2end": False,
    }
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace * (1 << 30)))
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT explicit-Q/DQ engine build failed")
    encoded_metadata = json.dumps(metadata).encode()
    with engine_path.open("wb") as stream:
        stream.write(len(encoded_metadata).to_bytes(4, byteorder="little", signed=True))
        stream.write(encoded_metadata)
        stream.write(serialized)

    report = {
        "base_checkpoint": str(args.base_checkpoint),
        "qat_checkpoint": str(args.qat_checkpoint),
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "input_shape": [1, 3, args.imgsz, args.imgsz],
        "output_shape": list(prediction.shape),
        "class_count": len(model.names),
        "onnx_quantize_linear": q_count,
        "onnx_dequantize_linear": dq_count,
        "explicit_qdq": True,
        "workspace_gib": args.workspace,
        "strongly_typed": True,
        "profiling_verbosity": "DETAILED",
        "modelopt_version": modelopt.__version__,
        "ultralytics_version": ultralytics_version,
    }
    (args.output / "export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

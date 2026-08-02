---
license: agpl-3.0
library_name: tensorrt
pipeline_tag: object-detection
tags: [yolo, yolov8, tensorrt, fp16, depgraph, deeppcb, tesla-t4]
datasets: [thangkt/PCB-Prune-YOLO-DeepPCB]
metrics: [map]
---

# PCB-Prune-YOLO TensorRT FP16

Static batch-1 TensorRT FP16 engines for the baseline and direct DepGraph P10,
P20, and P30 checkpoints. Engines were built and benchmarked directly on Tesla
T4 with TensorRT 10.16.1.11, CUDA 12.8, PyTorch 2.10.0+cu128, and Ultralytics
8.4.115. Input is `[1,3,640,640]`; raw output is `[1,10,8400]`; NMS is external.

| Model | Validation mAP50-95 | Mean latency | FPS | Engine size |
|---|---:|---:|---:|---:|
| Baseline | 0.78716 | 1.837 ms | 544.28 | 7.477 MiB |
| P10 direct | 0.77842 | 2.023 ms | 494.34 | 7.627 MiB |
| P20 direct | 0.76931 | 1.933 ms | 517.45 | 7.378 MiB |
| P30 direct | 0.75610 | 1.754 ms | 569.97 | 5.482 MiB |

Latency is pure synchronized engine forward after 50 warm-ups across 200
measurements and excludes preprocessing/NMS. TensorRT engines are not portable
across arbitrary TensorRT/CUDA/GPU combinations; rebuild them on the deployment
target when the stack differs. Full provenance and reports are included.

Project: https://github.com/pnthang04/PCB-Prune-YOLO

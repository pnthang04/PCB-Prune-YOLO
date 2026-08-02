# P30 TensorRT INT8 QAT smoke report

Ngày đo: 2026-08-02. Thí nghiệm chỉ dùng P30 direct đã fine-tune, training split
cho calibration/training và validation split để đánh giá. Không dùng test,
distillation hay HALP. Quyết định cuối: **`FIX_GRAPH_FIRST`**.

## Căn cứ và phân loại quyết định

- **OFFICIAL DOC** — TensorRT nhận graph có `QuantizeLinear/DequantizeLinear`
  là explicit quantization; Q/DQ quyết định nơi chuyển precision và không dùng
  calibration table ngoài. Activation INT8 dùng per-tensor, weight nên dùng
  per-channel. NVIDIA cũng cảnh báo explicit Q/DQ có thể chậm hơn nếu placement
  làm hỏng fusion. Nguồn: [TensorRT Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html).
- **OFFICIAL DOC** — Engine Inspector và detailed profiling verbosity là nguồn
  xác nhận format, tactic và precision của engine đã build, không suy ra chỉ từ
  tên ONNX. Nguồn: [TensorRT Engine Inspector](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html).
- **OFFICIAL CODE** — NVIDIA Model Optimizer 0.45.0 `INT8_DEFAULT_CFG` cho CNN:
  symmetric INT8, weight axis 0/per-channel, activation per-tensor; checkpoint
  động được lưu/khôi phục bằng `modelopt.torch.opt.save/restore`. Nguồn:
  [ModelOpt quantization](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html)
  và [saving/restoring](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html).
- **PROJECT ADAPTATION** — Smoke 3 epoch, AdamW, `lr0=1e-4`, batch 32, AMP tắt,
  max calibration trên 64 ảnh train seed 42. Sáu regression/classification
  output convolutions và DFL bị loại khỏi fake quant để giữ detector semantics.
- **ASSUMPTION** — Ba epoch chỉ đủ kiểm tra pipeline và xu hướng phục hồi, không
  đại diện accuracy QAT cuối. Một lần đo trên một T4 không đại diện mọi T4.
- **TODO** — Nếu quay lại INT8, sửa Q/DQ placement quanh SiLU/residual/concat,
  fuse Conv-BN trước export và đo từng graph change. Không train QAT dài trước
  khi engine forward nhanh hơn FP16.

## Vì sao 26 convolution PTQ có output FP32?

Đếm cũ dựa trên output format của 61 TensorRT convolution. Engine Inspector
cho thấy 26 lớp này chia thành hai nhóm khác nhau:

- 13 lớp dùng INT8 input, INT8 weight và tactic có tên `int8`, nhưng trả output
  FP32. Đây là mixed-precision convolution, không phải toàn bộ phép tính FP32.
- 13 lớp dùng float weight, FP32 input/output và FP32 tactic; đây mới là FP32
  island hoàn chỉnh.

ONNX nguồn PTQ không có Q/DQ (`Q=0`, `DQ=0`), nên đây là implicit
quantization. Ultralytics đặt precision constraint FP32 cho 65 Sigmoid; vì SiLU
là `x * sigmoid(x)`, constraint lan tới fusion boundary. Inspector còn cho
thấy các FP32 island quanh residual/concat, Detect terminal output và DFL.

Các bằng chứng loại trừ một số giả thuyết:

- Không phải thiếu calibration toàn cục: 13 lớp đã có INT8 weight/input và INT8
  tactic nhưng cần FP32 output.
- Không thể kết luận TensorRT chọn FP32 chỉ vì nhanh hơn: inspector lưu tactic
  đã chọn nhưng implicit build cũ không lưu toàn bộ alternative timing.
- Channel lẻ như 89/179 làm vectorization kém thuận lợi, nhưng không phải nguyên
  nhân duy nhất vì một số channel lẻ vẫn có mixed INT8 tactic.
- Resize, Sigmoid, residual/concat và Detect/DFL tạo precision/fusion boundary
  có bằng chứng trực tiếp trong engine layer graph.

Bảng đủ 61 convolution gồm shape, kernel, groups, format, weight type, tactic và
lý do nghi ngờ nằm tại
`outputs/deployment_optimization/qat_smoke/analysis/ptq_precision/convolution_precision.csv`.

## Smoke QAT

Config: `configs/qat/p30_int8_qat.yaml`. ModelOpt chèn 348 quantizer module,
trong đó 131 được bật trước khi Ultralytics chuẩn bị train. Calibration manifest
chứa 64 ảnh duy nhất, tất cả nằm trong `images/train`.

Lần chạy đầu phát hiện Ultralytics không pickle được dynamic `QuantConv2d`.
Pipeline được sửa theo API chính thức ModelOpt và chạy lại bằng output mới,
không ghi đè artifact. Run hợp lệ có:

- 3 epoch, 75 optimizer step;
- gradient tồn tại và hữu hạn;
- weight thay đổi;
- toàn bộ loss hữu hạn;
- validation trong train tăng mAP50-95 từ 0.08039 → 0.64645 → 0.72092;
- ModelOpt save → process mới restore → CUDA inference thành công;
- sáu class và output `[1,10,8400]` được giữ nguyên.

Checkpoint hợp lệ nằm tại
`outputs/qat/p30_int8_qat_smoke_e3_v2/weights/best.pt` và dùng định dạng
ModelOpt state + PyTorch state dict, không phải checkpoint Ultralytics thông
thường.

## Explicit Q/DQ export và precision coverage

ONNX opset 18 chứa 133 `QuantizeLinear` và 133 `DequantizeLinear`; TensorRT
10.16.1.11 parse trực tiếp thành network strongly typed, workspace 4 GiB,
static `[1,3,640,640]`, batch 1. Không dùng INT8 builder flag hoặc calibration
cache. Engine load/inference thành công.

| Engine | INT8-output conv | Tổng conv | Tỷ lệ | Mixed INT8→FP32 | Full FP32 |
|---|---:|---:|---:|---:|---:|
| P30 PTQ implicit | 35 | 61 | 57.38% | 13 | 13 |
| P30 QAT smoke explicit | 38 | 68 | 55.88% | 23 | 7 |

QAT giảm full-FP32 convolution từ 13 xuống 7, nhưng Conv-BN không fuse như
engine PTQ nên tổng convolution tăng 61 → 68. INT8-output tăng tuyệt đối chỉ 3
lớp và tỷ lệ còn giảm nhẹ. Vì vậy gate “coverage tăng rõ” **không đạt**.

So với PTQ, QAT engine có 204 layer thay vì 197, reformat 77 thay vì 72. Nsight
50 iteration đếm 3,673 `cudaLaunchKernel`, cao hơn 3,571 của PTQ. Explicit Q/DQ
được TensorRT giữ đúng semantics nhưng placement tạo thêm quantize/reformat và
không giải quyết launch overhead.

## Validation và benchmark

Tất cả latency dùng cùng runtime reuse execution context/stream/device buffers,
pinned host buffer, warm-up 50 và 200 lần đo. E2E gồm preprocess, H2D, inference
và NMS, không gồm disk/load engine.

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| P30 FP16 | 0.96301 | 0.92858 | 0.97700 | 0.75610 |
| P30 INT8 PTQ | 0.94393 | 0.92083 | 0.96978 | 0.61119 |
| P30 INT8 QAT smoke | 0.96343 | 0.92149 | 0.96983 | 0.72462 |

QAT smoke phục hồi 11.34 điểm phần trăm mAP50-95 so với PTQ, nhưng còn thấp
hơn FP16 3.15 điểm.

| Model | Forward mean/median/p95 | E2E mean/median/p95 | E2E FPS | Engine |
|---|---:|---:|---:|---:|
| P30 FP16 graph off | 1.8134/1.4946/3.1564 ms | 8.2640/8.0317/10.6907 ms | 121.01 | 5.482 MiB |
| P30 FP16 CUDA Graph | 1.7170/1.4893/2.7337 ms | 8.2768/8.0927/10.3485 ms | 120.82 | 5.482 MiB |
| P30 INT8 PTQ | 1.9047/1.7076/3.4931 ms | 8.7337/8.6998/11.3078 ms | 114.50 | 5.860 MiB |
| P30 INT8 QAT smoke graph off | 1.8896/1.7344/2.1640 ms | 8.9817/8.8427/11.1779 ms | 111.34 | 5.186 MiB |
| P30 INT8 QAT smoke CUDA Graph | 1.7621/1.7533/1.8010 ms | 8.9616/8.6533/11.9335 ms | 111.59 | 5.186 MiB |

Graph-off QAT chậm hơn FP16 4.20% forward và 8.68% E2E. Khi cùng bật CUDA
Graph, QAT vẫn chậm hơn FP16 2.63% forward. QAT cải thiện accuracy so với PTQ,
nhưng chưa tạo tiềm năng latency để biện minh cho training dài.

## Gate cuối

- Coverage tăng rõ: **không**.
- Forward nhanh hơn P30 FP16: **không**.
- Q/DQ giữ đúng: **có**, 133 cặp.
- Accuracy tốt hơn PTQ: **có**, +11.34 điểm mAP50-95.
- Reformat/kernel launch không tăng mạnh: **không**, đều tăng.

Không chạy full QAT và không thêm distillation. Distillation chỉ có thể hỗ trợ
accuracy; nó không sửa graph fusion, tactic hay latency.

**`FIX_GRAPH_FIRST`**

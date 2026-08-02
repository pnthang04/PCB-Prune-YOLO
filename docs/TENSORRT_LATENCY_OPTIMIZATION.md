# TensorRT latency optimization: baseline vs direct P30

Ngày đo: 2026-08-02. Nhánh thí nghiệm này chỉ dùng baseline và direct P30;
không sửa checkpoint/engine cũ, không chạy HALP và không dùng test set.

## Điều kiện

- Tesla T4, TensorRT 10.16.1.11, CUDA 12.8, Ultralytics 8.4.115.
- Static `[1,3,640,640]`, batch 1, engine không chứa NMS.
- Warm-up 50, đo 200 lần, CUDA synchronize; build/load engine và đọc ảnh từ
  disk không nằm trong latency.
- Nsight Systems 2024.6.2 và `trtexec` đúng TensorRT 10.16.1.11.

## Profile FP16 hiện trạng

Trong benchmark matched hiện tại, baseline/P30 có forward mean lần lượt
1.7259/1.7743 ms. E2E gồm preprocess, H2D, inference và NMS là
4.5377/4.4495 ms: P30 nhanh hơn 1.94% ở E2E nhưng chậm hơn ở forward trong lần
đo này. Không được diễn giải khác biệt E2E là tăng tốc kiến trúc vì số candidate
đi vào NMS phụ thuộc output model.

`trtexec --dumpProfile` cho thấy P30 giảm GPU-compute median từ 1.8214 xuống
1.4404 ms, nhưng enqueue median gần như không đổi ở khoảng 0.992 ms. Nsight
đếm 3,571 kernel launch cho P30 so với 2,616 cho baseline; pointwise, tensor
reformat và kernel-launch overhead vẫn lớn. P30 giảm MACs nhưng graph bị phân
mảnh hơn, giải thích vì sao latency ứng dụng không giảm tương ứng.

## Runtime reuse và CUDA Graph

`scripts/benchmark_tensorrt_runtime.py` giữ nguyên một execution context, CUDA
stream, input/output GPU buffers và pinned host buffer; H2D dùng asynchronous
copy. CUDA Graph chỉ capture các launch thuộc engine vì preprocessing và NMS
hiện vẫn chạy ngoài graph.

| Model/runtime | Forward mean/median/p95 (ms) | E2E mean/median/p95 (ms) | E2E FPS |
|---|---:|---:|---:|
| Baseline FP16, graph off | 1.6884/1.4590/2.6200 | 7.6638/7.5541/9.7348 | 130.48 |
| Baseline FP16, graph on | 1.7755/1.4555/2.3954 | 7.7762/7.6781/9.6306 | 128.60 |
| P30 FP16, graph off | 1.8134/1.4946/3.1564 | 8.2640/8.0317/10.6907 | 121.01 |
| P30 FP16, graph on | 1.7170/1.4893/2.7337 | 8.2768/8.0927/10.3485 | 120.82 |

P30 CUDA Graph giảm forward mean 5.32% và p95 13.39%, nhưng E2E mean tăng
0.16%. Do đó graph chưa được chấp nhận là tối ưu E2E mặc định. `trtexec` cho
thấy enqueue median giảm khoảng 90% khi bật graph, nhưng production runner vẫn
bị preprocessing/NMS và đồng bộ từng request chi phối.

Hai nhóm E2E ở phần profile hiện trạng và runtime ablation dùng implementation
preprocess/NMS khác nhau; chỉ so sánh on/off trong cùng nhóm, không so sánh số
tuyệt đối chéo giữa hai nhóm.

## P30 TensorRT INT8 PTQ

Engine mới được calibrate bằng 500 ảnh chọn xác định (seed 42) chỉ từ 800 ảnh
training. Danh sách chính xác nằm ở
`outputs/deployment_optimization/int8_ptq/p30_direct/calibration_images.txt`.
TensorRT MINMAX calibration được dùng; 65 Sigmoid bị giữ FP32 theo exporter
Ultralytics để bảo vệ confidence calibration. Input/output vẫn là FP32 tại biên
và output đã xác minh `[1,10,8400]`.

| Model | Precision | Recall | mAP50 | mAP50-95 | Forward mean | E2E mean | E2E FPS | Engine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P30 FP16 | 0.96301 | 0.92858 | 0.97700 | 0.75610 | 1.8134 ms | 8.2640 ms | 121.01 | 5.482 MiB |
| P30 INT8 PTQ | 0.94393 | 0.92083 | 0.96978 | 0.61119 | 1.9047 ms | 8.7337 ms | 114.50 | 5.860 MiB |

Các latency trong bảng dùng cùng custom runtime, graph off. PTQ giảm 14.49
điểm phần trăm mAP50-95, forward chậm hơn 5.03% và E2E chậm hơn 5.68%. Engine
không fallback hoàn toàn, nhưng chỉ 35/61 convolution có output INT8; 26/61 còn
FP32 và không có convolution FP16. Chi tiết từng layer nằm trong
`precision_layers.csv` và `precision_summary.json`.

## Quyết định

INT8 PTQ hiện tại không dùng để triển khai. Accuracy loss vượt xa gate 2 điểm
phần trăm và latency cũng không tốt hơn FP16. Bước tiếp theo hợp lệ là một thí
nghiệm QAT riêng bắt đầu từ P30 fine-tuned, giữ nguyên kiến trúc và chưa dùng
distillation. Chỉ cân nhắc teacher distillation nếu QAT thông thường không phục
hồi đủ validation accuracy. Trước QAT nên kiểm tra explicit Q/DQ để tránh phần
lớn graph bị ép FP32 quanh Sigmoid.

Artifact máy cục bộ và báo cáo JSON/CSV nằm ở
`outputs/deployment_optimization/`. Engine, raw plan và Nsight binary report bị
giữ ngoài Git; các báo cáo nhẹ, calibration manifest và layer precision được
giữ trong Git.

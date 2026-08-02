# PCB-Prune-YOLO: DepGraph Pruning for PCB Defect Detection

Phạm Ngọc Thắng

![Task](https://img.shields.io/badge/Task-Object_Detection-c0392b)
![Dataset](https://img.shields.io/badge/Dataset-DeepPCB-d35400)
![Model](https://img.shields.io/badge/Model-YOLOv8n-55a630)
![Language](https://img.shields.io/badge/Language-Python-3776ab)

**Quick Links:** [📦 Dataset](https://huggingface.co/datasets/thangkt/PCB-Prune-YOLO-DeepPCB) | [🤗 Baseline](https://huggingface.co/thangkt/PCB-Prune-YOLO-Baseline) | [🤗 P10 direct](https://huggingface.co/thangkt/PCB-Prune-YOLO-P10-Direct) | [🤗 P20 direct](https://huggingface.co/thangkt/PCB-Prune-YOLO-P20-Direct) | [🤗 P30 direct](https://huggingface.co/thangkt/PCB-Prune-YOLO-P30-Direct) | [⚡ TensorRT FP16](https://huggingface.co/thangkt/PCB-Prune-YOLO-TensorRT-FP16) | [🤗 P10 sparse](https://huggingface.co/thangkt/PCB-Prune-YOLO-P10-DepGraph) | [🤗 P40-A8 direct](https://huggingface.co/thangkt/PCB-Prune-YOLO-P40-A8-Direct) | [🤗 P40-A8 KD](https://huggingface.co/thangkt/PCB-Prune-YOLO-P40-A8-KD)

Hướng dẫn tự động chạy trên server: [`SERVER_RUNBOOK.md`](SERVER_RUNBOOK.md).

Project huấn luyện YOLOv8n phát hiện sáu loại lỗi PCB và nghiên cứu structured
channel pruning bằng DepGraph/Torch-Pruning. Baseline đã hoàn tất; P10 direct và
P10 sau group-level sparse learning đã được đánh giá trước fine-tune.

## Cấu trúc

```text
configs/                 Cấu hình dữ liệu, huấn luyện và benchmark
data/raw/DeepPCB/        Dataset gốc, chỉ đọc
data/processed/deeppcb/  Dữ liệu YOLO đã xử lý
outputs/                 Checkpoint, preview và báo cáo
scripts/                 Các lệnh của pipeline
src/pcb_prune_yolo/      Mã nguồn chính
tests/                   Unit test
```

## Đưa project lên server

Từ máy local, thay `USER` và `SERVER_IP` bằng thông tin server:

```bash
scp -r PCB-Prune-YOLO USER@SERVER_IP:~/PCB-Prune-YOLO
```

Trên server:

```bash
cd ~/PCB-Prune-YOLO
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Môi trường

Yêu cầu Python 3.10 trở lên. Chạy `nvidia-smi`, sau đó dùng bản PyTorch có CUDA tương thích với driver của server. Không cài đè PyTorch trước khi xác nhận CUDA. `requirements.txt` cố ý không chứa PyTorch.

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/check_environment.py --require-gpus 2
```

## Chuẩn bị dữ liệu

Dataset gốc phải nằm tại `data/raw/DeepPCB`. Script chỉ dùng ảnh kiểm tra có hậu tố `_test.jpg`; ảnh template không được đưa vào dữ liệu huấn luyện. Split test chính thức được giữ nguyên. Một nghìn ảnh thuộc split train gốc được chia train/val theo tỷ lệ 80/20 với seed 42.

Nếu chạy trên server, tải trực tiếp dữ liệu đã xử lý từ [PCB-Prune-YOLO-DeepPCB](https://huggingface.co/datasets/thangkt/PCB-Prune-YOLO-DeepPCB):

```bash
curl -L -o deeppcb_processed.zip https://huggingface.co/datasets/thangkt/PCB-Prune-YOLO-DeepPCB/resolve/main/deeppcb_processed.zip
mkdir -p data/processed
unzip deeppcb_processed.zip -d data/processed
```

Sau khi giải nén, dữ liệu phải nằm tại `data/processed/deeppcb`; không cần chạy lại bước chuyển đổi.

```bash
python scripts/prepare_deeppcb.py
python scripts/validate_dataset.py
python scripts/visualize_annotations.py --count 20
```

Các lớp sau khi chuyển đổi:

| ID | Lớp |
|---:|---|
| 0 | open |
| 1 | short |
| 2 | mousebite |
| 3 | spur |
| 4 | copper |
| 5 | pin-hole |

## Cấu hình baseline

Cấu hình dữ liệu nằm tại `configs/data/deeppcb.yaml`; cấu hình huấn luyện nằm tại `configs/train/yolov8n_baseline.yaml`.

Profile mặc định dành cho 2×T4: ảnh 640 px, batch tổng 128 (64 ảnh/GPU), 100 epoch, early stopping patience 20, AMP, seed 42, deterministic mode và `device: 0,1`. Batch này dùng khoảng 7.63 GiB/GPU trong smoke test thực tế trên hai T4 14.56 GiB; nếu thiếu VRAM, giảm `--batch` xuống 64.

## Huấn luyện

Smoke test chạy 1 epoch trên 20% tập train để kiểm tra nhanh pipeline DDP với khoảng hai training step:

```bash
python scripts/train_baseline.py --smoke --batch 128 --fraction 0.2
```

Full baseline chỉ chạy khi chủ động gọi:

```bash
python scripts/train_baseline.py
```

## Kết quả baseline

Mô hình tốt nhất được chọn trên validation tại epoch 98/100:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.96545 | 0.97221 | 0.98630 | 0.78524 |

Đây là kết quả trên validation; test set chỉ được dùng cho báo cáo cuối. Checkpoint và lịch sử huấn luyện được phát hành công khai tại [thangkt/PCB-Prune-YOLO-Baseline](https://huggingface.co/thangkt/PCB-Prune-YOLO-Baseline).

Tải checkpoint trực tiếp:

```bash
curl -L -o best.pt https://huggingface.co/thangkt/PCB-Prune-YOLO-Baseline/resolve/main/best.pt
```

Load bằng Ultralytics:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("pcb.jpg")
```

## Đánh giá và benchmark

Không dùng test set để chọn hyperparameter. Đánh giá validation trước; chỉ đánh giá test cho báo cáo cuối cùng.

```bash
python scripts/evaluate_model.py --checkpoint outputs/train/baseline/weights/best.pt --split val --device 0
python scripts/evaluate_model.py --checkpoint outputs/train/baseline/weights/best.pt --split test --device 0
python scripts/benchmark_model.py --model outputs/train/baseline/weights/best.pt --device cuda:0
```

Đánh giá xuất precision, recall, mAP50 và mAP50-95 tổng thể/theo lớp dưới dạng JSON và CSV. Benchmark batch size 1 xuất số tham số, dung lượng checkpoint, FPS, mean/median/p95 latency và peak GPU memory khi có CUDA.

## DepGraph pruning

Pipeline hiện tại:

```text
baseline best.pt → group-level sparse training → P10 structured pruning
→ validation → benchmark → save/load process mới
```

Sparse training giữ nguyên YOLO detection loss và thêm gradient regularization
của `GroupNormPruner` sau backward, trước optimizer step. Chi tiết paper/API và
phân loại PAPER/OFFICIAL CODE/ADAPTATION nằm trong
[`docs/DEPGRAPH_SPARSE.md`](docs/DEPGRAPH_SPARSE.md).

```bash
python scripts/train_sparse.py --config configs/prune/depgraph_sparse.yaml

python scripts/prune_model.py \
  --checkpoint outputs/sparse/depgraph_sparse_p10/weights/best.pt \
  --pruning-ratio 0.10 --round-to 0 \
  --output outputs/pruning_sparse --no-dry-run
```

Kết quả validation trước fine-tune:

| Model | Sparse learning | Params | MACs | mAP50 | mAP50-95 | Latency | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | No | 3,012,018 | 4.0733G | 0.98630 | 0.78524 | 8.289 ms | 120.64 |
| P10 direct, no round | No | 2,416,871 | 3.2695G | 0.00586 | 0.000923 | TODO | TODO |
| P10 sparse, no round | Yes | 2,415,613 | 3.2328G | 0.002615 | 0.000375 | 9.737 ms | 102.71 |
| P10 sparse reg=5e-4, trước FT | Yes | 2,415,613 | 3.2328G | 0.002427 | 0.000351 | 10.127 ms | 98.75 |
| P10 sparse reg=5e-4, sau FT | Yes | 2,415,613 | 3.2328G | 0.98124 | 0.76318 | 9.719 ms | 102.89 |
| P10 direct, sau FT khớp cấu hình | No | 2,416,871 | 3.2695G | 0.98273 | 0.77736 | 10.433 ms | 95.85 |
| P20 direct, trước FT | No | 1,913,971 | 2.5722G | 0.00000 | 0.00000 | 10.802 ms | 92.57 |
| P20 direct, sau FT | No | 1,913,971 | 2.5722G | 0.98184 | 0.76710 | 11.717 ms | 85.35 |
| P30 direct, trước FT | No | 1,452,562 | 1.9619G | 0.00000 | 0.00000 | 10.011 ms | 99.89 |
| P30 direct, sau FT | No | 1,452,562 | 1.9619G | 0.97788 | 0.75030 | 9.863 ms | 101.39 |

Lần sparse training đầu tiên dùng `reg=1e-4`, dừng sớm ở epoch 20 và có
validation mAP50-95 tốt nhất 0.78752 tại epoch 10. Tuy regularizer gradient khác
0, tỷ lệ group norm gần 0 vẫn bằng 0; P10 sau đó chưa cải thiện so với direct
pruning. Vì vậy P10 sparse hiện tại chưa được fine-tune và chưa được xem là mô
hình pruning thành công. Bước tiếp theo là điều chỉnh sparse regularization chỉ
dựa trên validation trước khi chạy lại P10.

Thí nghiệm tiếp theo dùng `reg=5e-4` đủ 30 epoch. Sparse checkpoint chưa prune
đạt mAP50-95 0.78938, nhưng group norm chỉ dịch xuống rất nhẹ và near-zero
fraction vẫn bằng 0. Sau P10, fine-tune dừng ở epoch 37 với best epoch 27 và phục
hồi mAP50-95 lên 0.76318. Mô hình giảm 19.80% tham số và 20.63% MACs so với
baseline, nhưng latency T4 tăng 17.25%; do đó chưa có speedup triển khai thực tế.

Control bắt buộc `direct pruning → fine-tune` được chạy đủ 50 epoch với
AdamW, `lr0=0.001`, `lrf=0.01`, momentum 0.9, weight decay 0.0005, batch 64 và
patience 10. Direct P10 đạt mAP50-95 0.77736, cao hơn sparse P10 0.01418
(1.42 điểm phần trăm) ở seed 42. Vì vậy sparse learning `reg=5e-4` chưa giúp
accuracy P10 sau fine-tune trong thí nghiệm hiện tại. Đây là kết luận một seed;
P20/P30 sẽ dùng direct pruning để tạo đường accuracy–compression.

Checkpoint direct P10 được phát hành public tại
[thangkt/PCB-Prune-YOLO-P10-Direct](https://huggingface.co/thangkt/PCB-Prune-YOLO-P10-Direct).

Direct P20 giảm 36.46% tham số và 36.85% MACs so với baseline. Trước fine-tune,
validation collapse về 0; sau đủ 50 epoch cùng cấu hình P10, mAP50-95 phục hồi
lên 0.76710. P20 thấp hơn P10 direct 1.03 điểm mAP50-95 và thấp hơn baseline
1.81 điểm, đồng thời latency T4 vẫn tăng lên 11.717 ms nên chưa có speedup thực tế.

Direct P30 giảm 51.77% tham số và 51.83% MACs. Sau 50 epoch, mAP50-95 đạt
0.75030: thấp hơn baseline 3.49 điểm, P10 2.71 điểm và P20 1.68 điểm. Model chỉ
3.014 MiB, nhưng latency 9.863 ms vẫn chậm hơn baseline 18.99%; vì vậy P30 là
ứng viên nén mạnh, không phải ứng viên accuracy hoặc latency tốt nhất.

## TensorRT FP16 trên Tesla T4

Bốn engine được build trực tiếp trên cùng Tesla T4 với TensorRT 10.16.1.11,
CUDA 12.8 và Ultralytics 8.4.115. Input tĩnh là `[1,3,640,640]`, batch 1,
FP16, không dynamic shape và không gắn NMS vào engine. Tất cả engine đã load
trong process mới và trả output `[1,10,8400]` với đúng sáu lớp.

| Model | Params | MACs | PyTorch mAP50-95 | TensorRT mAP50-95 | PyTorch latency | TensorRT latency | TensorRT FPS | Engine size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3,012,018 | 4.0733G | 0.78524 | 0.78716 | 8.289 ms | 1.837 ms | 544.28 | 7.477 MiB |
| P10 direct | 2,416,871 | 3.2695G | 0.77736 | 0.77842 | 10.433 ms | 2.023 ms | 494.34 | 7.627 MiB |
| P20 direct | 1,913,971 | 2.5722G | 0.76710 | 0.76931 | 11.717 ms | 1.933 ms | 517.45 | 7.378 MiB |
| P30 direct | 1,452,562 | 1.9619G | 0.75030 | 0.75610 | 9.863 ms | 1.754 ms | 569.97 | 5.482 MiB |

Latency PyTorch và TensorRT trong bảng là forward thuần, không gồm preprocess
hoặc NMS, với 50 warm-up và 200 lần đo có CUDA synchronize. Speedup TensorRT so
với chính PyTorch lần lượt là 4.51x, 5.16x, 6.06x và 5.62x. So với TensorRT
baseline, P10 đạt 0.91x, P20 0.95x và P30 1.05x; vì vậy chỉ P30 nhanh hơn
baseline TensorRT trong lần đo này, khoảng 4.7%.

Thời gian pipeline validation trung bình cho preprocess / engine inference /
postprocess-NMS lần lượt là baseline 1.413/3.290/2.230 ms, P10
1.379/3.842/2.153 ms, P20 1.346/3.857/2.285 ms và P30 1.425/3.515/2.046 ms mỗi
ảnh. JSON/CSV đầy đủ nằm tại `outputs/tensorrt_fp16/`; file `.engine` phụ thuộc
phần cứng/phần mềm build nên được giữ ngoài Git. Bốn engine và toàn bộ metadata
được phát hành tại
[thangkt/PCB-Prune-YOLO-TensorRT-FP16](https://huggingface.co/thangkt/PCB-Prune-YOLO-TensorRT-FP16).

P30 là lựa chọn triển khai ưu tiên khi cần nén mạnh: giảm 51.77% tham số,
51.83% MACs, đạt 569.97 FPS và nhanh hơn TensorRT baseline khoảng 4.7%. Đổi lại,
TensorRT mAP50-95 giảm từ 0.78716 xuống 0.75610 và recall giảm từ 0.96178 xuống
0.92858. Checkpoint PyTorch public:
[P20 direct](https://huggingface.co/thangkt/PCB-Prune-YOLO-P20-Direct) và
[P30 direct](https://huggingface.co/thangkt/PCB-Prune-YOLO-P30-Direct).

### Profile latency và INT8 PTQ

Profile mới trên cùng T4 cho thấy P30 giảm TensorRT GPU-compute nhưng có nhiều
kernel launch hơn baseline (3,571 so với 2,616), nên giảm MACs chưa chuyển đều
thành giảm latency. Runtime reuse đã được triển khai; CUDA Graph giảm forward
mean P30 5.32% nhưng E2E mean không cải thiện (`-0.16%`), vì vậy chưa bật mặc
định.

P30 INT8 PTQ calibrate bằng 500 ảnh **chỉ từ train** đạt validation mAP50-95
0.61119, giảm 14.49 điểm phần trăm so với P30 FP16; forward và E2E cũng chậm hơn
lần lượt 5.03% và 5.68% trong cùng runtime. Engine này không được chọn triển
khai và chưa chạy QAT/distillation. Báo cáo, protocol và quyết định tiếp theo ở
[`docs/TENSORRT_LATENCY_OPTIMIZATION.md`](docs/TENSORRT_LATENCY_OPTIMIZATION.md).

Smoke QAT explicit-Q/DQ 3 epoch sau đó phục hồi validation mAP50-95 lên
0.72462, tốt hơn PTQ 11.34 điểm nhưng vẫn thấp hơn FP16 3.15 điểm. ONNX giữ 133
cặp Q/DQ; full-FP32 convolution giảm 13 → 7, nhưng tổng TensorRT convolution
tăng 61 → 68, reformat tăng 72 → 77 và kernel launch tăng. QAT engine vẫn chậm
hơn P30 FP16 4.20% forward và 8.68% E2E, nên không chạy QAT dài hoặc
distillation. Gate hiện tại là `FIX_GRAPH_FIRST`; xem
[`docs/P30_INT8_QAT_SMOKE_REPORT.md`](docs/P30_INT8_QAT_SMOKE_REPORT.md).

### P40-HW latency gate

Ba candidate mới được prune lại từ baseline với target ratio 0.40: A8, A16 và
BLOCK. Tất cả giữ output `[1,10,8400]`, bảo vệ Detect/DFL và qua kiểm tra
save → process mới load → CUDA/TensorRT inference. Trong phép đo FP16 graph-off
cùng T4, P30 là 1.7575 ms; A8 đạt **1.3540 ms** (1.298x), A16 1.4903 ms và
BLOCK 1.5590 ms. A8 được chọn qua latency gate với 903,466 params và 1.1212G
MACs. Tuy nhiên validation trước fine-tune của A8 bằng 0 cho toàn bộ metric, vì
vậy chưa train dài hoặc distill. Chi tiết và artifact nằm tại
[`docs/P40_HW_LATENCY_GATE.md`](docs/P40_HW_LATENCY_GATE.md) và
`outputs/pruning_hw/comparison.{json,csv}`.

Ultralytics 8.4.115 (bản pin của project) có sẵn knowledge distillation
(`distill_model`/`dis`, score-weighted feature L2 loss ở đầu vào Detect head).
Fine-tune đầu tiên (50 epoch, cùng cấu hình AdamW `lr0=0.001`, `lrf=0.01`,
momentum 0.9, weight decay 0.0005, batch 64, patience 10, seed 42) cho thấy cả
hai nhánh vẫn đang cải thiện ở epoch cuối (chưa hội tụ), nên được chạy lại với
100 epoch, patience 20 và cosine LR — lần này cả hai đều bão hòa rõ ràng ở
cuối chu kỳ:

| Nhánh | Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Fine-tune chuẩn, 50 epoch | 50 | 0.867 | 0.805 | 0.893 | 0.634 |
| KD, 50 epoch | 50 | 0.895 | 0.828 | 0.913 | 0.660 |
| Fine-tune chuẩn, 100 epoch + cos_lr | 100 | 0.929 | 0.888 | 0.950 | 0.701 |
| **KD, 100 epoch + cos_lr** | 100 | 0.930 | 0.898 | 0.957 | **0.712** |

Train dài hơn giúp cả hai tăng mạnh (+6.7 và +5.2 điểm mAP50-95); khoảng cách
KD so với fine-tune chuẩn thu hẹp từ +2.7 xuống +1.1 điểm khi có đủ epoch, KD
vẫn thắng. Sweep thêm `dis=3.0`/`10.0` xác nhận mặc định `dis=6.0` đã gần tối
ưu (0.711/0.712/0.709), không cần đổi. Cùng kiến trúc 903,466 params/1.1212G
MACs (−70.00% params, −72.47% MACs so với baseline).

**So với P30** (mAP50-95 0.75030, −51.77%/−51.83% params/MACs): sau khi train
đủ epoch, P40-A8 KD chỉ còn kém P30 **3.83 điểm** mAP50-95 (so với 9.03 điểm
lúc mới fine-tune 50 epoch), trong khi vẫn nén mạnh hơn hẳn (37.8% ít params,
42.85% ít MACs hơn P30) và nhanh hơn ~1.3x trên TensorRT — một đánh đổi hợp lý
hơn nhiều so với trước, dù P30 vẫn là lựa chọn accuracy cao nhất.

Test set chưa được dùng. Hai checkpoint (bản 100 epoch) đã public tại
[thangkt/PCB-Prune-YOLO-P40-A8-Direct](https://huggingface.co/thangkt/PCB-Prune-YOLO-P40-A8-Direct)
và
[thangkt/PCB-Prune-YOLO-P40-A8-KD](https://huggingface.co/thangkt/PCB-Prune-YOLO-P40-A8-KD),
kèm model card, args, validation và benchmark; đã xác minh tải ẩn danh.

Engine TensorRT FP16 được build lại cho checkpoint 100-epoch (cùng phiên đo
với baseline để so sánh công bằng, 50 warm-up/200 lần đo, batch 1):

| Model | Params | MACs | TensorRT latency | FPS | So với baseline |
|---|---:|---:|---:|---:|---:|
| Baseline | 3,012,018 | 4.0733G | 1.716 ms | 582.75 | 1.00x |
| P40-A8 chuẩn (100 epoch) | 903,466 | 1.1212G | 1.422 ms | 703.22 | 1.21x |
| P40-A8 KD (100 epoch) | 903,466 | 1.1212G | 1.497 ms | 668.12 | 1.15x |

Cả hai vẫn nhanh hơn baseline TensorRT rõ rệt (kiến trúc không đổi qua
fine-tune nên latency gần như y hệt bản 50 epoch trước đó; chênh lệch nhỏ
giữa hai lần đo là nhiễu đo đạc/tactic selection, không phải khác biệt kiến
trúc), đã qua verify save → process mới load → inference. Chi tiết ở
[`docs/P40_HW_LATENCY_GATE.md`](docs/P40_HW_LATENCY_GATE.md).

## HALP: latency-aware pruning

Giai đoạn 1 và dry-run Stage 2 của adaptation HALP đã hoàn tất; đây chưa phải mô
hình HALP đã prune. Baseline backbone có 27 convolution thuộc 19 TensorRT
operator signature.
LUT T4 FP16 chứa 598 cấu hình `Cin×Cout`, warm-up 50 và đo 200 lần; toàn bộ 598
cấu hình thành công. Phân tích tìm được 56 latency cliff và 98 plateau. Group
step đo được thay đổi theo layer (8, 16, 24, 32, 40, 48 hoặc 64), không mặc định
mọi layer theo 8.

Thiết kế, provenance paper/code, khác biệt SSD–YOLOv8 và giới hạn adaptation nằm
trong [`docs/HALP_ADAPTATION_PLAN.md`](docs/HALP_ADAPTATION_PLAN.md). LUT và
staircase report nằm ở `outputs/halp/lut/`.

Stage 2 thu Taylor saliency trên 8 train minibatch bằng
`|γ·∂L/∂γ + β·∂L/∂β|`, dựng dependency graph nhưng chỉ cho phép root
`model.0`–`model.9`, tạo prefix group theo cliff đo được và giải augmented
knapsack cho milestone giảm 5% latency. Dry-run tìm 25 backbone root: 13 root
eligible, 12 root được giữ nguyên vì chưa có latency cliff đáng tin cậy, không
thiếu cặp LUT chính xác, không sửa trọng số/kênh, và forward vẫn là
`[1,10,8400]` với 6 lớp. Báo cáo nằm ở `outputs/halp/stage2/`; chạy lại bằng:

```bash
python scripts/run_halp_stage2.py --saliency-batches 8 --batch 8
```

Cost hiện tại bám bước đầu của paper: dùng latency của output convolution tại
trạng thái hiện tại; `Cin` downstream sẽ phải được tính lại ở mỗi milestone
structural-pruning sau này. Vì vậy chưa có pruning, fine-tune hay test-set
evaluation trong Stage 2 này.

Stage 3 đã áp dụng structural milestone 5% đầu tiên sau khi sửa Taylor group
aggregation đúng official HALP: cộng các term có dấu trong dependency group rồi
mới lấy trị tuyệt đối. Tám cặp `Cin×Cout` phát sinh được đo bổ sung trên cùng T4
với 50 warm-up và 200 lần đo; audit sau prune dùng LUT chính xác, không nội suy.
Checkpoint giảm 10.67% params và 3.81% MACs; save → process mới load → inference
đạt. Tuy nhiên trước fine-tune, validation mAP50-95 chỉ còn 0.67681 và latency
PyTorch tăng lên 10.115 ms (98.86 FPS), nên đây mới là checkpoint kỹ thuật, chưa
phải mô hình HALP tốt. Báo cáo nằm ở `outputs/halp/stage3_m05/`; chưa chạy test.

TensorRT gate công bằng cho thấy M05 **không tăng tốc full-engine forward**:
baseline 1.780 ms so với M05 1.838 ms (`0.968x`). `trtexec` per-layer cũng gần
như hòa nhưng M05 chậm hơn (`0.998x`). E2E gồm preprocess/H2D/NMS nhanh hơn
`1.076x`, nhưng đi cùng giảm 0.12971 mAP50-95 và 0.07235 recall nên có thể do
NMS xử lý ít candidate hơn; không được xem là tăng tốc kiến trúc. Chi tiết
PAPER/OFFICIAL CODE/ADAPTATION và bottleneck nằm trong
[`docs/HALP_STAGE3_TENSORRT_REPORT.md`](docs/HALP_STAGE3_TENSORRT_REPORT.md).

Để tiếp tục trên server mới từ một clone sạch, làm theo
[`docs/RESUME_HALP.md`](docs/RESUME_HALP.md); tài liệu này pin TensorRT, tải lại
baseline đúng đường dẫn, phục hồi official HALP đúng commit và chỉ rõ thứ tự đọc
ngữ cảnh trước Stage 2.

Checkpoint P10 fine-tuned và model card được phát hành public tại
[thangkt/PCB-Prune-YOLO-P10-DepGraph](https://huggingface.co/thangkt/PCB-Prune-YOLO-P10-DepGraph).
Vì structured pruning thay đổi kiến trúc, hãy clone và cài project trước khi
load checkpoint để class `PrunableC2f` khả dụng.

## Kiểm tra code

```bash
python -m compileall -q src scripts tests
python -m pytest -q
```

Không commit dataset, checkpoint, cache hoặc ảnh training-batch. Các JSON/CSV,
config đã dùng, confusion matrix và metric plot được giữ trong Git để clone dự
án vẫn xem được trạng thái thí nghiệm. Knowledge distillation chưa được sử dụng.

## Thứ tự chạy trên server

```bash
# 1. Kiểm tra môi trường
nvidia-smi
python scripts/check_environment.py --require-gpus 2

# 2. Chuẩn bị dữ liệu
python scripts/prepare_deeppcb.py

# 3. Validate dữ liệu
python scripts/validate_dataset.py

# 4. Tạo 20 ảnh preview
python scripts/visualize_annotations.py --count 20

# 5. Smoke train ngắn trên 2 GPU
python scripts/train_baseline.py --smoke --batch 128 --fraction 0.2

# 6. Train baseline đầy đủ trên 2 GPU
python scripts/train_baseline.py

# 7. Evaluate validation; test chỉ dùng cho báo cáo cuối
python scripts/evaluate_model.py --checkpoint outputs/train/baseline/weights/best.pt --split val --device 0
python scripts/evaluate_model.py --checkpoint outputs/train/baseline/weights/best.pt --split test --device 0

# 8. Benchmark batch size 1 trên GPU đầu tiên
python scripts/benchmark_model.py --model outputs/train/baseline/weights/best.pt --device cuda:0
```

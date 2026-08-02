# PCB-Prune-YOLO: DepGraph Pruning for PCB Defect Detection

Phạm Ngọc Thắng

![Task](https://img.shields.io/badge/Task-Object_Detection-c0392b)
![Dataset](https://img.shields.io/badge/Dataset-DeepPCB-d35400)
![Model](https://img.shields.io/badge/Model-YOLOv8n-55a630)
![Language](https://img.shields.io/badge/Language-Python-3776ab)

**Quick Links:** [📦 Tải dataset](https://huggingface.co/datasets/thangkt/PCB-Prune-YOLO-DeepPCB) | [🤗 Model baseline](https://huggingface.co/thangkt/PCB-Prune-YOLO-Baseline) | [⚙️ Cấu hình](#cấu-hình-baseline) | [🚀 Huấn luyện](#huấn-luyện) | [📊 Kết quả](#kết-quả-baseline)

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

Lần sparse training đầu tiên dùng `reg=1e-4`, dừng sớm ở epoch 20 và có
validation mAP50-95 tốt nhất 0.78752 tại epoch 10. Tuy regularizer gradient khác
0, tỷ lệ group norm gần 0 vẫn bằng 0; P10 sau đó chưa cải thiện so với direct
pruning. Vì vậy P10 sparse hiện tại chưa được fine-tune và chưa được xem là mô
hình pruning thành công. Bước tiếp theo là điều chỉnh sparse regularization chỉ
dựa trên validation trước khi chạy lại P10.

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

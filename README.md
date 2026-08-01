# PCB-Prune-YOLO: Efficient PCB Defect Detection

[Phạm Ngọc Thắng](#pcb-prune-yolo-efficient-pcb-defect-detection)

![Task](https://img.shields.io/badge/Task-Defect_Detection-c0392b)
![Domain](https://img.shields.io/badge/Domain-PCB_Quality_Control-d35400)
![Method](https://img.shields.io/badge/Method-Structured_Pruning-2980b9)
![Model](https://img.shields.io/badge/Model-PCB--Prune--YOLO-55a630)
![Language](https://img.shields.io/badge/Language-Python-3776ab)

**Quick Links:** [📦 Dữ liệu](#chuẩn-bị-dữ-liệu) | [⚙️ Cấu hình](#cấu-trúc) | [🚀 Huấn luyện](#chạy-thí-nghiệm) | [📊 Kết quả](#kết-quả-benchmark) | [🧪 Kiểm thử](#trạng-thái-triển-khai)

Dự án được xây dựng độc lập để huấn luyện, tinh gọn và đánh giá mô hình phát hiện lỗi trên bảng mạch in. Mục tiêu là giảm số lượng tham số, kích thước mô hình và độ trễ suy luận trong khi duy trì độ chính xác phù hợp.

## Quy trình

`chuẩn bị dữ liệu → huấn luyện mô hình gốc → đánh giá → tinh gọn → tinh chỉnh → đo hiệu năng`

## Cấu trúc

```text
configs/                 Cấu hình dữ liệu, huấn luyện, tinh gọn và benchmark
data/                    Dữ liệu gốc và dữ liệu đã xử lý
outputs/                 Mô hình, báo cáo và kết quả chạy
scripts/                 Các lệnh thực thi quy trình
src/pcb_prune_yolo/      Mã nguồn chính
src/torch_pruning/       Module tinh gọn mô hình nội bộ
tests/                   Kiểm thử
```

## Cài đặt

Yêu cầu Python 3.10 trở lên:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Chuẩn bị dữ liệu

Dữ liệu không được lưu trong repository. Annotation đầu vào có định dạng `x1,y1,x2,y2,class_id`.

```bash
make prepare-data IMAGES=PATH_IMAGES LABELS=PATH_LABELS
python scripts/validate_dataset.py --root data/processed --split train
python scripts/visualize_annotations.py --root data/processed
```

## Chạy thí nghiệm

```bash
python scripts/train_baseline.py
python scripts/evaluate_model.py --checkpoint outputs/train/baseline/weights/best.pt
python scripts/prune_model.py --checkpoint outputs/train/baseline/weights/best.pt --dry-run
python scripts/finetune_pruned.py --model PATH_PRUNED_MODEL
python scripts/benchmark_model.py --model outputs/train/baseline/weights/best.pt --device cpu
```

Giá trị mặc định nằm trong `configs/`; tham số dòng lệnh dùng để ghi đè khi cần.

## Kết quả benchmark

Hiệu năng cần được đo trực tiếp trên thiết bị triển khai vì số phép tính giảm không luôn đồng nghĩa với độ trễ thấp hơn.

| model | pruning ratio | parameters | MACs | mAP50 | mAP50-95 | latency | FPS | model size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0% | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| pruned | 20% | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Trạng thái triển khai

- Đã có quy trình chuyển đổi, chia, kiểm tra và xem trước dữ liệu.
- Đã có lệnh huấn luyện, đánh giá, tinh chỉnh và benchmark.
- Chế độ kiểm tra trước khi tinh gọn đã hỗ trợ dựng đồ thị phụ thuộc và đếm tham số.
- Tinh gọn thật và lưu mô hình sau tinh gọn chưa hoàn thiện.
- Phần tính MACs chưa hoàn thiện.

Chạy kiểm tra bằng `make test` và `make lint`. Các script không huấn luyện hoặc tải dữ liệu khi được import.

# Server Runbook — PCB-Prune-YOLO Baseline

## Mục tiêu

Chuẩn bị môi trường, tải dataset đã xử lý, kiểm tra dữ liệu, smoke train, train baseline trên 2 GPU T4, đánh giá val/test và benchmark checkpoint tốt nhất.

Chỉ thực hiện baseline. Không chạy hoặc sửa pruning, DepGraph hay knowledge distillation. Không dùng test set để chọn hyperparameter.

Mọi lệnh phải chạy từ thư mục gốc `PCB-Prune-YOLO`.

## 1. Kiểm tra server

```bash
nvidia-smi
python3 --version
```

Yêu cầu:

- Python 3.10 trở lên.
- Hai GPU T4 đều hiển thị trong `nvidia-smi`.
- Không cài đè PyTorch trước khi biết CUDA/driver của server.

## 2. Tạo môi trường

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Kiểm tra PyTorch có sẵn trong môi trường:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

Nếu chưa có PyTorch CUDA, cài bản tương thích với driver/CUDA của server trước khi tiếp tục. Không đoán phiên bản CUDA. Tùy chọn `--system-site-packages` giúp giữ bản PyTorch GPU có sẵn trong image của server.

Cài project mà không thay PyTorch đã chọn:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/check_environment.py --require-gpus 2
```

Chỉ tiếp tục khi script nhìn thấy đúng hai GPU.

## 3. Tải dataset

```bash
curl -L -o deeppcb_processed.zip https://huggingface.co/datasets/thangkt/PCB-Prune-YOLO-DeepPCB/resolve/main/deeppcb_processed.zip
mkdir -p data/processed
unzip -o deeppcb_processed.zip -d data/processed
```

Cấu trúc bắt buộc:

```text
data/processed/deeppcb/images/train
data/processed/deeppcb/images/val
data/processed/deeppcb/images/test
data/processed/deeppcb/labels/train
data/processed/deeppcb/labels/val
data/processed/deeppcb/labels/test
```

Không cần tải dữ liệu raw và không chạy `prepare_deeppcb.py` khi đã dùng ZIP này.

## 4. Kiểm tra code và dữ liệu

```bash
python -m compileall -q src scripts tests
python -m pytest -q
python scripts/validate_dataset.py
python scripts/visualize_annotations.py --count 20
```

Kết quả mong đợi:

- Unit test đạt.
- Train: 800 ảnh.
- Val: 200 ảnh.
- Test: 500 ảnh.
- Không có ảnh trùng giữa ba split.
- Có 20 ảnh trong `outputs/dataset_preview`.

Nếu validate lỗi, dừng lại; không train trên dữ liệu lỗi.

## 5. Smoke train

```bash
python scripts/train_baseline.py --smoke --batch 8
```

Smoke train phải chạy đúng 5 epoch trên `device=0,1`. Kết quả nằm trong `outputs/train/smoke`.

Nếu CUDA out of memory, thử lại với batch 4. Không thay đổi ảnh 640 px trong smoke test trừ khi cần chẩn đoán.

## 6. Train baseline

```bash
python scripts/train_baseline.py
```

Cấu hình mặc định:

- Model pretrained: `yolov8n.pt`
- Device: `0,1`
- Batch tổng: 32, tương đương 16 ảnh/GPU
- Image size: 640
- Epoch tối đa: 100
- Early stopping patience: 20
- Seed: 42
- AMP: bật
- Deterministic: bật

Nếu CUDA out of memory:

```bash
python scripts/train_baseline.py --batch 16
```

Checkpoint cần dùng cho các bước sau:

```text
outputs/train/baseline/weights/best.pt
```

## 7. Đánh giá

Đánh giá val trước:

```bash
python scripts/evaluate_model.py --checkpoint outputs/train/baseline/weights/best.pt --split val --device 0
```

Chỉ sau khi training và lựa chọn cấu hình hoàn tất mới đánh giá test:

```bash
python scripts/evaluate_model.py --checkpoint outputs/train/baseline/weights/best.pt --split test --device 0
```

Kết quả:

```text
outputs/evaluation/metrics_val.json
outputs/evaluation/metrics_val.csv
outputs/evaluation/metrics_test.json
outputs/evaluation/metrics_test.csv
```

## 8. Benchmark

```bash
python scripts/benchmark_model.py --model outputs/train/baseline/weights/best.pt --device cuda:0
```

Benchmark dùng batch size 1, có warm-up và CUDA synchronization. Kết quả:

```text
outputs/benchmark/benchmark.json
outputs/benchmark/benchmark.csv
```

## Điều kiện hoàn thành

Agent chỉ báo hoàn thành khi:

1. Hai GPU được nhận diện.
2. Dataset validate thành công.
3. Smoke train hoàn tất 5 epoch.
4. Full baseline tạo `best.pt`.
5. Có đủ báo cáo val, test và benchmark JSON/CSV.

Khi kết thúc, báo cáo đường dẫn checkpoint, epoch tốt nhất, lý do dừng, precision, recall, mAP50, mAP50-95, latency mean/median/p95, FPS và peak GPU memory. Không tạo số liệu giả nếu một bước thất bại.

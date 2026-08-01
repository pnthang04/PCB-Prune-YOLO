# PCB-Prune-YOLO

Khung thí nghiệm tối ưu YOLOv8n cho phát hiện sáu loại lỗi PCB trong DeepPCB bằng structured pruning dựa trên DepGraph. Nền tảng phương pháp là paper **“DepGraph: Towards Any Structural Pruning” (CVPR 2023)**. Project dùng package chính thức `ultralytics` và `torch-pruning`, không sao chép source của hai dự án.

## Quy trình

`chuẩn bị dữ liệu → train baseline → đánh giá → prune → fine-tune → benchmark`

## Cài đặt

Yêu cầu Python 3.10+:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Không pin CUDA trong project. Hãy cài bản PyTorch phù hợp với thiết bị theo hướng dẫn của PyTorch nếu cần GPU.

## Dữ liệu DeepPCB

Không commit dataset. Có thể đặt dữ liệu gốc trong `data/raw/` hoặc nơi khác. Truyền riêng thư mục ảnh và annotation (`x1,y1,x2,y2,class_id`):

```bash
python scripts/prepare_deeppcb.py --images PATH_IMAGES --labels PATH_LABELS --output data/processed --class-offset 1
python scripts/validate_dataset.py --root data/processed --split train
python scripts/visualize_annotations.py --root data/processed
```

Config Ultralytics nằm ở `configs/data/deeppcb.yaml`. Nếu di chuyển dữ liệu, sửa `path` trong file này.

## Chạy thí nghiệm

```bash
python scripts/train_baseline.py
python scripts/evaluate_model.py --checkpoint outputs/train/baseline/weights/best.pt
python scripts/prune_model.py --checkpoint outputs/train/baseline/weights/best.pt --dry-run
python scripts/finetune_pruned.py --model PATH_PRUNED_MODEL
python scripts/benchmark_model.py --model outputs/train/baseline/weights/best.pt --device cpu
```

Mọi mặc định nằm trong `configs/`; CLI dùng để override. Ultralytics tự lưu `best.pt` và `last.pt` khi train/fine-tune. Phiên bản đầu không có knowledge distillation.

## Diễn giải benchmark

Giảm FLOPs/MACs không đảm bảo latency giảm vì kernel, memory bandwidth và phần cứng khác nhau. Luôn đo latency trên chính thiết bị triển khai, sau warm-up và đồng bộ CUDA.

| model | pruning ratio | parameters | MACs | mAP50 | mAP50-95 | latency | FPS | model size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n baseline | 0% | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8n pruned | 20% | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Trạng thái triển khai

- Hoàn thiện skeleton: chuyển annotation, split tái lập, validate, preview, train/evaluate wrapper, latency benchmark và report JSON/CSV.
- Dry-run DepGraph xây graph và đếm parameters; MACs hiện là `TODO`.
- Pruning thật và lưu checkpoint pruned chủ động `raise NotImplementedError` cho đến khi xác định chính xác detection head/output layers cần bảo vệ.
- Chưa triển khai knowledge distillation.

Chạy kiểm tra bằng `make test` và `make lint`. Không script nào train hoặc tải dữ liệu khi import.

# Anti Face Deepfake — ViT Image Classification (DINOv3)

Đồ án: Phát hiện deepfake trên ảnh khuôn mặt bằng Vision Transformer (DINOv3) — self-supervised ViT của Meta AI.

## Cấu trúc thư mục

```
deepfake/
├── .venv/              # Môi trường ảo (Python ≥ 3.10, chuẩn 3.11)
├── agents/             # Agent AI knowledge base (rules, phases, progress)
├── configs/            # File cấu hình (YAML)
├── data/
│   ├── raw/            # Data thô (ảnh deepfake / real)
│   ├── processed/      # Data đã xử lý (crop, resize...)
│   └── external/       # Data ngoài
├── src/                # Package chính (script entry points)
│   ├── data/           # Dataset, transforms + build/split/download scripts
│   ├── models/         # ViT (DINOv3), ConvNeXt, LoRA
│   ├── training/       # Train loop (train.py, finetune_*.py)
│   ├── eval/           # Evaluation scripts (eval_*.py, evaluate.py)
│   ├── experiments/    # Comparison, figures, analysis scripts
│   └── utils/          # Logger, helpers, shell utilities
├── experiments/
│   ├── checkpoints/    # Checkpoint + pretrained weights
│   ├── results/        # Kết quả đánh giá, report, research
│   ├── plots/          # Figures
│   └── runs/           # Run logs/history
├── notebooks/          # Notebook phân tích
└── tests/              # Kiểm thử
```

## Cài đặt môi trường

```bash
python -m venv venv

# Kích hoạt môi trường ảo
source .venv/bin/activate

# 1) Torch + torchvision — cài RIÊNG với CUDA index (KHÔNG nằm trong requirements):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   (DINOv3 cần PyTorch >= 2.7.1; trên Mac: pip install torch torchvision)

# 2) Các thư viện Python còn lại — bản pin (tái lập chính xác):
pip install -r requirements.lock.txt
#   (hoặc `pip install -r requirements.txt` để lấy bản mới nhất thỏa floor)
```

## Chạy các lệnh chính (run locally)

Tất cả lệnh chạy từ **thư mục gốc repo**; mỗi script có `--help` liệt kê đầy đủ
option. Mặc định dùng `--device auto` (cuda nếu có, ngược lại mps/cpu).

### Tests (smoke)

```bash
python -m pytest                 # structural smoke tests (torch-gated test tự skip nếu thiếu torch)
```

### Data (download / build / split)

```bash
# Raw DF40 training corpus — tải từ Hugging Face (khuyến nghị, ~74.7 GB):
#   hf download ManhQuangAI/DF40_train --repo-type dataset --local-dir data/raw/DF40
export DF40_ROOT=/path/to/DF40    # mặc định data/raw/DF40
# LƯU Ý: src/data/download_df40.py (Google Drive/gdown) KHÔNG còn đáng tin —
# các link Drive thường không truy cập được (gdown rc=1). Dùng HF Hub làm nguồn chính.

# Build subset cân bằng real/fake
python src/data/build_df40_balanced.py --n 750 --dry-run
python src/data/build_df40_balanced.py --n 750

# Build test sets + split
python src/data/build_test_data.py --n 1000
python src/data/build_test_data_v2.py
python src/data/restructure_test_data_v3.py
python src/data/split_dataset.py        # tạo train/val/test CSVs — xem --help
```

### Training (fine-tune)

```bash
# Full fine-tune DINOv3 (backbone + head)
python src/training/train.py \
  --train-csv data/splits/train.csv --val-csv data/splits/val.csv --test-csv data/splits/test.csv \
  --epochs 5 --batch-size 32 --amp --num-workers 2

# LoRA fine-tune
python src/training/finetune_lora.py \
  --train-csv data/splits/train.csv --val-csv data/splits/val.csv \
  --lora-rank 16 --lora-alpha 32 --amp

# ViT-vs-CNN (matched params)
python src/training/finetune_compare.py --model-type vit \
  --train-csv data/splits/train.csv --val-csv data/splits/val.csv --test-csv data/splits/test.csv --amp
python src/training/finetune_compare.py --model-type cnn \
  --train-csv data/splits/train.csv --val-csv data/splits/val.csv --test-csv data/splits/test.csv --amp
```

> `--amp` bật mixed-precision bfloat16 (hoạt động trên CUDA/CPU; tự tắt trên MPS).
> Kết quả: checkpoint trong `experiments/checkpoints/`, report JSON trong
> `experiments/results/`.

### Eval

```bash
python src/eval/evaluate.py                                  # linear-probe eval
python src/eval/predict.py                                   # dự đoán 1 ảnh
python src/eval/eval_df40_vit_cnn.py --device cuda           # ViT vs CNN trên DF40
python src/eval/eval_identity_disjoint.py --model vit --root test_data_v3 --device cuda
python src/eval/eval_finetuned.py --device cuda              # eval sau fine-tune
python src/eval/eval_df40_all_methods.py --device cuda       # eval 40 method
python src/eval/analyze_threshold.py --ckpt experiments/checkpoints/finetune/vit_lora_finetuned.pt
python src/eval/benchmark_inference.py --device cuda         # latency/throughput
```

### Experiments & figures

```bash
python src/experiments/compare_models.py --device cuda                  # so sánh ViT vs CNN
python src/experiments/visualize_attention.py \
  --model experiments/checkpoints/weights/model.safetensors \
  --output-dir experiments/plots/attention                              # attention maps
python src/experiments/assemble_attention_figure.py                     # ghép ảnh attention
python src/experiments/make_report_figures.py                           # figure cho report
python src/experiments/make_method_report_md.py \
  --vit experiments/results/eval/vit.json --cnn experiments/results/eval/cnn.json \
  --output experiments/results/eval/report_40_methods_v3.md
python src/experiments/make_confusion_matrix.py
```

## Ghi chú

- **DINOv3**: yêu cầu PyTorch ≥ 2.7.1, timm ≥ 1.0.20 (hoặc HuggingFace Transformers ≥ 4.56).
  Pretrained weights tải từ HuggingFace Hub (`facebook/dinov3-*`), cần chấp nhận license của Meta.
- **Python**: chuẩn hoá ở **3.11** (xem [src/utils/setup_ubuntu.sh](src/utils/setup_ubuntu.sh));
  môi trường local nên dùng ≥ 3.10 (3.9 đã EOL từ 10/2025).
- **DF40_ROOT**: đường dẫn tới data DF40 gốc (mặc định `data/raw/DF40`). Các script
  `src/data/build_*` và `src/data/download_df40.py` đọc nó từ env var `DF40_ROOT`
  (hoặc đối số `--src`) — không hardcode đường dẫn máy.
- **Reproducibility**: dùng `requirements.lock.txt` (bản pin) để tái lập môi trường;
  tái tạo lockfile trong môi trường GPU thật bằng
  `pip freeze | grep -vE '^(torch|torchvision|torchaudio|nvidia-)' > requirements.lock.txt`
  (loại `torch`/`torchvision` vì chúng được cài riêng với CUDA index).

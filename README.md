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
# Kích hoạt môi trường ảo
source .venv/bin/activate

# 1) Torch + torchvision — cài RIÊNG với CUDA index (KHÔNG nằm trong requirements):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   (DINOv3 cần PyTorch >= 2.7.1; trên Mac: pip install torch torchvision)

# 2) Các thư viện Python còn lại — bản pin (tái lập chính xác):
pip install -r requirements.lock.txt
#   (hoặc `pip install -r requirements.txt` để lấy bản mới nhất thỏa floor)
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
  tái tạo lockfile trong môi trường GPU thật bằng `pip freeze > requirements.lock.txt`.

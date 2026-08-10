# Anti Face Deepfake — ViT Image Classification (DINOv3)

Đồ án: Phát hiện deepfake trên ảnh khuôn mặt bằng Vision Transformer (DINOv3) — self-supervised ViT của Meta AI.

## Cấu trúc thư mục

```
deepfake/
├── .venv/              # Môi trường ảo (Python 3.9)
├── data/
│   ├── raw/            # Data thô (ảnh deepfake / real)
│   ├── processed/      # Data đã xử lý (crop, resize...)
│   └── splits/         # Bộ train/val/test
├── models/             # Pretrained weights DINOv3 / checkpoints
├── src/                # Package chính
│   ├── data/           # Dataset, transforms
│   ├── models/         # ViT model, classifier head
│   ├── training/       # Train loop, loss
│   └── utils/          # Logger, metrics, helpers
├── configs/            # File cấu hình (YAML)
├── scripts/            # train.py / evaluate.py / predict.py
├── notebooks/          # Notebook thử nghiệm
├── outputs/
│   ├── checkpoints/    # Checkpoint model
│   ├── logs/           # Log huấn luyện
│   └── results/        # Kết quả đánh giá
└── tests/              # Kiểm thử
```

## Cài đặt môi trường

```bash
# Kích hoạt môi trường ảo
source .venv/bin/activate

# Cài dependencies (khi đưa data/model vào)
pip install torch torchvision timm
pip install matplotlib tqdm pandas scikit-learn tensorboard
```

## Ghi chú

- **DINOv3**: yêu cầu PyTorch ≥ 2.7.1, timm ≥ 1.0.20 (hoặc HuggingFace Transformers ≥ 4.56).
  Pretrained weights tải từ HuggingFace Hub (`facebook/dinov3-*`), cần chấp nhận license của Meta.
- **Python**: 3.9 (system). Lưu ý 3.9 đã EOL từ 10/2025 — khuyến nghị nâng lên 3.10+ sau này.

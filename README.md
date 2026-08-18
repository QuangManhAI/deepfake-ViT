# Anti Face Deepfake — ViT Image Classification (DINOv3)

Đồ án: Phát hiện deepfake trên ảnh khuôn mặt bằng Vision Transformer (DINOv3) — self-supervised ViT của Meta AI.

## Cấu trúc thư mục

```
deepfake/
├── .venv/              # Môi trường ảo (Python 3.9)
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

# Cài dependencies (khi đưa data/model vào)
pip install torch torchvision timm
pip install matplotlib tqdm pandas scikit-learn tensorboard
```

## Ghi chú

- **DINOv3**: yêu cầu PyTorch ≥ 2.7.1, timm ≥ 1.0.20 (hoặc HuggingFace Transformers ≥ 4.56).
  Pretrained weights tải từ HuggingFace Hub (`facebook/dinov3-*`), cần chấp nhận license của Meta.
- **Python**: 3.9 (system). Lưu ý 3.9 đã EOL từ 10/2025 — khuyến nghị nâng lên 3.10+ sau này.

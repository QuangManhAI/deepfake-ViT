# 03 — Finetune: replicate recipe v5_weakfix (baseline → v2 → v3)

## 0. Môi trường

- GPU: RTX 3060 12GB (dùng chung — chạy 1 tiến trình train/lúc, batch 64, bf16).
- Python: `/workspace/hoangtuan/deepfake-ViT/.venv/bin/python`
  (torch 2.6.0, pandas 2.3.3, opencv, sklearn) — venv của repo này `.venv` không có pandas.
- Loss: `LabelSmoothingCrossEntropy(0.05)`; Optimizer: AdamW; EMA 0.999;
  Scheduler: CosineAnnealingLR(eta_min=1e-6); clip grad 1.0; bf16 autocast; seed 42.
- Transform train: BICUBIC 256 → HFlip(0.5) → ColorJitter(0.2,0.2,0.2,0.05)@0.5 →
  GaussianBlur(3,5)@0.3 → AdjustSharpness(2.0)@0.3 → ToTensor → ImageNet normalize.
- Aug eval: BICUBIC 256 → ToTensor → normalize.

## 1. Baseline — "v5-equivalent" cho ConvNeXt (`scripts/finetune_convnext_weakfix.py --stage baseline`)

| Cấu hình | Giá trị |
|---|---|
| Init | pretrained backbone `dinov3_next_cnn/model-2.safetensors` + head MLP mới |
| Train | `train_v5_combined_universal_kaggle_boost.csv` (54,000 = 27,006R/26,994F) |
| Val | `val_v5_combined_universal_kaggle_boost.csv` (6,000) |
| Test | zero-leakage benchmark (2,354) |
| Sampler | **plain shuffle** (CSV cân bằng sẵn — giống cách v5 train) |
| Epochs / batch | **3** / 64 |
| LR | backbone `2e-5`, head `5e-4`, weight_decay 0.05 |

> Đúng recipe mà v5 của hoangtuan dùng (đọc từ config checkpoint
> `20260823_074722_v5_combined_universal`: epochs 3, batch 64, base_lr 2e-5, head_lr 5e-4,
> wd 0.05, seed 42, EMA) → để baseline ConvNeXt so được với baseline v5 (95.37%).

## 2. v2 — finetune method-balanced (init từ baseline)

| Cấu hình | Giá trị |
|---|---|
| Init | `outputs/finetune/convnext_baseline.pt` |
| Train | `data/splits/train_v5_weakfix.csv` (121,884 = 31,006R/90,878F) |
| Sampler | **method-balanced** — P(real)=0.5; fake chia đều `0.5/num_methods` (42 method) |
| Số mẫu/epoch | `2 × num_real` ≈ 62,012 |
| Epochs / batch | **2** / 64 |
| LR | backbone `2e-5`, head `5e-4`, weight_decay 0.05 |

## 3. v3 — finetune faceswap-focused (init từ v2)

| Cấu hình | Giá trị |
|---|---|
| Init | `outputs/finetune/convnext_weakfix_v2.pt` |
| Train | `data/splits/train_v5_weakfix_v3.csv` (129,884 = 31,006R/98,878F; faceswap 12,600) |
| Sampler | **faceswap-focused** — P(faceswap)=**0.35**, P(real)=0.35, P(method khác)=0.30 chia đều |
| Số mẫu/epoch | `2 × num_real` ≈ 62,012 |
| Epochs / batch | **3** / 64 |
| LR | backbone `1.5e-5`, head `4e-4` (nhẹ hơn v2 để không quên 40 method đã sửa) |

Hai đòn bẩy (giống hệt v5_weakfix v3):
1. **Nhiều data hơn:** +8,000 frame faceswap identity-disjoint (12,600 tổng).
2. **Được nhìn nhiều hơn:** faceswap chiếm 35% số batch mỗi epoch (gấp ~10 lần v2).

## 4. Scripts

| Script | Việc làm |
|---|---|
| `scripts/finetune_convnext_weakfix.py` | 1 script 3 stage (`--stage baseline|v2|v3`, `--init-ckpt`) |
| `scripts/eval_convnext_weakfix.py` | Eval baseline/v2/v3 trên benchmark, per-method, faceswap detail |

Tái lập:

```bash
PY=/workspace/hoangtuan/deepfake-ViT/.venv/bin/python
$PY scripts/finetune_convnext_weakfix.py --stage baseline
$PY scripts/finetune_convnext_weakfix.py --stage v2 --init-ckpt outputs/finetune/convnext_baseline.pt
$PY scripts/finetune_convnext_weakfix.py --stage v3 --init-ckpt outputs/finetune/convnext_weakfix_v2.pt
$PY scripts/eval_convnext_weakfix.py
```

→ Kết quả chi tiết: [04_ket_qua.md](04_ket_qua.md)

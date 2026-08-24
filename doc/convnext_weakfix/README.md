# Vá method yếu cho model dinov3_next_cnn (ConvNeXt-Tiny) — replicate phương pháp v5_weakfix

> **Ngày thực hiện:** 24/08/2026
> **Model gốc:** `dinov3_next_cnn` (DINOv3 ConvNeXt-Tiny, `/workspace/quangmanh/deepfake/models/dinov3_next_cnn`)
> **Phương pháp:** replicate nguyên bản `doc/v5_weakfix` (hoangtuan) — cùng data, cùng kỹ thuật, cùng hyperparam
> **Bản tốt nhất (final):** **v3** — Test acc **99.70%** (baseline 99.41%)
> **So với v5:** v3 97.88% → **99.70%**; faceswap v5 v3 88.89% → **100%**

## Mục lục

| File | Nội dung |
|---|---|
| [01_muc_tieu_va_model_goc.md](01_muc_tieu_va_model_goc.md) | Mục tiêu, model gốc `dinov3_next_cnn`, test benchmark, phân tích method yếu |
| [02_data.md](02_data.md) | **Data dùng chung** với v5_weakfix (identity-disjoint, dataset v2/v3) |
| [03_finetune.md](03_finetune.md) | Replicate recipe: baseline → v2 (method-balanced) → v3 (faceswap-focused) |
| [04_ket_qua.md](04_ket_qua.md) | **Kết quả** baseline → v2 → v3, per-method, faceswap chi tiết, regression |
| [05_ket_luan.md](05_ket_luan.md) | Kết luận, hạn chế còn lại, hướng phát triển |

## Tóm tắt 1 trang

1. **Model gốc** (`dinov3_next_cnn`) là DINOv3 **ConvNeXt-Tiny** (27.8M tham số, feature 768-d)
   phân loại deepfake nhị phân. Không có checkpoint classifier sẵn → phải train "baseline"
   trước (tương đương v5 của hoangtuan) rồi mới áp dụng quy trình vá method yếu.
2. **Phương pháp = nguyên bản `doc/v5_weakfix`**: dùng đúng các CSV train v2/v3 do hoangtuan build
   (`train_v5_weakfix.csv` 121,884 ảnh; `train_v5_weakfix_v3.csv` 129,884 ảnh, faceswap 12,600),
   test trên cùng benchmark zero-leakage 2,354 ảnh. Ràng buộc **identity-disjoint** được giữ
   (`identity_overlap_added = 0` đã verify từ bản build gốc).
3. **Pipeline 3 bước** — cùng hyperparam v5_weakfix:
   - **baseline**: 3 epoch trên `train_v5_combined_universal_kaggle_boost.csv` (54,000 ảnh),
     base_lr 2e-5 / head 5e-4, EMA 0.999, LabelSmoothingCE 0.05 → "v5-equivalent" của ConvNeXt.
   - **v2**: init từ baseline, sampler **method-balanced**, 2 epoch trên dataset v2.
   - **v3**: init từ v2, sampler **faceswap-focused** P(faceswap)=0.35, 3 epoch trên dataset v3.
4. Kết quả: baseline **99.41%** (FP 0 / FN 14) → v2 **99.66%** (FN 8) → **v3 99.70%**
   (FP 1 / FN 6, AUC 99.99%). **faceswap 88.89% → 85.19% (v2) → 100% (v3, 27/27)** —
   bắt đủ 10/10 case khó của v5 kể cả `ffc:701`. Mọi stage đều cao hơn v5 tương ứng 1.8–4pp.

**Checkpoint final:** `/workspace/quangmanh/deepfake/outputs/finetune/convnext_weakfix_v3.pt`

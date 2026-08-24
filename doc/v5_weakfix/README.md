# Vá method yếu cho model v5 (hoangtuan) — DINOv3 deepfake

> **Ngày thực hiện:** 23–24/08/2026
> **Model gốc:** v5 của hoangtuan (`/workspace/hoangtuan/deepfake-ViT`)
> **Bản tốt nhất (final):** **v3** — Test acc **97.88%** (baseline 95.37%)
> **Cải thiện chính:** faceswap **62.96% → 88.89%**, starganv2 **72.5% → 95%**, whichfaceisreal **73.3% → 93.3%**, facedancer **74.1% → 92.6%**

## Mục lục

| File | Nội dung |
|---|---|
| [01_muc_tieu_va_model_goc.md](01_muc_tieu_va_model_goc.md) | Mục tiêu, model gốc v5, test benchmark, phân tích method yếu |
| [02_data.md](02_data.md) | **Data ở đâu + xây data thế nào** (identity-disjoint, dataset v2/v3) |
| [03_finetune.md](03_finetune.md) | Finetune v2 (method-balanced) + chẩn đoán faceswap + finetune v3 (faceswap-focused) |
| [04_ket_qua.md](04_ket_qua.md) | **Kết quả** baseline → v2 → v3, per-method, faceswap chi tiết, regression |
| [05_ket_luan.md](05_ket_luan.md) | Kết luận, hạn chế còn lại, hướng phát triển |

## Tóm tắt 1 trang

1. **Model gốc** (v5, hoangtuan) là DINOv3 ViT-Small/16 phân loại deepfake nhị phân, đạt
   **95.37%** acc trên test 2,354 ảnh (1:1 real/fake, 40 method, zero-leak). Có **12 method sụp < 90%**,
   tập trung vào **face-swap / reenactment / talking-head / GAN-edit** — ảnh fake trông quá giống thật.
2. **Nguyên nhân gốc:** v5 train chỉ dùng **~600 frame/method** từ `DF40_train_extracted`
   (pool thật có 22K–62K) vì chủ đích **identity-disjoint** (train không trùng nhân vật test).
   Không đủ frame để học đặc trưng từng method.
3. **Data bổ sung:** Mạnh thêm `/workspace/data/celebvhq` (real) + `/workspace/data/deep-fake-face-swap`
   (fake face-swap). Em khai thác thêm pool `DF40_train_extracted` + `df-40-test-full` (đã loại path test),
   **giữ nguyên ràng buộc identity-disjoint** (verify 0 overlap).
4. **v2** (finetune 2 epoch, sampler method-balanced): 97.20%, sửa được 10/12 method sụp,
   **nhưng faceswap kẹt 62.96%**.
5. **v3** (finetune từ v2, 3 epoch, sampler **faceswap-focused** + thêm 8,000 frame faceswap identity-disjoint):
   **97.88%**, faceswap **88.89%**, FN 78→28, FP 31→22.

**Checkpoint final:** `/workspace/hoangtuan/deepfake-ViT/experiments/checkpoints/exp05_v5_weakfix_v3/best_model.pt`

# 04 — Kết quả

## 1. Tổng quan baseline → v2 → v3

| Chỉ số | baseline v5 | v2 (weakfix) | **v3 (faceswap-focused) ★** |
|---|---|---|---|
| **Test Accuracy** | 95.37% | 97.20% | **97.88%** |
| **ROC-AUC** | 99.35% | 99.73% | 99.70% |
| Precision | 97.26% | 98.10% | 98.12% |
| Recall (fake) | 93.37% | 96.26% | **97.62%** |
| F1 | 95.28% | 97.17% | 97.87% |
| Real acc (FP) | 97.37% (**31**) | 98.13% (**22**) | 98.13% (**22**) |
| Fake recall (FN) | 93.37% (**78**) | 96.26% (**44**) | **97.62% (28)** |
| Confusion matrix | [[1146,31],[78,1099]] | [[1155,22],[44,1133]] | [[1155,22],[28,1149]] |

**Đọc nhanh:** v3 cải thiện đáng kể fake recall (bắt được nhiều fake hơn: FN 78→28) trong khi
giữ nguyên FP (22) — tức không hy sinh độ chính xác trên real để lấy recall. AUC gần như không đổi.

## 2. Per-method — v3 so baseline (mọi method sụp cũ)

| Method | Loại | baseline | v2 | **v3** | Δ(v3−base) |
|---|---|---|---|---|---|
| **faceswap** | face-swap | 62.96% | 62.96% | **88.89%** | +25.9 |
| **starganv2** | GAN edit | 72.5% | 100% | **95.0%** | +22.5 |
| **whichfaceisreal** | GAN | 73.33% | 90.0% | **93.33%** | +20.0 |
| **facedancer** | reenactment | 74.07% | 81.48% | **92.59%** | +18.5 |
| **sadtalker** | talking-head | 80.77% | 84.62% | **80.77%** | 0 |
| **fsgan** | face-swap | 84.62% | 92.31% | **92.31%** | +7.7 |
| **wav2lip** | lip-sync | 86.36% | 86.36% | **90.91%** | +4.6 |
| **e4s** | GAN edit | 86.67% | 100% | **100%** | +13.3 |
| **simswap** | face-swap | 88.89% | 96.30% | **100%** | +11.1 |
| **blendface** | face-swap | 88.89% | 92.59% | **92.59%** | +3.7 |
| **lia** | face-swap | 88.89% | 85.19% | **92.59%** | +3.7 |
| **inswap** | face-swap | 94.74% | 89.47% | **94.74%** | 0 |
| CollabDiff | diffusion | 90.32% | 96.77% | **100%** | +9.7 |
| pirender | 92.59% | 96.30% | **96.30%** | +3.7 |
| uniface | 96.30% | 96.30% | **100%** | +3.7 |
| mobileswap | 96.43% | 98.21% | **100%** | +3.6 |
| SiT / e4e | 95.00% | 100% | 97.5% | +2.5 |
| pixart | 97.30% | 97.30% | **100%** | +2.7 |
| **real** | real | 97.37% | 98.13% | **98.13%** | +0.8 |
| sd2.1, StyleGAN2/3/XL, VQGAN, stargan, fomm, ddim, danet, mcnet, hyperreenact, MRAA, facevid2vid, tpsm, RDDM, DiT, deepfacelab, heygen, MidJourney | … | 95–100% | 95–100% | 95–100% | ~0 |

## 3. faceswap — chi tiết từng ảnh (10 miss của baseline/v2)

| identity | nguồn | base | v2 | **v3** | kết quả |
|---|---|---|---|---|---|
| ffc:462 | FF++ | 0.169 | 0.165 | **0.958** | ✅ sửa |
| ffc:812 | FF++ | 0.178 | 0.104 | **0.960** | ✅ sửa |
| ffc:176 | FF++ | 0.133 | 0.237 | **0.603** | ✅ sửa |
| ffc:044 | FF++ | 0.124 | 0.197 | **0.978** | ✅ sửa |
| ffc:701 | FF++ | 0.074 | 0.027 | **0.027** | ❌ còn miss |
| oth:id3_id28 | VoxCeleb | 0.440 | 0.445 | **0.759** | ✅ sửa |
| oth:id3_id23 | VoxCeleb | 0.168 | 0.276 | **0.328** | ❌ còn miss |
| oth:id4_id28 | VoxCeleb | 0.637 | 0.478 | **0.615** | ✅ sửa |
| cdc:id50 | Celeb-DF | 0.405 | 0.283 | **0.828** | ✅ sửa |
| cdc:id10 | Celeb-DF | 0.639 | 0.491 | **0.480** | ❌ còn miss |

→ **7/10 miss đã sửa.** 3 ca còn lại:
- `ffc:701` — FF++ frame **cực sắc** (sharpness 527), model vẫn cực tự tin real (0.027).
- `oth:id3_id23` — VoxCeleb ngoài phân phối train (0.328).
- `cdc:id10` — Celeb-DF, sát ngưỡng (0.480).

## 4. Regression nhỏ (v3 so v2) — đều mức nhiễu

| Method | v2 | v3 | Δ | Giải thích |
|---|---|---|---|---|
| starganv2 | 100% | 95.0% | −5.0 | n=40, mất 2 ảnh; vẫn cao hơn baseline 22.5 điểm |
| sadtalker | 84.62% | 80.77% | −3.85 | mất đúng **1 frame biên** (`id2_da1vvigy5tQ`, prob 0.509→0.426) |

> Không method nào khác sụt quá 2.5 điểm. Nguyên nhân: sampler v3 giảm trọng số các method
> không phải faceswap (~0.86% thay vì ~1.4%/epoch) — đổi lại faceswap tăng ~10 lần.

## 5. Files kết quả (hoangtuan repo)

| File | Nội dung |
|---|---|
| `experiments/checkpoints/exp05_v5_weakfix_v3/best_model.pt` | **Checkpoint final (v3)** — dùng để deploy |
| `experiments/checkpoints/exp05_v5_weakfix/best_model.pt` | Checkpoint v2 |
| `experiments/results/v5_weakfix_v3_training_report.json` | Báo cáo v3 (metrics + method breakdown) |
| `experiments/results/v5_weakfix_v3_per_method_accuracy.csv` | Per-method v3 |
| `experiments/results/v5_weakfix_v3_vs_baseline.csv` | v3 so baseline |
| `experiments/results/v5_weakfix_v3_vs_v2.csv` | v3 so v2 |
| `experiments/results/v5_weakfix_analysis.md` | Phân tích đầy đủ (baseline + v2 + v3) |
| `experiments/results/v5_weakfix_v3_dataset_summary.json` | Tóm tắt dataset v3 + xác nhận 0 identity overlap |

→ Kết luận & hướng đi: [05_ket_luan.md](05_ket_luan.md)

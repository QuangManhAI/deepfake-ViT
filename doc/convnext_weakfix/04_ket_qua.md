# 04 — Kết quả

> Toàn bộ số liệu dưới đây theo **đúng protocol của v5_weakfix** (eval bf16 autocast,
> BICUBIC 256, batch 128, ngưỡng 0.5) — tái sinh từ `scripts/eval_convnext_weakfix.py`.

## 1. Tổng quan baseline → v2 → v3 (ConvNeXt-Tiny)

| Chỉ số | baseline ConvNeXt | v2 (method-balanced) | **v3 (faceswap-focused) ★** |
|---|---|---|---|
| **Test Accuracy** | 99.41% | 99.66% | **99.70%** |
| **ROC-AUC** | 99.98% | 99.99% | 99.99% |
| Precision | 100.00% | 100.00% | 99.91% |
| Recall (fake) | 98.81% | 99.32% | **99.49%** |
| F1 | 99.40% | 99.66% | 99.70% |
| Real acc (FP) | 100.00% (**0**) | 100.00% (**0**) | 99.92% (**1**) |
| Fake recall (FN) | 98.81% (**14**) | 99.32% (**8**) | **99.49% (6)** |
| Confusion matrix | [[1177,0],[14,1163]] | [[1177,0],[8,1169]] | [[1176,1],[6,1171]] |

**Đọc nhanh:** baseline ConvNeXt đã rất mạnh (99.41% — cao hơn hẳn baseline v5 95.37%).
v2 hạ FN 14→8, v3 hạ tiếp 8→6 trong khi chỉ đổi lấy 1 FP. Mọi chỉ số đều **cao hơn kết quả
tương ứng của v5** trong `doc/v5_weakfix`:

| Stage | v5 (ViT-S/16) | **ConvNeXt-Tiny** | Δ |
|---|---|---|---|
| baseline | 95.37% | **99.41%** | +4.04 |
| v2 | 97.20% | **99.66%** | +2.46 |
| **v3** | 97.88% | **99.70%** | +1.82 |

## 2. Per-method — baseline → v2 → v3 (method sụp ở baseline hoặc có thay đổi)

| Method | Loại | N | baseline | v2 | **v3** | Δ(v3−base) |
|---|---|---|---|---|---|---|
| **heygen** | talking-head | 1 | 0.00% | 100% | **100%** | +100.0 |
| **sadtalker** | talking-head | 26 | 88.46% | 96.15% | **92.31%** | +3.85 |
| **faceswap** | face-swap | 27 | 88.89% | 85.19% | **100%** | +11.11 |
| **fsgan** | face-swap | 26 | 96.15% | 96.15% | **96.15%** | 0 |
| **facedancer** | reenactment | 27 | 96.30% | 96.30% | **96.30%** | 0 |
| **lia** | face-swap | 27 | 96.30% | 96.30% | **96.30%** | 0 |
| **CollabDiff** | diffusion | 31 | 96.77% | 100% | **100%** | +3.23 |
| **SiT** | diffusion | 40 | 97.50% | 100% | **100%** | +2.50 |
| **mobileswap** | face-swap | 56 | 98.21% | 100% | **100%** | +1.79 |
| **sd2.1** | diffusion | 64 | 98.44% | 100% | **100%** | +1.56 |
| **mcnet** | reenactment | 27 | 100% | 100% | **96.30%** | −3.70 |
| starganv2, whichfaceisreal, e4s, simswap, blendface, wav2lip, styleclip, inswap, pirender, uniface, fomm, … (29 method) | … | … | 100% | 100% | **100%** | 0 |
| **real** | real | 1177 | 100% | 100% | **99.92%** | −0.08 |

> Khác biệt lớn so với v5: **starganv2 và whichfaceisreal — 2 method v5 baseline sụp nặng
> (72.5%, 73.33%) — ConvNeXt baseline đã bắt 100% từ đầu**, không cần vá. Với ConvNeXt, nhóm
> "method yếu" hẹp hơn nhiều: chỉ heygen (n=1), sadtalker, faceswap, fsgan, facedancer, lia
> dưới 100% ở baseline.

## 3. faceswap — chi tiết (10 case khó của v5 vs ConvNeXt)

Bảng dưới là prob của **chính 10 ảnh mà v5 baseline miss** (theo `doc/v5_weakfix/04`) qua
3 stage của ConvNeXt:

| identity | nguồn | v5 v3 | ConvNeXt base | ConvNeXt v2 | **ConvNeXt v3** |
|---|---|---|---|---|---|
| ffc:462 | FF++ | 0.958 ✅ | 0.062 ❌ | 0.052 ❌ | **0.952 ✅** |
| ffc:812 | FF++ | 0.960 ✅ | 0.043 ❌ | 0.052 ❌ | **0.969 ✅** |
| ffc:176 | FF++ | 0.603 ✅ | 0.042 ❌ | 0.048 ❌ | **0.941 ✅** |
| ffc:701 | FF++ | 0.027 ❌ | — ✅ | — ✅ | **0.966 ✅** |
| ffc:044 | FF++ | 0.978 ✅ | — ✅ | — ✅ | **0.975 ✅** |
| oth:id3_id28 | VoxCeleb | 0.759 ✅ | — ✅ | — ✅ | **0.969 ✅** |
| oth:id3_id23 | VoxCeleb | 0.328 ❌ | — ✅ | — ✅ | **0.974 ✅** |
| oth:id4_id28 | VoxCeleb | 0.615 ✅ | — ✅ | — ✅ | **0.918 ✅** |
| cdc:id50 | Celeb-DF | 0.828 ✅ | — ✅ | — ✅ | **0.968 ✅** |
| cdc:id10 | Celeb-DF | 0.480 ❌ | — ✅ | — ✅ | **0.974 ✅** |

→ **ConvNeXt v3 bắt đủ 10/10** case khó của v5 — kể cả `ffc:701` (FF++ siêu sắc mà v5 v3
bất lực, prob 0.027) và `oth:id3_id23`, `cdc:id10`.

Tóm tắt faceswap qua các stage (27 ảnh):

| Stage | Số miss | Chi tiết |
|---|---|---|
| baseline | 3 | ffc:176, ffc:462, ffc:812 (prob 0.04–0.06 — tự tin sai) |
| v2 | 4 | 3 FF++ trên + thêm `oth:faceswap:id1_id3_0009` (prob 0.288) |
| **v3** | **0** | **27/27 — 100%** |

## 4. Regression nhỏ (v3 so v2 / so baseline) — trung thực

| Method | baseline | v2 | v3 | Ghi chú |
|---|---|---|---|---|
| faceswap | 88.89% | 85.19% | **100%** | v2 tụt 1 (FN 3→4) nhưng v3 gỡ hết (FN 0) |
| mcnet | 100% | 100% | 96.30% | v3 mất đúng 1 ảnh (FN 1) — mức nhiễu |
| sadtalker | 88.46% | 96.15% | 92.31% | v3 mất 1 so v2 nhưng vẫn cao hơn baseline |
| real | 100% | 100% | 99.92% | v3 có 1 FP (1 ảnh real bị gán fake) |

> 6 FN còn lại của v3: sadtalker 2, fsgan 1, facedancer 1, lia 1, mcnet 1 — đều là ảnh đơn lẻ
> sát ngưỡng, không tập trung vào một method nào. FP 1 ở real.

## 5. Files kết quả (repo này)

| File | Nội dung |
|---|---|
| `outputs/finetune/convnext_baseline.pt` | Checkpoint baseline (v5-equivalent) |
| `outputs/finetune/convnext_weakfix_v2.pt` | Checkpoint v2 |
| `outputs/finetune/convnext_weakfix_v3.pt` | **Checkpoint final (v3)** |
| `experiments/results/convnext_weakfix/eval_{baseline,v2,v3}.json` | Metrics từng stage |
| `experiments/results/convnext_weakfix/per_method_{baseline,v2,v3}.csv` | Per-method từng stage |
| `experiments/results/convnext_weakfix/faceswap_detail.csv` | faceswap per-identity (v3) |
| `experiments/results/convnext_weakfix/comparison.json` | So sánh baseline → v3 |
| `experiments/results/convnext_weakfix/*_training_report.json` | Báo cáo train từng stage |

→ Kết luận & hướng đi: [05_ket_luan.md](05_ket_luan.md)

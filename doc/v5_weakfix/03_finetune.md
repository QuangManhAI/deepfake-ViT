# 03 — Finetune: v2 (method-balanced) → chẩn đoán faceswap → v3 (faceswap-focused)

## 0. Môi trường

- GPU: RTX 3060 12GB (dùng chung — chạy 1 tiến trình train/lúc, batch vừa phải).
- Python venv: `/workspace/hoangtuan/deepfake-ViT/.venv/bin/python`
  (torch 2.6.0, pandas 2.3.3, opencv, sklearn; bf16 autocast).
- Loss: `LabelSmoothingCrossEntropy(0.05)`; Optimizer: AdamW; EMA 0.999;
  Scheduler: CosineAnnealing; tất cả finetune init từ checkpoint có sẵn (không đụng ckpt gốc).

## 1. v2 — finetune tổng quát (`scripts/finetune_v5_weakfix.py`)

| Cấu hình | Giá trị |
|---|---|
| Init | `exp05_v5_combined/best_model.pt` (0 missing / 0 unexpected) |
| Train | `data/splits/train_v5_weakfix.csv` (121,884 = 31,006R/90,878F) |
| Val | `val_v5_combined_universal_kaggle_boost.csv` (6,000) |
| Test | zero-leakage benchmark (2,354) |
| Sampler | **method-balanced** — P(real)=0.5; fake chia đều `0.5/num_methods` → mọi method yếu được nhìn ngang nhau |
| Số mẫu/epoch | `2 × num_real` ≈ 62,012 |
| Epochs / batch | 2 / 64 |
| LR | backbone `2e-5`, head `5e-4`, weight_decay 0.05 |
| Aug | HorizontalFlip, ColorJitter, GaussianBlur, RandomSharpness |

**Kết quả v2:** test acc **97.20%** (AUC 99.73), FP 31→22, FN 78→44. Sửa được 10/12 method sụp
(starganv2→100, whichfaceisreal→90, e4s→100, fsgan→92.3, simswap→96.3, CollabDiff→96.8, …).
**Nhưng faceswap kẹt nguyên 62.96%** (17/27 — miss đúng 10 ảnh giống hệt baseline).

## 2. Chẩn đoán faceswap — vì sao không học được?

### 2.1 Nhìn vào 10 miss

Model baseline lẫn v2 miss đúng **10 ảnh** faceswap test. Chia theo nguồn:
- **5 FF++** (`ffc:462, 812, 176, 701, 044`): model **cực tự tin sai** — prob 0.027–0.237.
- **3 VoxCeleb** (`oth:id3_id28, id3_id23, id4_id28`): 0.28–0.48.
- **2 Celeb-DF** (`cdc:id50, id10`): 0.28–0.49.

### 2.2 Thống kê ảnh: miss ≠ hit

| Nhóm | Sharpness (median) | Brightness | File size |
|---|---|---|---|
| Test faceswap **hit** (17) | 14.3 | 68.9 | 69KB |
| Test faceswap **miss** (10) | **64.5** | 79.9 | 85KB |
| DF40 faceswap **train** | 94.4 | 103.9 | 91KB |

→ **Miss là những frame SẮC NÉT, chất lượng cao** (sharpness tới 527 cho `ffc:701`), không có
artifact nén/khử ảnh để "bắt bài". Model học được các frame mờ (artifact rõ) nhưng không nhận ra
frame face-swap sạch. Các method face-swap khác (fsgan/simswap/blendface/lia) test đều **mờ**
(sharp 13–23) nên dễ hơn.

### 2.3 Model thực tế chưa học faceswap

Đo prob trên **chính frame faceswap train** (800 mẫu):

| Model | Mean prob | % gán fake |
|---|---|---|
| baseline-v5 | 0.301 | 20.5% |
| v2-weakfix | 0.475 | 45.4% |

→ baseline coi 80% frame faceswap train là real. v2 đẩy lên 0.475 nhưng chưa qua ngưỡng 0.5.
**Nguyên nhân:** sampler method-balanced chỉ cho faceswap ~1.4% batch mỗi epoch
(n faceswap = 4,600 trong 121K, mỗi epoch chỉ rút ~870 lần từ nhóm này). Đã thêm +4,000 frame
faceswap nhưng không đủ để học một lớp khó như vậy trong 2 epoch.

## 3. v3 — faceswap-focused (`scripts/finetune_v5_weakfix_v3.py`)

| Cấu hình | Giá trị |
|---|---|
| Init | **`exp05_v5_weakfix/best_model.pt`** (v2) |
| Train | `data/splits/train_v5_weakfix_v3.csv` (129,884 = 31,006R/98,878F; faceswap 12,600) |
| Sampler | **faceswap-focused** — P(faceswap)=**0.35**, P(real)=0.35, P(method khác)=0.30 chia đều 35 method |
| Số mẫu/epoch | `2 × num_real` ≈ 62,012 (faceswap ~21,700 lần/epoch → ~4.7 vòng/frame) |
| Epochs / batch | **3** / 64 |
| LR | backbone `1.5e-5`, head `4e-4` (nhẹ hơn v2 để không quên 40 method đã sửa) |

Hai đòn bẩy chính:
1. **Nhiều data hơn:** +8,000 frame faceswap identity-disjoint (12,600 tổng).
2. **Được nhìn nhiều hơn:** faceswap chiếm 35% số batch mỗi epoch (gấp ~10 lần v2).

**Kết quả v3:** test acc **97.88%** (AUC 99.70), FP 22 (giữ nguyên), FN 44→**28**.
faceswap **62.96% → 88.89%** (24/27 — sửa được **7/10** miss).

## 4. Lưu ý kỹ thuật (gotcha)

- Hàm `evaluate()` trong finetune script trả key **`prec` / `rec` / `auc`** — không phải
  `precision` / `recall` / `roc_auc`. Do đó đoạn `report["test_metrics"]` trong script finetune
  sẽ **crash KeyError ở cuối** (sau khi đã in xong bảng kết quả). Không ảnh hưởng checkpoint
  (đã lưu trước đó) — chỉ cần regenerate report bằng
  `scripts/eval_v5_weakfix_v3_report.py` (load ckpt, eval test, ghi lại JSON/CSV).
- Kiểm tra identity-disjoint sau khi build: `v5_weakfix_v3_dataset_summary.json` có
  `identity_overlap_added = 0`.

→ Kết quả chi tiết: [04_ket_qua.md](04_ket_qua.md)

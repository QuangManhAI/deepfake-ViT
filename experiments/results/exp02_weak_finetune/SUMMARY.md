# Finetune exp02 → vá method yếu — so sánh 5 checkpoint

Test: `test_balanced.csv` (2.354 ảnh, cân bằng 1.177 real / 1.177 fake, BICUBIC).

## Tổng quan

| Metric | exp02_best (gốc) | replay (16K) | max_v1 (152K) | max_v2 (169K, real/) | **max_v3 (152K + gif)** |
|---|---|---|---|---|---|
| Accuracy | 95.79% | 96.86% | 97.54% | 92.35% ❌ | **98.05%** |
| ROC-AUC | 0.9866 | 0.9937 | 0.9959 | 0.9837 | **0.9980** |
| FN (fake bỏ sót) | 53 | 30 | 33 | **154** ❌ | **17** |
| FP (real nhầm) | 46 | 44 | **25** | 26 | 29 |

## Real theo domain (không sụp — quan trọng nhất)

| Domain | n | exp02_best | replay | max_v1 | max_v2 | max_v3 |
|---|---|---|---|---|---|---|
| cdc (Celeb-DF) | 178 | 91.6% | 99.4% | 100.0% | 100.0% | **100.0%** |
| ffc (FF++) | 999 | 96.9% | 95.7% | **97.5%** | 97.5% | 97.1% |
| Total real | 1177 | 96.1% | 96.3% | **97.9%** | 97.8% | 97.5% |

## Method yếu (FN theo từng checkpoint)

| Method | n | exp02_best | replay | max_v1 | max_v2 | **max_v3** |
|---|---|---|---|---|---|---|
| **MidJourney** | 26 | 16 | 14 | 14 | 2 | **0** 🎯 |
| whichfaceisreal | 30 | 10 | 1 | 1 | 12 | **0** |
| faceswap | 27 | 6 | 2 | 1 | 1 | **1** |
| styleclip | 40 | 5 | 0 | 0 | 2 | **0** |
| CollabDiff | 31 | 4 | 0 | 1 | 4 | **1** |
| sadtalker | 26 | 2 | 3 | 1 | 6 | **1** |
| wav2lip | 22 | 2 | 2 | 1 | 2 | **1** |
| heygen | 1 | 1 | 1 | 1 | 1 | **1** (n=1) |
| danet | 27 | 0 | 0 | 1 | 9 | **0** |
| mcnet | 27 | 0 | 0 | 1 | 4 | **0** |
| stargan | 40 | 0 | 0 | 0 | 6 | **0** |
| starganv2 | 40 | 0 | 0 | 0 | 6 | **0** |

## Đọc kết quả

- **max_v3 là model khuyến nghị cuối cùng** (`outputs/finetune/exp02_weak_max_v3.pt`): acc 98.05%, **AUC 0.9980, FN 17** — đều tốt nhất. **MidJourney FN 14 → 0** (trước đây xem như "bất khả phá" do chỉ có 5 ảnh sạch). Real giữ 97.5% (cdc 100%, ffc 97.1%).
- **Điều gì làm MidJourney hết bị miss:** thêm **247 gif frames** `MidJourney/fake/*.gif` vào fake pool (252 sạch = 5 png + 247 gif). Gif là các frame sạch single-frame MidJourney không nằm trong test.
- **Bài học: KHÔNG thêm `real/` folders của df-40-test-full vào real pool.** max_v2 (real = max_v1 + 17K real/) sụp: acc 92.35%, **FN 33 → 154** lan rộng cả method không có real/ (SiT 12, StyleGANXL 12, DiT 11, VQGAN 9...). Nguyên nhân: 17K ảnh clean source kiểu pipeline DF40 làm boundary dịch về "real" → model chốt fake thành real. MidJourney 14→2 trong v2 thực ra là do gif, không phải real/.
- **max_v3 = max_v1 + gif (không real/)**: tách 2 thay đổi, chứng minh gif là fix đúng, real/ phải loại.

## Dữ liệu finetune

- `data/finetune_exp02/train_replay.csv` (16.221 rows) — real = 5.2K FF++ replay + 2.9K celeb, fake 8.1K.
- `data/finetune_exp02/train_max.csv` (**152.410 rows, bản rebuild mặc định hiện tại**) — real 53.5K (FF++ replay 27.6K + celeb 25.9K), fake 98.9K (12 method yếu toàn bộ clean pool, gồm MidJourney 252 = 5 png + 247 gif). Imbalance xử lý bằng **WeightedRandomSampler** (cân bằng per-step) + val stratify.
- `data/finetune_exp02/train_max_v3.csv` = cùng bản với train_max.csv khi build `--no-real-dir` (real/ bị loại). Build flag: `--no-real-dir` / `--out`.
- **0 path trùng chính xác** test manifest/test_balanced (assert trong build). FF++ real replay là video-level (user chấp thuận).

## Checkpoints

| Checkpoint | Data | Kết quả |
|---|---|---|
| `exp02_weak_replay.pt` | train_replay.csv | acc 96.86% |
| `exp02_weak_max.pt` (v1) | train_max.csv (152K) | acc 97.54% |
| `exp02_weak_max_v2.pt` | train_max.csv (169K + real/) | acc 92.35% ❌ |
| **`exp02_weak_max_v3.pt`** | train_max_v3.csv (152K + gif) | **acc 98.05% — khuyến nghị** |

## Scripts

- `scripts/survey_data_expansion.py` → `experiments/results/data_expansion_survey.md` (headroom từng method)
- `scripts/build_finetune_max.py` (flag `--no-real-dir`/`--out`), `scripts/rebuild_finetune_replay.py`, `scripts/finetune_exp02.py`, `scripts/eval_exp02.py`

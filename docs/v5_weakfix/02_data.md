# 02 — Data: ở đâu & xây thế nào

## 1. Vị trí data

| Nguồn | Đường dẫn | Nội dung |
|---|---|---|
| **Test benchmark** | `/workspace/data/zero_leakage_benchmark_fixed/test_balanced_fixed_zero_leakage.csv` | 2,354 ảnh, 1:1, 40 method, zero-leak |
| **Ảnh test gốc** | `/workspace/data/test_data_v3/<method>/{fake,real}/` | ảnh 256×256 |
| **DF40 train pool** | `/workspace/data/DF40_train_extracted/<method>/frames/<idA_idB>/*.png` | 22K–62K frame/method (dùng để boost) |
| **df-40-test-full** | `/workspace/data/df-40-test-full/<method>/{fake,real}/` | pool test-full (dùng 1 phần, đã loại path test) |
| **celebvhq (MỚI — Mạnh thêm)** | `/workspace/data/celebvhq/35666/*.mp4` | video real người nổi tiếng |
| **celebvhq frames (extract)** | `/workspace/data/celebvhq_frames/real/*.png` | 4,000 frame real 512×512 |
| **deep-fake-face-swap (MỚI — Mạnh thêm)** | `/workspace/data/deep-fake-face-swap/images/train/*.jpg` | 8,076 ảnh face-swap fake |
| **v5 train CSV (replay)** | `/workspace/hoangtuan/deepfake-ViT/data/splits/train_v5_combined_universal_kaggle_boost.csv` | 54,000 ảnh v5 đã dùng |

## 2. Ràng buộc identity-disjoint (cốt lõi)

Người dùng làm rõ: **v5 train chỉ dùng ~600 frame/method là vì muốn train/test không trùng
nhân vật (identity)**. Vì vậy mọi frame bổ sung đều phải loại trừ nhân vật đã xuất hiện trong test.

**Cách kiểm tra:** mỗi frame train thuộc folder `<idA_idB>` (2 nhân vật). Test có cột `identity`
dạng `ffc:N`, `oth:...:idA_idB`, `cdc:idN`. Nếu folder frame trùng bất kỳ token số nào của test
identity method đó → **loại frame**.

```
test tokens của method M = mọi số trích từ identity + basename 'M__A__B' của test M
frame tokens             = mọi số trong tên folder (vd '786_819' -> {786, 819})
giữ frame  <=>  frame_tokens ∩ test_tokens = ∅
```

Script: `scripts/build_finetune_v5_weakfix.py` (`method_test_identity_tokens` / `frame_identity_tokens`).

## 3. Dataset v2 — `data/splits/train_v5_weakfix.csv`

**121,884 ảnh = 31,006 real / 90,878 fake.** Thành phần:

| Thành phần | N | Ghi chú |
|---|---|---|
| **Replay v5 train** | 54,000 | Giữ nguyên hành vi cũ (chống forgetting) |
| **Boost DF40_train_extracted** | 51,600 | Fake, identity-disjoint, dedup vs v5 (xem bảng dưới) |
| **deep-fake-face-swap** | 8,076 | Fake, method `deepfake_faceswap` (dữ liệu Mạnh thêm) |
| **df-40-test-full** (đã loại path test) | 4,208 | starganv2 1,748 · whichfaceisreal 660 · CollabDiff 1,000 · heygen_new 800 |
| **celebvhq frames** | 4,000 | Real, method `celebvhq_real` (dữ liệu Mạnh thêm) |

### Boost theo method (DF40_train_extracted)

| Tier | Method | Số frame/method |
|---|---|---|
| Tier 1 | faceswap, sadtalker, facedancer | 4,000 |
| Tier 2 | fsgan, simswap, blendface | 3,000 |
| Tier 2 | wav2lip, e4s, inswap | 2,500 |
| Tier 2 | lia | 2,000 |
| Tier 3 | mobileswap | 1,500 |
| Tier 3 | one_shot_free, uniface, pirender | 1,200 |
| Tier 3 | ddim, SiT, pixart, tpsm, MRAA, mcnet, danet, hyperreenact, fomm, facevid2vid, VQGAN, StyleGAN2/3/XL, sd2.1, RDDM | 1,000 |
| (0 được thêm) | styleclip, stargan | 0 — pool không có frames/ |

### Số frame bị loại vì trùng identity (một số method)

| Nguồn | Số frame loại | Nguồn | Số frame loại |
|---|---|---|---|
| train_extracted/mobileswap | 3,719 | train_extracted/inswap | 784 |
| train_extracted/simswap | 622 | train_extracted/sd2.1 | 879 |
| train_extracted/fsgan | 373 | train_extracted/facedancer | 304 |
| train_extracted/MRAA | 381 | train_extracted/blendface | 184 |
| train_extracted/faceswap | 187 | train_extracted/e4s | 182 |

→ Chi tiết đầy đủ: `experiments/results/v5_weakfix_dataset_summary.json` (hoangtuan repo).

## 4. Dataset v3 — `data/splits/train_v5_weakfix_v3.csv` (bản faceswap-focused)

**129,884 ảnh = 31,006 real / 98,878 fake** = **dataset v2 + thêm 8,000 frame faceswap
identity-disjoint** nữa từ `DF40_train_extracted/faceswap` (pool có 22,852 → sau khi loại
replay v5 còn 22,267 → loại trùng identity 187 → còn 18,080 sạch).

| Chỉ số | v2 | v3 |
|---|---|---|
| Tổng | 121,884 | 129,884 |
| Số frame faceswap | 4,600 | **12,600** (+8,000) |
| Identity overlap của phần thêm | — | **0 (đã verify)** |

Script: `scripts/expand_faceswap_v3.py`.

**Điểm mạnh hoá so với v2:** hàm `all_test_identity_tokens()` thu thập **MỌI token số** từ mọi
định dạng identity test của faceswap (`ffc:N`, `oth:*:idA_idB*`, `cdc:idN`, cả frame number
trong basename) — rồi loại mọi frame train có token trùng. → Bảo đảm chặt hơn cho yêu cầu
identity-disjoint (kể cả nhân vật VoxCeleb/Celeb-DF trong test).

## 5. Các script build data

| Script | Việc làm |
|---|---|
| `/workspace/hoangtuan/deepfake-ViT/scripts/prepare_celebvhq_frames.py` | Extract 200 video celebvhq × 20 frame (512×512) → `/workspace/data/celebvhq_frames/real/` |
| `scripts/build_finetune_v5_weakfix.py` | Build `train_v5_weakfix.csv` (v2) — replay + boost + 2 dataset mới + test-full |
| `scripts/expand_faceswap_v3.py` | Build `train_v5_weakfix_v3.csv` (v3) — thêm +8,000 faceswap identity-disjoint, verify 0 overlap |

## 6. Tóm tắt cách phối hợp 2 dataset Mạnh thêm

- **`deep-fake-face-swap` (8,076 fake):** đưa thẳng vào train với nhãn `deepfake_faceswap`
  → cung cấp face-swap "mới lạ" (swap lên celeb), giúp model thấy đa dạng kiểu face-swap.
- **`celebvhq` (4,000 real):** extract frame, đưa vào với nhãn `celebvhq_real`
  → thêm real đa dạng ngoài FFHQ, giúp giảm FP (real bị gán fake).

→ Cách huấn luyện trên data này: [03_finetune.md](03_finetune.md)

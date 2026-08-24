# 02 — Data: dùng chung bộ data của v5_weakfix

## 1. Nguyên tắc

Thay vì build lại data, project này **tái sử dụng nguyên bản** các CSV mà hoangtuan đã build
cho v5_weakfix (cùng máy, path tuyệt đối còn hiệu lực). Đây đảm bảo "cùng data" 100% — điều kiện
để so sánh phương pháp giữa 2 model (ViT-S/16 vs ConvNeXt-Tiny) là công bằng.

Các CSV được **copy vào repo** này tại `data/splits/` để project tự chứa:

| File (repo này) | Nguồn gốc | Số dòng |
|---|---|---|
| `data/splits/train_v5_combined_universal_kaggle_boost.csv` | replay v5 train | 54,000 |
| `data/splits/val_v5_combined_universal_kaggle_boost.csv` | val v5 | 6,000 |
| `data/splits/train_v5_weakfix.csv` | **dataset v2** | 121,884 |
| `data/splits/train_v5_weakfix_v3.csv` | **dataset v3** | 129,884 |
| `data/splits/v5_weakfix_v3_dataset_summary.json` | chứng nhận identity-disjoint | — |
| `/workspace/data/zero_leakage_benchmark_fixed/test_balanced_fixed_zero_leakage.csv` | test benchmark | 2,354 |

## 2. Ràng buộc identity-disjoint (cốt lõi)

Đây là lý do v5 train chỉ dùng ~600 frame/method: train/test không được trùng nhân vật.
Mọi frame bổ sung trong dataset v2/v3 đều đã loại nhân vật trùng test (token số của `identity`).

Chứng nhận từ bản build gốc (`v5_weakfix_v3_dataset_summary.json`):

```json
{
  "v2_rows": 121884,
  "total_rows": 129884,
  "faceswap_before": 4600,
  "faceswap_added": 8000,
  "faceswap_after": 12600,
  "identity_dropped_from_pool": 187,
  "identity_overlap_added": 0
}
```

→ **`identity_overlap_added = 0`** — phần faceswap thêm ở v3 không trùng bất kỳ nhân vật nào
trong test.

## 3. Thành phần dataset

### Dataset v2 — `train_v5_weakfix.csv` (121,884 ảnh = 31,006 real / 90,878 fake)

| Thành phần | N | Ghi chú |
|---|---|---|
| Replay v5 train | 54,000 | Giữ hành vi cũ (chống forgetting) |
| Boost DF40_train_extracted | 51,600 | Fake, identity-disjoint |
| deep-fake-face-swap | 8,076 | Fake, method `deepfake_faceswap` |
| df-40-test-full (đã loại path test) | 4,208 | starganv2 / whichfaceisreal / CollabDiff / heygen_new |
| celebvhq frames | 4,000 | Real, method `celebvhq_real` |

- 42 fake method; `faceswap` = 4,600 frame.
- Real pool gồm: FaceForensics++ Real, Celeb-DF Real, DF40 Real, ffhq_real, MidJourney Real,
  celebvhq_real, CollabDiff Real, starganv2 Real, whichfaceisreal Real.

### Dataset v3 — `train_v5_weakfix_v3.csv` (129,884 ảnh = 31,006 real / 98,878 fake)

= dataset v2 + **thêm 8,000 frame faceswap identity-disjoint** từ
`DF40_train_extracted/faceswap` → `faceswap` = **12,600** frame (identity overlap 0).

## 4. Tóm tắt cách phối hợp 2 dataset "mới" của Mạnh

- **`deep-fake-face-swap` (8,076 fake):** đưa vào train với nhãn `deepfake_faceswap`
  → face-swap "mới lạ" (swap lên celeb).
- **`celebvhq` (4,000 real):** thêm real đa dạng ngoài FFHQ, giúp giảm FP.

→ Cách huấn luyện trên data này: [03_finetune.md](03_finetune.md)

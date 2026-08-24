# Khảo sát mở rộng data finetune (method yếu)

| Method | Đang dùng | df-40-test-full sạch | DF40_train_extracted | **Tối đa** | Headroom |
|---|---|---|---|---|---|
| MidJourney     |      5 |      5 |          — | **      5** | +     0 |
| whichfaceisreal |    251 |    251 |          — | **    251** | +     0 |
| faceswap       |    800 |   3097 |     22,852 | ** 22,852** | +22,052 |
| styleclip      |   1200 |   1824 |          — | **  1,824** | +   624 |
| CollabDiff     |    250 |    250 |          — | **    250** | +     0 |
| sadtalker      |    600 |   4416 |     22,797 | ** 22,797** | +22,197 |
| wav2lip        |    600 |   7654 |     22,682 | ** 22,682** | +22,082 |
| heygen         |    838 |    838 |          — | **    838** | +     0 |
| stargan        |    984 |    984 |          — | **    984** | +     0 |
| starganv2      |   1000 |   1001 |          — | **  1,001** | +     1 |
| deepfacelab    |   1200 |   2342 |          — | **  2,342** | + 1,142 |
| MRAA           |    400 |   3048 |     22,811 | ** 22,811** | +22,411 |
| **TOTAL** | 8,128 | | | **98,637** | +90,509 |

## Ghi chú
- **MidJourney: hard ceiling 5 ảnh** — chỉ tồn tại trong test split; DF40_train_extracted không có. Không thể mở rộng.
- `df-40-test-full` là test split DF40, nhưng test_data_v3 chỉ sample một phần → frame/video còn lại sạch (loại trừ chính xác).
- `DF40_train_extracted` (train split) sạch hoàn toàn, 31 method × 21-31K frame.
- **faceswap/sadtalker/wav2lip/MRAA** có headroom khổng lồ (~22K/method) từ train split.
- styleclip +624, deepfacelab +1.142 — headroom vừa từ test split.

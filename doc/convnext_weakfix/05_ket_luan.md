# 05 — Kết luận, hạn chế, hướng phát triển

## 1. Kết luận

- **v3 là bản final tốt nhất**: test acc **99.70%** (baseline ConvNeXt 99.41%), FN **14→6**,
  FP **0→1**, AUC 99.99%. Không đụng checkpoint gốc — chỉ tạo checkpoint mới.
- **Replicate thành công phương pháp v5_weakfix** với cùng data / kỹ thuật / hyperparameter:
  pipeline baseline → v2 (method-balanced) → v3 (faceswap-focused) áp lên **ConvNeXt-Tiny**
  cho kết quả **tương tự thậm chí cao hơn** v5 (ViT-S/16): v3 97.88% → **99.70%**.
- **ConvNeXt baseline mạnh hơn hẳn v5**: 99.41% vs 95.37% (+4.04pp). starganv2 (72.5%→100%)
  và whichfaceisreal (73.33%→100%) — 2 method v5 baseline sụp nặng — ConvNeXt **bắt được 100%
  ngay từ baseline**, không cần vá.
- **faceswap được xử lý triệt để**: baseline 88.89% → v2 85.19% (tụt nhẹ) → **v3 100% (27/27)**.
  ConvNeXt v3 bắt đủ **10/10** case khó của v5, kể cả `ffc:701` (FF++ siêu sắc) mà v5 v3 bất lực.
- **Ràng buộc identity-disjoint giữ nghiêm**: dùng đúng CSV v2/v3 đã build sẵn (verify
  `identity_overlap_added = 0`), không build lại, không rò rỉ thêm.

## 2. Điều gì tạo nên kết quả

1. **Phương pháp v5_weakfix đúng và tái lập được**: sampler method-balanced (v2) rồi
   faceswap-focused (v3) + thêm 8,000 frame faceswap identity-disjoint — 2 đòn bẩy tăng cả
   data lẫn tần suất nhìn faceswap (~10× so v2) đã chứng minh hiệu quả trên 2 kiến trúc khác nhau.
2. **ConvNeXt-Tiny có sẵn đặc trưng tốt hơn cho bài này**: baseline đã 99.41% → quy trình vá
   chỉ cần "thêm chút" thay vì "sửa nhiều" như v5.
3. **Init từ checkpoint trước + LR nhẹ dần** (2e-5/5e-4 → 1.5e-5/4e-4) giữ thành quả các bước trước,
   tránh forgetting 40 method.

## 3. Hạn chế còn lại (trung thực)

| Hạn chế | Chi tiết |
|---|---|
| 6 FN còn lại của v3 | sadtalker 2, fsgan 1, facedancer 1, lia 1, mcnet 1 — ảnh đơn lẻ sát ngưỡng. |
| 1 FP ở v3 | 1 ảnh real bị gán fake (baseline/v2 đều FP=0). |
| heygen (n=1) | Baseline 0% nhưng v2/v3 bắt được — mẫu duy nhất, không đánh giá được ý nghĩa. |
| SadTalker v3 hơi tụt so v2 | 92.31% vs 96.15% (mất 1 ảnh) nhưng vẫn cao hơn baseline 88.46%. |
| Chỉ đo trên 1 benchmark | test zero-leakage 2,354 ảnh; chưa đánh giá chéo trên DF40 test-full / Celeb-DF / VoxCeleb. |

## 4. Hướng phát triển (nếu muốn đi tiếp)

1. **V4 để gỡ nốt 6 FN + 1 FP**: init từ v3, thêm chút trọng số cho sadtalker/fsgan/facedancer/
   lia/mcnet + real → lợi ích nhỏ dần (diminishing returns), ước lượng +0.2–0.5 điểm.
2. **Đánh giá chéo đa benchmark**: linear-probe / full-finetune trên DF40 test-full, Celeb-DF,
   VoxCeleb để đo mức tổng quát thật sự ngoài bộ test 2,354 ảnh.
3. **Threshold/calibration**: ngưỡng 0.5 cố định; nếu ưu tiên bắt fake, hạ ngưỡng sẽ đổi FP↔FN
   (test CSV đã có sẵn cột prob ở `eval_*.npz`).
4. **So sánh chính thức ViT-S/16 vs ConvNeXt-Tiny** trên cùng pipeline: kết quả hiện tại cho thấy
   ConvNeXt thắng 1.8–4pp ở mọi stage — đáng ghi thành báo cáo benchmark riêng.

## 5. Tái lập (reproduce)

```bash
REPO=/workspace/quangmanh/deepfake
PY=/workspace/hoangtuan/deepfake-ViT/.venv/bin/python   # venv có pandas

# Data: dùng CSV đã build sẵn (copy tại data/splits/, nguồn /workspace/hoangtuan/deepfake-ViT/data/splits)

# 1) Baseline (v5-equivalent) — 3 epoch trên 54K v5 CSV
$PY $REPO/scripts/finetune_convnext_weakfix.py --stage baseline

# 2) v2 — method-balanced, init từ baseline, 2 epoch
$PY $REPO/scripts/finetune_convnext_weakfix.py --stage v2 --init-ckpt outputs/finetune/convnext_baseline.pt

# 3) v3 — faceswap-focused, init từ v2, 3 epoch
$PY $REPO/scripts/finetune_convnext_weakfix.py --stage v3 --init-ckpt outputs/finetune/convnext_weakfix_v2.pt

# 4) Eval thống nhất cả 3 stage (bf16 protocol, giống v5_weakfix)
$PY $REPO/scripts/eval_convnext_weakfix.py
```

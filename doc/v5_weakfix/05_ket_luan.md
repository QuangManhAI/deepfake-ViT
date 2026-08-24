# 05 — Kết luận, hạn chế, hướng phát triển

## 1. Kết luận

- **v3 là bản final tốt nhất**: test acc **97.88%** (baseline 95.37%), FN **78→28**, FP **31→22**,
  AUC 99.70%. Không đụng checkpoint gốc v5 / v2 — chỉ tạo checkpoint mới.
- **Toàn bộ chỗ sụp baseline đã ≥ 90%** trừ 2 case:
  - **sadtalker 80.77%** — chỉ 1 frame biên (sát ngưỡng 0.5), về lại mức baseline.
  - **faceswap 88.89%** — đã cải thiện lớn (+26 điểm) nhưng còn 3 ca khó.
- **Hai dataset Mạnh thêm đều được dùng đúng mục đích:**
  - `deep-fake-face-swap` → 8,076 fake face-swap (method `deepfake_faceswap`).
  - `celebvhq` → 4,000 real (method `celebvhq_real`), giúp giữ FP thấp.
- **Ràng buộc identity-disjoint được giữ nghiêm** — mọi frame bổ sung đều loại nhân vật trùng
  test, verify `identity_overlap_added = 0` (xem `v5_weakfix_v3_dataset_summary.json`).

## 2. Điều gì tạo nên thành công của v3

1. **Chẩn đoán đúng** thay vì thêm data mù: 10 miss faceswap đều là frame **sắc nét chất lượng
   cao** — vấn đề không phải thiếu data mà là **model chưa được học lớp này đủ**.
2. **Tăng cả 2 đòn bẩy cùng lúc:**
   - Thêm +8,000 frame faceswap **identity-disjoint** (data đúng phân phối, không rò rỉ).
   - Sampler **faceswap-focused** (P=35%) → faceswap được nhìn gấp ~10 lần v2.
3. **Khởi tạo từ v2 + LR nhẹ hơn** → giữ lại thành quả của 40 method khác, chỉ chuyên tâm vá faceswap.

## 3. Hạn chế còn lại (trung thực)

| Hạn chế | Chi tiết |
|---|---|
| faceswap còn 88.89% | 3 ca khó: 1 FF++ siêu sắc (`ffc:701`), 1 VoxCeleb OOD, 1 Celeb-DF sát ngưỡng. |
| sadtalker về baseline | 1 frame biên `id2_da1vvigy5tQ` (prob 0.509→0.426). |
| regression nhỏ | starganv2 100→95 (mất 2/40). |
| face-swap mới lạ ngoài 6 method quen | DFS chỉ được học 68.6% (prob trung bình 0.61) — model còn mơ hồ với face-swap chất lượng cao chưa từng thấy. |

## 4. Hướng phát triển (nếu muốn đi tiếp)

1. **V4 để gỡ nốt:** init từ v3, sampler giữ faceswap cao + thêm chút trọng số cho sadtalker/
   starganv2 → ước lượng chỉ lấy thêm 1–3 điểm, lợi ích nhỏ dần (diminishing returns).
2. **Augmentation mạnh cho face-swap:** blur/JPEG-compress mạnh hơn để mô phỏng frame thật
   (test miss là frame sắc, còn hit là frame mờ — thêm 2 luồng sẽ cân bằng).
3. **Thêm face-swap cross-dataset:** tận dụng nốt phần test-full + nhiều frame hơn từ
   `DF40_train_extracted/faceswap` (còn ~18K sạch chưa dùng).
4. **Threshold/calibration:** hiện ngưỡng 0.5 cố định; nếu ứng dụng ưu tiên bắt fake hơn,
   hạ ngưỡng sẽ đổi FP↔FN (xem `test_balanced_fixed_zero_leakage` có cột prob để tinh chỉnh).
5. **Linear-probe / đánh giá chéo** trên benchmark khác (DF40 test-full, Celeb-DF, VoxCeleb) để
   đo mức tổng quát thật sự ngoài bộ test 2,354 ảnh này.

## 5. Tái lập (reproduce)

```bash
REPO=/workspace/hoangtuan/deepfake-ViT
PY=$REPO/.venv/bin/python

# 1) Build dataset v3 (replay + boost + 2 dataset mới + thêm faceswap)
$PY $REPO/scripts/build_finetune_v5_weakfix.py   # -> train_v5_weakfix.csv (v2)
$PY $REPO/scripts/expand_faceswap_v3.py          # -> train_v5_weakfix_v3.csv (v3), verify 0 overlap

# 2) Finetune v3 từ v2 ckpt
$PY $REPO/scripts/finetune_v5_weakfix_v3.py \
    --init-ckpt $REPO/experiments/checkpoints/exp05_v5_weakfix/best_model.pt

# 3) Regenerate report (script finetune crash KeyError ở cuối — dùng eval script)
$PY $REPO/scripts/eval_v5_weakfix_v3_report.py
```

> Lưu ý: trước đó phải có sẵn frame celebvhq (`prepare_celebvhq_frames.py`) và 2 dataset mới
> tại các path trong [02_data.md](02_data.md).

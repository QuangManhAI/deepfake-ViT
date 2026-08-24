# 01 — Mục tiêu & model gốc

## 1. Mục tiêu

Colleague **hoangtuan** đã finetune model deepfake và đánh giá lúc 19h 23/08/2026 tại
`/workspace/hoangtuan/deepfake-ViT`. Yêu cầu:

1. Phân tích **model yếu ở method nào** (method nào "sụp").
2. Dùng 2 dataset mới bổ sung để **cải thiện chỗ sụp**:
   - `/workspace/data/celebvhq` — video real (người nổi tiếng).
   - `/workspace/data/deep-fake-face-swap` — ảnh face-swap fake.
3. **Ràng buộc quan trọng:** giữ **identity-disjoint** — train/test không trùng nhân vật
   (đây chính là lý do v5 train chỉ dùng ~600 frame/method).

## 2. Model gốc (v5)

| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | DINOv3 **ViT-Small/16** + head phân loại |
| Embedding dim | 384 (12 layer transformer) |
| Head | `nn.Linear(384, 2)` (2 lớp: real / fake) |
| Pretrained | `/workspace/hoangtuan/deepfake-ViT/models/dinov3_small/model.safetensors` |
| Checkpoint v5 | `/workspace/hoangtuan/deepfake-ViT/experiments/checkpoints/exp05_v5_combined/best_model.pt` |
| Builder | `src/models/dinov3_vit.py` → `build_dinov3_classifier(weights_path, num_classes=2, img_size=256)` |
| Input | 256×256 RGB, normalize ImageNet |

## 3. Test benchmark (chuẩn đánh giá)

- CSV: `/workspace/data/zero_leakage_benchmark_fixed/test_balanced_fixed_zero_leakage.csv`
- **2,354 ảnh = 1,177 real / 1,177 fake** (cân bằng tuyệt đối), **40 method fake** + real.
- Certified **0% MD5 leak** ở mức frame (không có frame nào lặp giữa train và test).
- Cột `identity` ghi nhận nhân vật theo nhiều định dạng:
  - `ffc:N` → FF++
  - `oth:...:idA_idB_...` → VoxCeleb
  - `cdc:...` → Celeb-DF
  - `starganv2_clean_N`, `e4s:...`, `fe:...`, `efs:...` → các nguồn khác

### Kết quả baseline (v5)

| Chỉ số | Giá trị |
|---|---|
| Test Accuracy | **95.37%** |
| ROC-AUC | 99.35% |
| Precision | 97.26% |
| Recall (fake) | 93.37% |
| F1 | 95.28% |
| Confusion matrix | `[[1146, 31], [78, 1099]]` |
| Real acc (FP) | 97.37% (31 ảnh real bị gán fake) |
| Fake recall (FN) | 93.37% (78 ảnh fake bị gán real) |

## 4. Phân tích method yếu (baseline)

Method sụp = **fake acc < 90%**, sắp theo độ sụp:

| Method | Loại | N | Acc | Mean fake-prob | Nhóm |
|---|---|---|---|---|---|
| heygen | talking-head | 1 | 0.0% | 0.226 | (n=1, không có ý nghĩa) |
| **faceswap** | face-swap | 27 | **62.96%** | 0.605 | FF++ faceswap |
| **starganv2** | GAN attribute | 40 | **72.5%** | 0.661 | GAN edit |
| **whichfaceisreal** | GAN | 30 | **73.33%** | 0.740 | GAN synthetic |
| **facedancer** | reenactment | 27 | **74.07%** | 0.694 | talking/reenact |
| **sadtalker** | talking-head | 26 | **80.77%** | 0.754 | audio→video |
| **fsgan** | face-swap | 26 | **84.62%** | 0.812 | face-swap |
| **wav2lip** | lip-sync | 22 | **86.36%** | 0.772 | audio→video |
| **e4s** | GAN edit | 15 | **86.67%** | 0.812 | one-shot edit |
| **lia** | face-swap | 27 | **88.89%** | 0.807 | face-swap |
| **simswap** | face-swap | 27 | **88.89%** | 0.833 | face-swap |
| **blendface** | face-swap | 27 | **88.89%** | 0.857 | face-swap |
| CollabDiff | diffusion edit | 31 | 90.32% | 0.836 | (biên) |

> Các method còn lại (sd2.1, StyleGAN2/3/XL, VQGAN, stargan, fomm, …) đều 95–100%.

**Pattern rõ rệt:** chỗ sụp toàn bộ là **face-swap / reenactment / talking-head / GAN-edit** —
loại ảnh fake trông vẫn "giống mặt người thật" nên model nhận nhầm thành real. Diffusion / GAN
thuần túy (ảnh không phải người) thì model bắt rất tốt.

## 5. Nguyên nhân sâu xa

1. **Train quá ít frame cho method yếu:** v5 train chỉ dùng **~600 frame/method** từ
   `DF40_train_extracted` (pool thật có **22K–62K frame/method**). Không đủ dữ liệu để học
   đặc trưng riêng của từng method. Đây là hệ quả trực tiếp của việc ép identity-disjoint.
2. **Thiếu dữ liệu face-swap mới lạ** ngoài 6 method face-swap quen thuộc trong v5 train.
3. **Real FP (31 ảnh):** chủ yếu FFHQ + FF++/ffc — model hơi nhạy với ảnh studio sạch.

→ Chi tiết xây dựng data để vá: [02_data.md](02_data.md)

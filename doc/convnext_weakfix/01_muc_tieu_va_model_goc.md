# 01 — Mục tiêu & model gốc

## 1. Mục tiêu

Replicate trọn vẹn quy trình **vá method yếu** của `doc/v5_weakfix` (đã làm cho model v5
DINOv3 ViT-S/16 của hoangtuan, đạt 97.88%) lên model **`dinov3_next_cnn`** của repo này:

1. Dùng **đúng data train và test** được nhắc tới trong v5_weakfix (các CSV đã build sẵn,
   identity-disjoint, cùng benchmark zero-leakage).
2. Finetune model với **cùng phương pháp — cùng kỹ thuật — cùng hyperparameter**.
3. Kỳ vọng kết quả **tương tự** (test acc tầm 97–98%).

## 2. Model gốc — `dinov3_next_cnn`

| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | DINOv3 **ConvNeXt-Tiny** (reconstruct theo tên tensor safetensors) |
| Stem | Conv2d(3→96, 4×4, stride 4) + LayerNorm |
| Stages | `depths=[3,3,9,3]`, `dims=[96,192,384,768]` |
| Block | depthwise 7×7 → LN → pw1(×4) → GELU → pw2 → LayerScale |
| Embedding dim | 768 (GAP + final LayerNorm) |
| Số tham số | ~27.8M (model `DinoConvNextClassifier` 28.1M gồm head) |
| Pretrained | `/workspace/quangmanh/deepfake/models/dinov3_next_cnn/model-2.safetensors` |
| Classifier | `DinoConvNextClassifier` — head MLP: LN → Dropout → Linear(768→384) → GELU → Dropout → Linear(384→2) |
| Input | 256×256 RGB, normalize ImageNet (BICUBIC resize) |
| Builder | `src/models/dinov3_convnext.py` → `load_dinov3_convnext`; `src/models/classifier_v2.py` → `DinoConvNextClassifier` |

> Lưu ý: khác với v5 (vốn đã có sẵn checkpoint `exp05_v5_combined`), `dinov3_next_cnn` chỉ có
> **backbone pretrained**, chưa có classifier nào được train. Vì vậy bước đầu tiên là train
> **baseline** trên đúng CSV 54,000 ảnh mà v5 đã dùng (cùng recipe 3 epoch) để tạo ra
> "v5-equivalent" cho ConvNeXt, rồi mới áp dụng 2 bước finetune vá method yếu (v2, v3).

## 3. Test benchmark (chuẩn đánh giá — dùng chung với v5_weakfix)

- CSV: `/workspace/data/zero_leakage_benchmark_fixed/test_balanced_fixed_zero_leakage.csv`
- **2,354 ảnh = 1,177 real / 1,177 fake** (cân bằng tuyệt đối), **40 method fake** + real.
- Certified **0% MD5 leak** ở mức frame. Cột `identity` ghi nhân vật theo định dạng
  `ffc:N` (FF++), `oth:*:idA_idB*` (VoxCeleb), `cdc:idN` (Celeb-DF), …

### Method yếu mà v5_weakfix nhắm tới (mốc tham chiếu)

| Method | Loại | N | Acc v5 baseline | Acc v5 v3 |
|---|---|---|---|---|
| **faceswap** | face-swap | 27 | 62.96% | **88.89%** |
| **starganv2** | GAN edit | 40 | 72.5% | 95.0% |
| **whichfaceisreal** | GAN | 30 | 73.33% | 93.33% |
| **facedancer** | reenactment | 27 | 74.07% | 92.59% |
| **sadtalker** | talking-head | 26 | 80.77% | 80.77% |
| **fsgan** | face-swap | 26 | 84.62% | 92.31% |
| **wav2lip** | lip-sync | 22 | 86.36% | 90.91% |
| **e4s** | GAN edit | 15 | 86.67% | 100% |
| **simswap** | face-swap | 27 | 88.89% | 100% |
| **blendface** | face-swap | 27 | 88.89% | 92.59% |
| **lia** | face-swap | 27 | 88.89% | 92.59% |

→ Toàn bộ chỗ sụp của v5 nằm ở **face-swap / reenactment / talking-head / GAN-edit** — loại ảnh
fake trông vẫn giống mặt người thật. Quy trình vá (data + sampler + hyperparam) được replicate
nguyên vẹn lên ConvNeXt để kiểm chứng xem cải thiện có "tương tự" không.

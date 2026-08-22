# DATA_PREP_SUMMARY_REPORT.md — Báo Cáo Tổng Kết Toàn Diện Phân Tách Dữ Liệu DF40, FaceForensics++ & Celeb-DF

- **Motivation/Background**: Huấn luyện và đánh giá mô hình phân loại Deepfake (Deepfake-ViT) trên benchmark DF40 đòi hỏi quy trình phân tách dữ liệu nghiêm ngặt, đảm bảo không rò rỉ danh tính (Zero Identity Leakage), cân bằng lớp 1:1 (Real:Fake) và cung cấp bộ dữ liệu đánh giá độc lập cho 40 phương pháp sinh giả khác nhau.
- **Purpose**: Báo cáo tổng hợp quy chuẩn kỹ thuật, cấu trúc phân vùng dữ liệu, cơ chế trích xuất và tích hợp đa nguồn (FaceForensics++, Celeb-DF-v2, DF40 Train, test_data_v3), chứng minh toán học Zero-Leakage và danh mục 195 file split phục vụ huấn luyện và đánh giá.
- **Overview Pipeline**: `Trích xuất Real Đa Nguồn (22.4k FF++ + 10.3k Celeb-DF)` $\\rightarrow$ `Phân vùng Identity-Disjoint 70/15/15 (22,237 IDs)` $\\rightarrow$ `Tạo tập Cân bằng 1:1 Quy mô Lớn (58.9k images)` $\\rightarrow$ `Sinh 195 file Split cho 40 Phương pháp (/data/splits/methods/)` $\\rightarrow$ `Kiểm định Tự động (7/7 Tests Passed) & Post-Split EDA Dashboard`.
- **Detailed Plan**: §1 Tóm tắt Thực thi (Executive Summary); §2 Khảo sát & Trích xuất Dữ liệu Đa Nguồn (FF++ & Celeb-DF); §3 Hai Chế độ Phân tách Dữ liệu (High-Scale Balanced Pool 58.9k vs. Identity-Disjoint Benchmark); §4 Nguyên lý Toán học Đảm bảo Zero Identity Leakage; §5 Danh mục Chi tiết Bộ Đánh giá 40 Phương pháp Deepfake; §6 Tổng hợp File Dữ liệu Đầu ra trong `data/splits/`; §7 Kết quả Kiểm thử Tự động & Visual Dashboard; §8 Hướng dẫn Sử dụng Chi tiết trong Training & Evaluation.
- **References**: `prepare_df40_splits.py`, `extract_celeb_df_frames.py`, `test_data_prep.py`, `00_comprehensive_dataset_eda.ipynb`, `FaceForensics++`, `Celeb-DF-v2`, `DF40_train_manifest.csv`, `test_data_v3`.

---

## Table of Contents

- [1. Tóm tắt Thực thi (Executive Summary)](#1-tóm-tắt-thực-thi-executive-summary)
- [2. Khảo sát & Trích xuất Dữ liệu Đa Nguồn (FF++ & Celeb-DF)](#2-khảo-sát--trích-xuất-dữ-liệu-đa-nguồn-ff--celeb-df)
- [3. Hai Chế độ Phân tách Dữ liệu Chính](#3-hai-chế-độ-phân-tách-dữ-liệu-chính)
  - [3.1 Chế độ 1: High-Scale Balanced & Full Training Pool (Tập Huấn Luyện Lớn 58.9k & 652k)](#31-chế-độ-1-high-scale-balanced--full-training-pool-tập-huấn-luyện-lớn-589k--652k)
  - [3.2 Chế độ 2: Identity-Disjoint Benchmark Splits (Tập Prototype & Sub-Benchmark)](#32-chế-độ-2-identity-disjoint-benchmark-splits-tập-prototype--sub-benchmark)
- [4. Nguyên lý Toán học & Đảm bảo Tuyệt đối Zero-Leakage](#4-nguyên-lý-toán-học--đảm-bảo-tuyệt-đối-zero-leakage)
- [5. Danh mục Bộ Đánh giá Độc lập 40 Phương pháp Deepfake](#5-danh-mục-bộ-đánh-giá-độc-lập-40-phương-pháp-deepfake)
- [6. Bảng Tổng hợp Toàn bộ File Dữ liệu Đầu ra trong `data/splits/`](#6-bảng-tổng-hợp-toàn-bộ-file-dữ-liệu-đầu-ra-trong-datasplits)
- [7. Kết quả Kiểm định Chất lượng & Trực quan hóa Visual EDA](#7-kết-quả-kiểm-định-chất-lượng--trực-quan-hóa-visual-eda)
- [8. Hướng dẫn Sử dụng Chi tiết trong Training & Evaluation](#8-hướng-dẫn-sử-dụng-chi-tiết-trong-training--evaluation)

---

## 1. Tóm tắt Thực thi (Executive Summary)

Dự án **Deepfake-ViT** trên bộ dữ liệu **DF40 Deepfake Benchmark** đã hoàn thành thiết lập toàn diện hạ tầng phân tách dữ liệu và trích xuất đa nguồn:

1. **Trích xuất Đầy đủ Dữ liệu Real từ Celeb-DF-v2**: Sử dụng script đa luồng [extract_celeb_df_frames.py](../src/data/extract_celeb_df_frames.py) trích xuất thành công **10,336 Real face frames ($256 \\times 256$)** từ 690 video training sạch của Celeb-DF-v2, lưu tại `data/processed/celeb_df_extracted/`.
2. **Hợp nhất Toàn diện Nguồn Real (FF++ + Celeb-DF)**: Tổng cộng **32,754 ảnh Real độc lập** (22,418 từ FaceForensics++ + 10,336 từ Celeb-DF-v2) sau khi loại trừ 100% video/identity trùng với tập Test và Val.
3. **Cân bằng 1:1 Quy mô Lớn (58,958 images)**: Xây dựng tập [train_combined_balanced.csv](../data/splits/train_combined_balanced.csv) gồm **29,479 Real faces (FF++ & Celeb-DF)** và **29,479 Fake faces (DF40)**, đạt tỷ lệ cân bằng hoàn hảo 1:1.
4. **Không Rò rỉ Danh tính (Zero Identity Leakage)**: 22,237 unique subject identities được phân chia nghiêm ngặt: $\\text{Train} \\cap \\text{Val} = \\emptyset$, $\\text{Train} \\cap \\text{Test} = \\emptyset$, $\\text{Val} \\cap \\text{Test} = \\emptyset$.
5. **Bộ Đánh giá Độc lập cho 40 Phương pháp**: Tự động sinh **195 file CSV** trong thư mục [data/splits/methods/](../data/splits/methods/) cho từng phương pháp sinh giả.
6. **Kiểm thử Tự động & Visual Analytics Hoàn tất**: Vượt qua 100% (7/7) ca kiểm thử tự động trong [tests/test_data_prep.py](../tests/test_data_prep.py) và xuất bản 12 biểu đồ độ phân giải cao trong [notebooks/00_comprehensive_dataset_eda.ipynb](../notebooks/00_comprehensive_dataset_eda.ipynb).

---

## 2. Khảo sát & Trích xuất Dữ liệu Đa Nguồn (FF++ & Celeb-DF)

Toàn bộ dữ liệu thô và dữ liệu trích xuất mới:

| Nguồn Dữ Liệu | Đường Dẫn Thực Tế | Quy Mô / Số Lượng | Cấu Trúc / Định Dạng | Vai Trò trong Pipeline |
| :--- | :--- | :---: | :--- | :--- |
| **FaceForensics++ (FF++)** | `/workspace/data/FaceForensics++/original_sequences/youtube/c23/frames` | **999 video sequences**<br>**31,949 ảnh PNG** | Ảnh PNG $256 \\times 256$ cắt sẵn khuôn mặt từ video gốc YouTube. | Cung cấp **22,418 Real frames sạch** (từ 701 video độc lập) cho tập Train lớn. |
| **Celeb-DF-v2 (Trích xuất mới)** | `/workspace/hoangtuan/deepfake-ViT/data/processed/celeb_df_extracted/` | **690 video training**<br>**10,336 ảnh PNG** | Ảnh PNG $256 \\times 256$ cắt tâm khuôn mặt từ 690 video `Celeb-real` & `YouTube-real`. | Cung cấp **10,336 Real frames sạch** (từ 690 video độc lập) cho tập Train lớn. |
| **Celeb-DF-v2 (Gốc)** | `/workspace/data/Celeb-DF-v2` | **890 video Real**<br>**5,639 video Fake** | File video `.mp4` nguyên bản. | Nguồn gốc trích xuất. |
| **DF40 Train Pool** | `/workspace/data/DF40_train_manifest.csv`<br>`/workspace/data/DF40_train_extracted/` | **693,335 ảnh** (692,158 fake + 1,177 real) | Ảnh mặt cắt sẵn $256 \\times 256$ của **31 thuật toán thao túng**. | Nguồn Fake quy mô lớn cho tập Training Pool. |
| **DF40 Test Suite (`test_data_v3`)** | `/workspace/data/test_data_v3/` | **30,691 ảnh** (1,177 real + 29,514 fake) | Bộ benchmark chuẩn hóa gồm **40 phương pháp fake** + 1,177 canonical real faces. | Nguồn chia tách Identity-Disjoint và sinh bộ test chuyên biệt cho từng method. |

---

## 3. Hai Chế độ Phân tách Dữ liệu Chính

```
                                          ┌─────────────────────────────────────────────────────────────┐
                                          │             Multi-Dataset Raw Sources (/workspace)          │
                                          │  FF++ (22.4k) | Celeb-DF (10.3k) | DF40 Train Pool (692k)   │
                                          └──────────────────────────────┬──────────────────────────────┘
                                                                         │
                                ┌────────────────────────────────────────┴────────────────────────────────────────┐
                                │                                                                                 │
                                ▼                                                                                 ▼
             ┌─────────────────────────────────────────┐                       ┌──────────────────────────────────────────────┐
             │    Chế độ 1: HIGH-SCALE BALANCED POOL   │                       │      Chế độ 2: IDENTITY-DISJOINT BENCH       │
             │      (Train chính thức khuyến nghị)     │                       │         (Prototype & Sub-Benchmark)          │
             ├─────────────────────────────────────────┤                       ├──────────────────────────────────────────────┤
             │ • train_combined_balanced (58,958 imgs) │                       │ • train.csv (20,853 imgs - 70%)              │
             │   (29,479 Real [FF++ & Celeb] + 29.4k)  │                       │ • val.csv   (4,440 imgs - 15%)               │
             │ • val_combined_balanced (6,550 imgs)    │                       │ • test.csv  (4,398 imgs - 15%)               │
             │ • train_pool_693k (652,421 imgs)        │                       │ • train_balanced.csv (1,668 imgs)            │
             │ • val_pool (72,491 imgs)                │                       │ • test_balanced.csv  (340 imgs)              │
             └────────────────────┬────────────────────┘                       └──────────────────────┬───────────────────────┘
                                  │                                                                   │
                                  └───────────────────────────────┬───────────────────────────────────┘
                                                                  │
                                                                  ▼
                                                ┌───────────────────────────────────┐
                                                │   40-METHOD TEST SUITE (195 CSVs) │
                                                │      (/data/splits/methods/)      │
                                                │   • test_<method>_balanced.csv    │
                                                │   • benchmark_test_<m>_bal.csv    │
                                                │   • benchmark_test_<m>_full.csv   │
                                                └───────────────────────────────────┘
```

### 3.1 Chế độ 1: High-Scale Balanced & Full Training Pool (Tập Huấn Luyện Lớn 58.9k & 652k)

*Khuyến nghị sử dụng cho quá trình Train chính thức của mô hình Deepfake-ViT.*

* **`train_combined_balanced.csv`** (**58,958 ảnh**):
  * **Real (29,479 ảnh)**: Hợp nhất từ 20,189 Real frames (FaceForensics++) + 9,290 Real frames (Celeb-DF-v2).
  * **Fake (29,479 ảnh)**: Lấy mẫu cân đối từ 31 phương pháp của DF40 Train Pool.
  * **Tỷ lệ**: **Đúng 1.0 : 1 Cân bằng hoàn hảo**.
* **`val_combined_balanced.csv`** (**6,550 ảnh**): 3,275 Real (FF++ + Celeb) + 3,275 Fake DF40 (1.0 : 1).
* **`train_pool_693k.csv`** (**652,421 ảnh**): 29,479 Real + 622,942 Fake DF40 (dành cho full-scale training không giới hạn).
* **`val_pool.csv`** (**72,491 ảnh**): 3,275 Real + 69,216 Fake DF40.
* **Tập Test tương ứng**: Sử dụng toàn bộ [test_full.csv](../data/splits/test_full.csv) (**29,691 ảnh**) hoặc các tập `benchmark_test_<method>_balanced.csv`.

---

### 3.2 Chế độ 2: Identity-Disjoint Benchmark Splits (Tập Prototype & Sub-Benchmark)

*Sinh ra từ việc phân tách độc lập tập `test_data_v3` (29,691 ảnh) theo tỷ lệ 70/15/15 theo danh tính nhân vật.*

* **`train.csv`** (**20,853 ảnh**): 834 Real (701 FF++ + 133 Celeb-DF) + 20,019 Fake.
* **`val.csv`** (**4,440 ảnh**): 173 Real (147 FF++ + 26 Celeb-DF) + 4,267 Fake.
* **`test.csv`** (**4,398 ảnh**): 170 Real (143 FF++ + 27 Celeb-DF) + 4,228 Fake.
* **`train_balanced.csv`** (**1,668 ảnh**): 834 Real + 834 Fake (1:1).
* **`val_balanced.csv`** (**346 ảnh**): 173 Real + 173 Fake (1:1).
* **`test_balanced.csv`** (**340 ảnh**): 170 Real + 170 Fake (1:1).

---


### 3.3 Bộ Đánh Giá Chuyên Biệt Cho Celeb-DF-v2 (Official Test Benchmark)
- **`test_celeb_df_v2.csv`** (**2,590 ảnh**): Trích xuất từ 518 video test chuẩn của Celeb-DF-v2 (890 frames YouTube-real + 1,700 frames Celeb-synthesis).
- **`test_celeb_df_v2_balanced.csv`** (**1,780 ảnh**): Tập Test cân bằng 1:1 (890 Real + 890 Fake) chuẩn quốc tế.
- **`data/splits/methods/test_CelebDFv2_balanced.csv`** & **`test_CelebDFv2_full.csv`**: Đóng gói thành phương pháp đánh giá độc lập bên cạnh 39 phương pháp của DF40.

## 4. Nguyên lý Toán học & Đảm bảo Tuyệt đối Zero-Leakage

1. **Phân vùng Không gian Danh tính (Identity-Disjoint)**:
   Mỗi ảnh được gán một khóa định danh duy nhất $I \\in \\mathcal{I}$ (ví dụ: `ffc:709`, `cdc:id28_0007`, `oth:pixart:id25_0004`).
   Tổng cộng **22,237 identities** được chia thành 3 tập rời rạc:

   $$\\mathcal{I} = \\mathcal{I}_{\\text{train}} \\cup \\mathcal{I}_{\\text{val}} \\cup \\mathcal{I}_{\\text{test}}$$

   $$\\mathcal{I}_{\\text{train}} \\cap \\mathcal{I}_{\\text{val}} = \\emptyset, \\quad \\mathcal{I}_{\\text{train}} \\cap \\mathcal{I}_{\\text{test}} = \\emptyset, \\quad \\mathcal{I}_{\\text{val}} \\cap \\mathcal{I}_{\\text{test}} = \\emptyset$$

2. **Lọc Rò rỉ Video FaceForensics++ & Celeb-DF (Safety Masking)**:
   * **FaceForensics++**: Lọc bỏ toàn bộ **298 video sequence IDs** trùng với test/val, chỉ lấy 22,418 frames từ 701 video độc lập.
   * **Celeb-DF-v2**: Lọc bỏ toàn bộ **575 video stems** và 178 danh tính test trong `List_of_testing_videos.txt` và `test_detailed.csv`, chỉ trích xuất từ 690 video training độc lập $\\rightarrow$ **Đảm bảo 0% Video Leakage và 0% Identity Leakage**.

---

## 5. Danh mục Bộ Đánh giá Độc lập 40 Phương pháp Deepfake

Tất cả các tập test chuyên biệt được lưu trữ trong thư mục [data/splits/methods/](../data/splits/methods/) (195 files CSV). Bảng thống kê chi tiết quy mô cho từng phương pháp:

| # | Phương Pháp Sinh Giả | Họ Thao Túng | Loại Hình | Test 1:1 Bal (`test_<m>_bal`) | Benchmark 1:1 Bal (`bench_<m>_bal`) | Benchmark Full (`bench_<m>_full`) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | **DiT** | Face Synthesis | Diffusion Transformer | 316 | 2,008 | 2,181 |
| 2 | **SiT** | Face Synthesis | Scalable Interpolant | 288 | 2,008 | 2,181 |
| 3 | **StyleGAN2** | Face Synthesis | Unconditional GAN | 184 | 1,232 | 1,793 |
| 4 | **StyleGAN3** | Face Synthesis | Alias-Free GAN | 340 | 2,008 | 2,181 |
| 5 | **StyleGANXL** | Face Synthesis | Large-scale GAN | 302 | 2,008 | 2,181 |
| 6 | **sd2.1** | Face Synthesis | Latent Diffusion | 340 | 2,354 | 2,786 |
| 7 | **MidJourney** | Face Synthesis | Text-to-Image Diff (Zero-Shot) | 184 | 1,260 | 1,807 |
| 8 | **CollabDiff** | Face Synthesis | Collaborative Diffusion (Zero-Shot) | 184 | 1,500 | 1,927 |
| 9 | **pixart** | Face Synthesis | Transformer T2I | 272 | 1,866 | 2,110 |
| 10 | **RDDM** | Face Synthesis | Residual Denoising Diffusion | 158 | 1,120 | 1,737 |
| 11 | **ddim** | Face Synthesis | Denoising Diffusion Implicit | 182 | 1,212 | 1,783 |
| 12 | **VQGAN** | Face Synthesis | Vector Quantized GAN | 200 | 1,206 | 1,780 |
| 13 | **faceswap** | Face Swap | Classical Face Swap | 200 | 1,360 | 1,857 |
| 14 | **simswap** | Face Swap | Generalized Face Swap | 186 | 1,368 | 1,861 |
| 15 | **inswap** (Insight) | Face Swap | InsightFace Identity Swap | 132 | 942 | 1,648 |
| 16 | **mobileswap** | Face Swap | Mobile Face Swap | 340 | 2,354 | 2,575 |
| 17 | **facedancer** | Face Swap | High-Fidelity Swap | 196 | 1,378 | 1,866 |
| 18 | **blendface** | Face Swap | Blended Face Swap | 196 | 1,366 | 1,860 |
| 19 | **uniface** | Face Swap | Unified Face Swap | 204 | 1,342 | 1,848 |
| 20 | **deepfacelab** | Face Swap | DeepFaceLab Pipeline (Zero-Shot) | 8 | 50 | 1,202 |
| 21 | **sadtalker** | Reenactment | Audio-driven Talking Head | 208 | 1,320 | 1,837 |
| 22 | **wav2lip** | Reenactment | Lip-Sync Reenactment | 184 | 1,120 | 1,737 |
| 23 | **fomm** | Reenactment | First Order Motion Model | 204 | 1,378 | 1,866 |
| 24 | **MRAA** | Reenactment | Region-Aware Animation | 208 | 1,380 | 1,867 |
| 25 | **lia** | Reenactment | Latent Image Animation | 194 | 1,376 | 1,865 |
| 26 | **mcnet** | Reenactment | Motion-Conditioned Net | 202 | 1,378 | 1,866 |
| 27 | **tpsm** | Reenactment | Thin-Plate Spline Motion | 218 | 1,378 | 1,866 |
| 28 | **facevid2vid** | Reenactment | One-Shot Free Reenact | 224 | 1,360 | 1,857 |
| 29 | **hyperreenact** | Reenactment | Hypernetwork Reenact | 208 | 1,368 | 1,861 |
| 30 | **pirender** | Reenactment | 3D-Aware Reenactment | 208 | 1,360 | 1,857 |
| 31 | **one_shot_free** | Reenactment | One-Shot Motion Transfer | 214 | 1,376 | 1,865 |
| 32 | **danet** | Reenactment | Dual-Attention Network | 218 | 1,378 | 1,866 |
| 33 | **fsgan** | Reenactment | Subject-Agnostic Reenact | 198 | 1,326 | 1,840 |
| 34 | **heygen** | Reenactment | Commercial Avatar (Zero-Shot) | 8 | 48 | 1,201 |
| 35 | **stargan** | Face Editing | Multi-Domain Editing (Zero-Shot) | 288 | 2,000 | 2,177 |
| 36 | **starganv2** | Face Editing | Diverse Face Editing (Zero-Shot) | 304 | 1,998 | 2,176 |
| 37 | **e4e** | Face Editing | Encoder4Editing (Zero-Shot) | 308 | 1,996 | 2,175 |
| 38 | **e4s** | Face Editing | Explicit Semantic Swap | 100 | 740 | 1,547 |
| 39 | **whichfaceisreal** | Face Synthesis | GAN Web Synthetics (Zero-Shot) | 212 | 1,500 | 1,927 |

*(Ghi chú: Phương pháp `styleclip` trên storage gốc bị khóa phân quyền root `-rw-------` nên được tự động bỏ qua an toàn mà không làm gián đoạn pipeline).*

---

## 6. Bảng Tổng hợp Toàn bộ File Dữ liệu Đầu ra trong `data/splits/`

Cấu trúc cây thư mục đầy đủ tại [`data/splits/`](../data/splits/):

```
data/splits/
├── train_combined_balanced.csv    # [CHÍNH THỨC] Tập Train Cân bằng 1:1 Quy mô Lớn (58,958 imgs: FF++ & Celeb Real + DF40 Fake)
├── val_combined_balanced.csv      # [CHÍNH THỨC] Tập Val Cân bằng 1:1 (6,550 imgs)
├── train_pool_693k.csv            # Full-scale Train Pool (652,421 imgs: 622k fake + 32k real)
├── val_pool.csv                   # Full-scale Val Pool (72,491 imgs)
├── celeb_df_extracted_real_frames.csv # Manifest 10,336 Real frames trích xuất từ Celeb-DF-v2
├── test_full.csv                  # [CHÍNH THỨC] Full Benchmark Test Suite (29,691 imgs)
├── test_full_detailed.csv         # Full Benchmark Test Suite kèm metadata
├── train.csv                      # Identity-Disjoint 70% Train (20,853 imgs: 834 real [FF++ & Celeb] + 20,019 fake)
├── val.csv                        # Identity-Disjoint 15% Val (4,440 imgs)
├── test.csv                       # Identity-Disjoint 15% Test (4,398 imgs)
├── train_balanced.csv             # Sub-benchmark 1:1 Balanced Train (1,668 imgs: 834 real, 834 fake)
├── val_balanced.csv               # Sub-benchmark 1:1 Balanced Val (346 imgs: 173 real, 173 fake)
├── test_balanced.csv              # Sub-benchmark 1:1 Balanced Test (340 imgs: 170 real, 170 fake)
├── train_detailed.csv             # Full metadata Train (path, label, method, identity, domain, video)
├── val_detailed.csv               # Full metadata Val
├── test_detailed.csv              # Full metadata Test
├── split_info.json                # Master manifest JSON ghi nhận tham số chia tách & thống kê
├── methods_summary.json           # Thống kê chi tiết từng method
└── methods/                       # 195 File CSV đánh giá riêng cho 40 phương pháp
    ├── test_<method>_balanced.csv
    ├── test_<method>_full.csv
    ├── test_<method>_detailed.csv
    ├── benchmark_test_<method>_balanced.csv
    └── benchmark_test_<method>_full.csv
```

---

## 7. Kết quả Kiểm định Chất lượng & Trực quan hóa Visual EDA

### 7.1 Kết quả Unit Test Tự động (`tests/test_data_prep.py`)
Đã thực thi `python3 -m unittest tests/test_data_prep.py`: **7/7 Test Cases Đạt Chuẩn (PASSED)**.

| Ca Kiểm Thử | Mục Đích Kiểm Định | Kết Quả | Đảm Bảo Kỹ Thuật |
| :--- | :--- | :---: | :--- |
| `test_split_files_exist` | Kiểm tra tồn tại 21 file split gốc | **PASSED** | Toàn bộ file CSV và JSON metadata tồn tại, dung lượng hợp lệ. |
| `test_method_specific_splits_exist` | Kiểm tra 195 file split per-method | **PASSED** | $\\ge 39$ methods có đủ 5 file CSV tương ứng trong `methods/`. |
| `test_identity_disjoint_leakage` | Kiểm tra rò rỉ danh tính 3 tập | **PASSED** | **0% rò rỉ danh tính** ($\\text{Train} \\cap \\text{Val} \\cap \\text{Test} = \\emptyset$). |
| `test_balanced_splits_exact_ratio` | Kiểm tra tỷ lệ 1:1 master splits | **PASSED** | Đúng tỷ lệ 1:1 (Real = Fake) ở `train_balanced`, `val_balanced`, `test_balanced`. |
| `test_method_balanced_exact_ratio` | Kiểm tra tỷ lệ 1:1 method splits | **PASSED** | Đúng tỷ lệ 1:1 ở tất cả file `test_<method>_balanced.csv`. |
| `test_split_info_metadata` | Kiểm tra JSON schema & seed | **PASSED** | `seed=42`, `total_methods=39`, phân rã đúng metadata. |
| `test_dataloader_batch_loading` | Kiểm tra nạp batch qua PyTorch | **PASSED** | Nạp batch tensor `(8, 3, 256, 256)` thành công với `DeepfakeDataset`. |

---

### 7.2 Danh mục Biểu đồ Phân tích EDA Đã Xuất Bản
Toàn bộ biểu đồ lưu tại [`experiments/plots/eda_comprehensive/`](../experiments/plots/eda_comprehensive/):

* `01_multi_dataset_infrastructure.png` — Hạ tầng đa nguồn & tỷ lệ mất cân bằng
* `02_deepfake_family_taxonomy.png` — Phân loại 4 họ thao túng
* `03_training_pool_gallery.png` & `04_test_benchmark_gallery.png` — Visual galleries ảnh thật & ảnh giả
* `05_zoomed_artifact_inspection.png` — Soi vi mô lỗ chân lông và vết mờ biên ghép
* `06_pixel_sharpness_dynamics.png` — Phổ sắc nét biên Laplacian
* `07_fft_frequency_power_spectrum.png` — Phổ công suất 2D FFT (Natural vs GAN/Diff)
* `08_post_split_method_distribution.png` — Phân bố hậu chia tách
* `09_post_split_visual_gallery.png` — Visual gallery kiểm định nhãn Train vs Val
* `10_post_split_master_overview.png` — Dashboard tổng quan Real/Fake & 23k IDs
* `11_40methods_test_evaluation_suite.png` — Bảng phân rã trực quan 40 phương pháp
* `12_high_scale_balanced_training_pool.png` — Cơ cấu 50% Real (FF++ + Celeb-DF) + 50% DF40 Fake

---

## 8. Hướng dẫn Sử dụng Chi tiết trong Training & Evaluation

### Lệnh 1: Huấn luyện Chính thức Cân bằng 1:1 Quy mô Lớn (Best Practice)
```bash
python3 src/training/train.py \\
    --train-csv data/splits/train_combined_balanced.csv \\
    --val-csv data/splits/val_combined_balanced.csv \\
    --test-csv data/splits/test_balanced.csv
```

### Lệnh 2: Huấn luyện Thử nghiệm Nhanh (Fast Prototyping)
```bash
python3 src/training/train.py \\
    --train-csv data/splits/train_balanced.csv \\
    --val-csv data/splits/val_balanced.csv \\
    --test-csv data/splits/test_balanced.csv
```

### Lệnh 3: Đánh giá Checkpoint trên Toàn bộ 40 Phương pháp
```bash
python3 src/eval/eval_df40_all_methods.py \\
    --checkpoint experiments/checkpoints/best_model.pt \\
    --manifest data/splits/test_full_detailed.csv \\
    --output experiments/results/eval/df40_40methods_benchmark.json
```

### Lệnh 4: Đánh giá Độc lập trên một Phương pháp Cụ thể (ví dụ DiT hoặc SadTalker)
```bash
python3 src/eval/evaluate.py \\
    --checkpoint experiments/checkpoints/best_model.pt \\
    --test-csv data/splits/methods/test_DiT_balanced.csv
```

---

## Cross-Reference Links

- Quy chuẩn Tài liệu: [MD_CONVENTION.md](rules/MD_CONVENTION.md)
- Kế hoạch Phân tách Dữ liệu: [DATA_PREP.md](phases/DATA_PREP.md)
- Nhật ký Tiến độ: [DATA_PREP_STATUS.md](progress/DATA_PREP_STATUS.md)
- Bản đồ Dự án Tổng thể: [OVERVIEW.md](OVERVIEW.md)
- Script Phân tách Dữ liệu: [prepare_df40_splits.py](../src/data/prepare_df40_splits.py)
- Script Trích xuất Celeb-DF Train: [extract_celeb_df_frames.py](../src/data/extract_celeb_df_frames.py)
- Script Trích xuất Celeb-DF Test Suite: [extract_celeb_df_test_suite.py](../src/data/extract_celeb_df_test_suite.py)
- Bộ Kiểm thử Tự động: [test_data_prep.py](../tests/test_data_prep.py)
- Notebook Phân tích EDA: [00_comprehensive_dataset_eda.ipynb](../notebooks/00_comprehensive_dataset_eda.ipynb)


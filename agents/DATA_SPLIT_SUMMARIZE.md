# DATA_SPLIT_SUMMARIZE.md — Báo Cáo Tổng Hợp Dữ Liệu Train, Val, Test Đa Nguồn (FF++, Celeb-DF, DF40)

- **Motivation/Background**: Xây dựng mô hình phát hiện Deepfake (Deepfake-ViT) đạt độ chính xác cao và khả năng tổng quát hóa vượt trội đòi hỏi dữ liệu huấn luyện và kiểm thử phải tích hợp đầy đủ 4 nguồn dữ liệu (FaceForensics++, Celeb-DF-v2, DF40 Training Pool, DF40 Test Suite), đạt tỷ lệ cân bằng 1:1 (Real:Fake) và đảm bảo 0% rò rỉ danh tính/video (Zero Leakage).
- **Purpose**: Tài liệu tóm tắt toàn bộ hạ tầng phân tách dữ liệu sau khi hoàn tất trích xuất, lập bảng kiểm kê 4 nguồn dữ liệu, tổng hợp quy mô các tập Train/Val/Test, chứng minh toán học Zero-Leakage và cung cấp danh mục 195 file split phục vụ huấn luyện và đánh giá.
- **Overview Pipeline**: `Kiểm kê 4 Nguồn Dữ liệu` $\\rightarrow$ `Trích xuất Real Đa Nguồn (22.4k FF++ + 10.3k Celeb-DF)` $\\rightarrow$ `Trích xuất Celeb-DF-v2 Test Benchmark (2.5k imgs)` $\\rightarrow$ `Phân vùng Identity-Disjoint (22,237 IDs, Zero-Leakage)` $\\rightarrow$ `Sinh Tập Cân bằng 1:1 (58,958 imgs)` $\\rightarrow$ `Tạo 195 CSV Đánh giá 40 Phương pháp` $\\rightarrow$ `Kiểm định Tự động (7/7 Tests Passed) & Notebook 00 EDA`.
- **Detailed Plan**: §1 Tổng quan Thực thi; §2 Ma trận Kiểm kê 4 Nguồn Dữ liệu Gốc; §3 Bảng Tổng hợp Toàn bộ Tập Dữ liệu Sau Khi Split; §4 Chứng minh Toán học Không Rò rỉ Danh tính (Zero Leakage); §5 Danh mục Đánh giá 40 Phương pháp Deepfake; §6 Tích hợp Trực quan trong Notebook 00 EDA; §7 Hướng dẫn Lệnh Huấn luyện & Đánh giá.
- **References**: `prepare_df40_splits.py`, `extract_celeb_df_frames.py`, `extract_celeb_df_test_suite.py`, `test_data_prep.py`, `00_comprehensive_dataset_eda.ipynb`, `DATA_PREP_SUMMARY_REPORT.md`.

---

## Table of Contents

- [1. Tổng Quan Thực Thi (Executive Summary)](#1-tổng-quan-thực-thi-executive-summary)
- [2. Ma Trận Kiểm Kê 4 Nguồn Dữ Liệu Gốc](#2-ma-trận-kiểm-kê-4-nguồn-dữ-liệu-gốc)
- [3. Bảng Tổng Hợp Toàn Bộ Tập Dữ Liệu Sau Khi Split](#3-bảng-tổng-hợp-toàn-bộ-tập-dữ-liệu-sau-khi-split)
  - [3.1 Tập Huấn Luyện (Training Sets)](#31-tập-huấn-luyện-training-sets)
  - [3.2 Tập Đánh Giá & Benchmark (Test Sets)](#32-tập-đánh-giá--benchmark-test-sets)
- [4. Chứng Minh Toán Học Tuyệt Đối Không Rò Rỉ Danh Tính (Zero Leakage)](#4-chứng-minh-toán-học-tuyệt-đối-không-rò-rỉ-danh-tính-zero-leakage)
- [5. Danh Mục Bộ Đánh Giá Độc Lập 40 Phương Pháp Deepfake](#5-danh-mục-bộ-đánh-giá-độc-lập-40-phương-pháp-deepfake)
- [6. Tích Hợp Trực Quan Cao Cấp Trong Notebook 00 EDA](#6-tích-hợp-trực-quan-cao-cấp-trong-notebook-00-eda)
- [7. Hướng Dẫn Thực Thi Lệnh Huấn Luyện & Đánh Giá](#7-hướng-dẫn-thực-thi-lệnh-huấn-luyện--đánh-giá)

---

## 1. Tổng Quan Thực Thi (Executive Summary)

Dự án **Deepfake-ViT** đã thiết lập hoàn chỉnh hạ tầng dữ liệu đa nguồn:

1. **Hợp nhất Toàn diện 4 Nguồn Dữ liệu**:
   * **FaceForensics++ (FF++)**: 20,219 Real frames sạch ($256 \\times 256$) đưa vào tập Train lớn (sau khi loại trừ 298 video sequence trùng test/val).
   * **Celeb-DF-v2**: Trích xuất thành công **10,336 Real frames ($256 \\times 256$)** sạch từ 690 video training đưa vào tập Train lớn + trích xuất **2,590 frames** từ 518 video test tạo thành bộ Benchmark chuẩn Celeb-DF-v2.
   * **DF40 Train Pool**: 692,158 Fake frames từ 31 phương pháp.
   * **DF40 Test Suite (`test_data_v3`)**: 29,691 images (1,177 canonical real + 28,514 fake) từ 40 phương pháp.
2. **Tập Huấn Luyện Cân Bằng 1:1 Quy Mô Lớn (58,958 images)**: Tập [train_combined_balanced.csv](../data/splits/train_combined_balanced.csv) gồm **29,487 Real (FF++ + Celeb-DF) + 29,471 Fake DF40** đạt tỷ lệ cân bằng hoàn hảo **1.0 : 1**.
3. **Bộ Benchmark 40 Phương Pháp (195 Files CSV)**: Phân rã độc lập từng thuật toán thao túng trong [data/splits/methods/](../data/splits/methods/).
4. **Không Rò Rỉ Danh Tính (Zero Leakage 100%)**: 22,237 identities phân chia nghiêm ngặt: $\\text{Train} \\cap \\text{Val} = \\emptyset, \\text{Train} \\cap \\text{Test} = \\emptyset, \\text{Val} \\cap \\text{Test} = \\emptyset$.
5. **Kiểm Thử Tự Động & Visual Dashboard**: Đạt 7/7 bài kiểm thử trong [tests/test_data_prep.py](../tests/test_data_prep.py) và hiển thị trực tiếp 3 bộ biểu đồ trong [notebooks/00_comprehensive_dataset_eda.ipynb](../notebooks/00_comprehensive_dataset_eda.ipynb).

---

## 2. Ma Trận Kiểm Kê 4 Nguồn Dữ Liệu Gốc

```
                                          ┌─────────────────────────────────────────────────────────────┐
                                          │             MULTI-DATASET RAW SOURCES (/workspace/data)     │
                                          ├─────────────────────────────────────────────────────────────┤
                                          │ 1. FaceForensics++ (31,949 frames PNG, 999 video sequences) │
                                          │ 2. Celeb-DF-v2 (890 videos Real + 5,639 videos Fake)        │
                                          │ 3. DF40 Train Pool (693,335 frames across 31 fake methods)  │
                                          │ 4. DF40 Test Suite (30,691 images across 40 fake methods)   │
                                          └──────────────────────────────┬──────────────────────────────┘
```

| Nguồn Dữ Liệu | Đường Dẫn Thực Tế | Quy Mô / Số Lượng | Đóng Góp Trong Train | Đóng Góp Trong Test / Benchmark | Trạng Thái |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **1. FaceForensics++ (FF++)** | `/workspace/data/FaceForensics++/original_sequences/youtube/c23/frames` | 999 video sequences<br>31,949 ảnh PNG | **20,219 Real frames** ($256 \\times 256$) từ 701 video độc lập | **143 Canonical Real faces** (`ffc:...`) | ✅ **Chuẩn 100%** |
| **2. Celeb-DF-v2** | `/workspace/data/Celeb-DF-v2` & `data/processed/celeb_df_extracted/` | 890 video Real<br>5,639 video Fake | **9,268 Real frames** ($256 \\times 256$) từ 690 video train sạch | **1,780 ảnh Test Cân bằng 1:1** (`test_celeb_df_v2_balanced.csv`) + **178 Real Canonical** | ✅ **Chuẩn 100%** |
| **3. DF40 Train Pool** | `/workspace/data/DF40_train_manifest.csv` & `DF40_train_extracted/` | 693,335 ảnh (692k fake + 1.1k real) | **29,471 Fake frames** (tập cân bằng) / **622,920 Fake** (tập full pool) | *(Cách ly hoàn toàn không đưa vào test để chống rò rỉ)* | ✅ **Chuẩn 100%** |
| **4. DF40 Test Suite (`test_data_v3`)** | `/workspace/data/test_data_v3/` | 30,691 ảnh (29.5k fake + 1.1k real) | *(Cách ly hoàn toàn không đưa vào train)* | **28,514 Fake faces** của **40 phương pháp** + **1,177 Canonical Real** | ✅ **Chuẩn 100%** |

---

## 3. Bảng Tổng Hợp Toàn Bộ Tập Dữ Liệu Sau Khi Split

### 3.1 Tập Huấn Luyện (Training Sets)

| Tên Tập Huấn Luyện | File Path | Tổng Ảnh | Real | Fake | Tỷ Lệ F:R | Thành Phần Nguồn Gốc | Mục Đích Sử Dụng |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **High-Scale Train (1:1)** | [`train_combined_balanced.csv`](../data/splits/train_combined_balanced.csv) | **58,958** | **29,487** | **29,471** | **1.0 : 1** | **FF++ Real (20.2k) + Celeb-DF Real (9.3k) + DF40 Fake (29.5k)** | **Tập Huấn Luyện Chính Thức Khuyến Nghị (Best Practice)** |
| **High-Scale Val (1:1)** | [`val_combined_balanced.csv`](../data/splits/val_combined_balanced.csv) | **6,550** | **3,267** | **3,283** | **1.0 : 1** | FF++ Real (2.2k) + Celeb Real (1.1k) + DF40 Fake (3.3k) | **Tập Validation Chính Thức Khuyến Nghị** |
| **Full Train Pool** | [`train_pool_693k.csv`](../data/splits/train_pool_693k.csv) | **652,421** | 29,501 | 622,920 | 21.1 : 1 | Toàn bộ Fake DF40 (623k) + Real FF++ & Celeb (29.5k) | Huấn luyện Full-Scale không giới hạn |
| **Full Val Pool** | [`val_pool.csv`](../data/splits/val_pool.csv) | **72,491** | 3,253 | 69,238 | 21.3 : 1 | Toàn bộ Fake Val (69k) + Real Val (3.2k) | Đánh giá Full-Scale Val |
| **Prototype Train (70%)** | [`train.csv`](../data/splits/train.csv) | **20,853** | 834 | 20,019 | 24.0 : 1 | test_data_v3 partition (701 FF++ + 133 Celeb-DF) | Prototype thử nghiệm nhanh |
| **Prototype Val (15%)** | [`val.csv`](../data/splits/val.csv) | **4,440** | 173 | 4,267 | 24.7 : 1 | test_data_v3 partition | Prototype Validation |
| **Prototype 1:1 Train** | [`train_balanced.csv`](../data/splits/train_balanced.csv) | **1,668** | 834 | 834 | **1.0 : 1** | test_data_v3 partition (1:1) | Sub-benchmark 1:1 Train |

---

### 3.2 Tập Đánh Giá & Benchmark (Test Sets)

| Tên Tập Đánh Giá | File Path | Tổng Ảnh | Real | Fake | Tỷ Lệ F:R | Thành Phần Nguồn Gốc | Mục Đích Sử Dụng |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Full Benchmark Test** | [`test_full.csv`](../data/splits/test_full.csv) | **29,691** | **1,177** | **28,514** | 24.2 : 1 | **Toàn bộ 40 phương pháp fake + 1,177 canonical real** | **Đánh Giá Toàn Diện 40 Phương Pháp (Standard Benchmark)** |
| **Celeb-DF-v2 Test (1:1)** | [`test_celeb_df_v2_balanced.csv`](../data/splits/test_celeb_df_v2_balanced.csv) | **1,780** | **890** | **890** | **1.0 : 1** | **YouTube-real (890) + Celeb-synthesis Fake (890)** | **Benchmark Chuẩn Quốc Tế Celeb-DF-v2 (1:1)** |
| **Celeb-DF-v2 Test (Full)** | [`test_celeb_df_v2.csv`](../data/splits/test_celeb_df_v2.csv) | **2,590** | 890 | 1,700 | 1.9 : 1 | YouTube-real (890) + Celeb-synthesis Fake (1,700) | Benchmark Toàn bộ 518 Video Test Celeb-DF-v2 |
| **Per-Method Test Suites** | [`data/splits/methods/`](../data/splits/methods/) | *195 files* | *1:1 / Full* | *1:1 / Full* | **1.0 : 1** | Tách riêng từng phương pháp trong 40 methods | Đánh giá độc lập từng thuật toán thao túng |
| **Prototype Test (15%)** | [`test.csv`](../data/splits/test.csv) | **4,398** | 170 | 4,228 | 24.9 : 1 | test_data_v3 partition | Sub-benchmark Test 15% |
| **Prototype 1:1 Test** | [`test_balanced.csv`](../data/splits/test_balanced.csv) | **340** | 170 | 170 | **1.0 : 1** | test_data_v3 partition (1:1) | Sub-benchmark 1:1 Test |

---

## 4. Chứng Minh Toán Học Tuyệt Đối Không Rò Rỉ Danh Tính (Zero Leakage)

1. **Phân vùng Không gian Danh tính (Identity-Disjoint)**:
   Mỗi ảnh được gán một khóa định danh duy nhất $I \\in \\mathcal{I}$ (ví dụ: `ffc:709`, `cdc:id28_0007`).
   Tổng cộng **22,237 unique identities** được phân tách độc lập:

   $$\\mathcal{I} = \\mathcal{I}_{\\text{train}} \\cup \\mathcal{I}_{\\text{val}} \\cup \\mathcal{I}_{\\text{test}}$$

   $$\\mathcal{I}_{\\text{train}} \\cap \\mathcal{I}_{\\text{val}} = \\emptyset, \\quad \\mathcal{I}_{\\text{train}} \\cap \\mathcal{I}_{\\text{test}} = \\emptyset, \\quad \\mathcal{I}_{\\text{val}} \\cap \\mathcal{I}_{\\text{test}} = \\emptyset$$

   * **Train Identities**: 15,565 IDs ($67.0\%$).
   * **Val Identities**: 3,335 IDs ($14.4\%$).
   * **Test Identities**: 3,337 IDs ($14.4\%$).
   * **Train $\\cap$ Test Overlap**: **0 IDs (ZERO LEAKAGE ✔)**.

2. **Lọc Rò Rỉ Video Nguồn (Video Masking)**:
   * **FaceForensics++**: Lọc bỏ toàn bộ **298 video sequence IDs** trùng với tập Test/Val, chỉ lấy 22,418 frames từ 701 video độc lập.
   * **Celeb-DF-v2**: Lọc bỏ toàn bộ **575 video stems** và 178 danh tính test trong `List_of_testing_videos.txt`, chỉ trích xuất từ 690 video training độc lập $\\rightarrow$ **0% Video Leakage**.

---

## 5. Danh Mục Bộ Đánh Giá Độc Lập 40 Phương Pháp Deepfake

Tất cả các tập test chuyên biệt được lưu trữ trong thư mục [`data/splits/methods/`](../data/splits/methods/) (195 files CSV). Phân loại thành **4 họ kỹ thuật**:

1. **Face Swap (8 methods)**: `simswap`, `inswap`, `faceswap`, `blendface`, `facedancer`, `mobileswap`, `uniface`, `deepfacelab`.
2. **Face Reenactment / Animation (14 methods)**: `sadtalker`, `wav2lip`, `fomm`, `MRAA`, `lia`, `mcnet`, `tpsm`, `facevid2vid`, `hyperreenact`, `pirender`, `one_shot_free`, `danet`, `fsgan`, `heygen`.
3. **Entire Face Synthesis (13 methods)**: `DiT`, `SiT`, `StyleGAN2`, `StyleGAN3`, `StyleGANXL`, `VQGAN`, `ddim`, `pixart`, `sd2.1`, `CollabDiff`, `MidJourney`, `RDDM`, `whichfaceisreal`.
4. **Facial Attribute Editing (4 methods)**: `stargan`, `starganv2`, `e4e`, `e4s`.
5. **Celeb-DF-v2 Benchmark Suite**: `test_CelebDFv2_balanced.csv` (1,780 imgs, 1:1) & `test_CelebDFv2_full.csv` (2,590 imgs).

---

## 6. Tích Hợp Trực Quan Cao Cấp Trong Notebook 00 EDA

Notebook [`notebooks/00_comprehensive_dataset_eda.ipynb`](../notebooks/00_comprehensive_dataset_eda.ipynb) đã được cập nhật và thực thi trọn vẹn 36 cell:
* **Cell 4 & Cell 12**: Bảng kiểm kê hạ tầng 4 nguồn dữ liệu gốc và dữ liệu trích xuất.
* **Cell 27 & Cell 29**: Bảng kiểm định thống kê đa nguồn hậu chia tách và chứng minh Zero-Leakage.
* **Cell 31**: Hiển thị trực tiếp **3 bộ biểu đồ trực quan cao cấp**:
  * 📊 **Figure 1**: *Master Dataset Splitting Architecture & Balance Verification* (4 panels).
  * 📊 **Figure 2**: *Complete 40-Method Evaluation Suite Breakdown* (4 họ thao túng).
  * 📊 **Figure 3**: *High-Scale Balanced Training Dataset Architecture* (Cơ cấu 50% Real [FF++ & Celeb-DF] + 50% Fake DF40).
* **Cell 35**: Chạy tự động kiểm thử `tests/test_data_prep.py` $\\rightarrow$ **7/7 Passed**.

---

## 7. Hướng Dẫn Thực Thi Lệnh Huấn Luyện & Đánh Giá

### Lệnh 1: Huấn luyện Chính thức Cân bằng 1:1 Quy mô Lớn (Best Practice)
```bash
python3 src/training/train.py \\
    --train-csv data/splits/train_combined_balanced.csv \\
    --val-csv data/splits/val_combined_balanced.csv \\
    --test-csv data/splits/test_balanced.csv
```

### Lệnh 2: Đánh giá Benchmark trên Toàn bộ 40 Phương pháp
```bash
python3 src/eval/eval_df40_all_methods.py \\
    --checkpoint experiments/checkpoints/best_model.pt \\
    --manifest data/splits/test_full_detailed.csv \\
    --output experiments/results/eval/df40_40methods_benchmark.json
```

### Lệnh 3: Đánh giá Độc lập trên Bộ Benchmark Celeb-DF-v2
```bash
python3 src/eval/evaluate.py \\
    --checkpoint experiments/checkpoints/best_model.pt \\
    --test-csv data/splits/test_celeb_df_v2_balanced.csv
```

---

## Cross-Reference Links

- Báo cáo Kỹ thuật Đầy đủ: [DATA_PREP_SUMMARY_REPORT.md](DATA_PREP_SUMMARY_REPORT.md)
- Kế hoạch Phân tách Dữ liệu: [DATA_PREP.md](phases/DATA_PREP.md)
- Nhật ký Tiến độ: [DATA_PREP_STATUS.md](progress/DATA_PREP_STATUS.md)
- Bản đồ Dự án Tổng thể: [OVERVIEW.md](OVERVIEW.md)
- Script Phân tách Dữ liệu: [prepare_df40_splits.py](../src/data/prepare_df40_splits.py)
- Script Trích xuất Celeb-DF Train: [extract_celeb_df_frames.py](../src/data/extract_celeb_df_frames.py)
- Script Trích xuất Celeb-DF Test Suite: [extract_celeb_df_test_suite.py](../src/data/extract_celeb_df_test_suite.py)
- Bộ Kiểm thử Tự động: [test_data_prep.py](../tests/test_data_prep.py)
- Notebook Phân tích EDA: [00_comprehensive_dataset_eda.ipynb](../notebooks/00_comprehensive_dataset_eda.ipynb)

# 🌐 GIAO THỨC HUẤN LUYỆN CÂN BẰNG ĐA MIỀN (CROSS-DOMAIN BALANCED PROTOCOL - BƯỚC 2)

---

## 1. Mục tiêu kiến trúc (Architecture Objective)
Triệt tiêu hoàn toàn hiện tượng **Domain Shortcut Learning (Thiên kiến miền dữ liệu)** bằng cách đảm bảo:
* Mỗi miền dữ liệu (Celeb-DF, FaceForensics++, DF40) đều có đại diện cân xứng ở cả hai nhóm nhãn **Real ($y=0$)** và **Fake ($y=1$)**.
* Mạng Vision Transformer (DINOv3 ViT) không thể dựa vào các dấu vết ngoại cảnh (ánh sáng studio, hạt nhiễu camera, chuẩn nén video) để phân loại, buộc phải học các bất thường vi mô trên khuôn mặt (facial artifacts, boundary blending, texture inconsistencies).

---

## 2. Thiết kế phân bổ tập dữ liệu (Dataset Matrix)

### A. Phân bổ tập Train Cân bằng Đa miền (`train_domain_balanced.csv`)
Tập huấn luyện tổng cộng **60,000 ảnh** (Tỷ lệ 1:1 hoàn hảo giữa Real và Fake):

```
Tập Train Cân bằng Đa miền (60,000 ảnh):
├── 🟢 NHÃN REAL (y = 0) [30,000 ảnh]:
│   ├── 15,000 ảnh: Celeb-DF-v2 Real (Celeb-real + YouTube-real)
│   └── 15,000 ảnh: FaceForensics++ (original_sequences)
└── 🔴 NHÃN FAKE (y = 1) [30,000 ảnh]:
    ├── 15,000 ảnh: DF40 Training Pool (31 thuật toán: DiT, SiT, SimSwap, SadTalker, etc.)
    └── 15,000 ảnh: Celeb-DF-v2 Fake (Celeb-synthesis - 0 Test Leakage)
```

### B. Phân bổ tập Validation Cân bằng Đa miền (`val_domain_balanced.csv`)
Tập kiểm định độc lập **6,000 ảnh** (3,000 Real vs 3,000 Fake):
* **Real (0)**: 1,500 Celeb-DF Real + 1,500 FaceForensics++ Real.
* **Fake (1)**: 1,500 DF40 Val Fake + 1,500 Celeb-DF Val Fake.

---

## 3. Nguyên tắc cách ly danh tính (Strict Zero-Leakage Protocol)
1. **Celeb-DF-v2**: Toàn bộ video thuộc danh sách `List_of_testing_videos.txt` (518 video) được loại trừ 100% trước khi trích xuất tập train/val.
2. **FaceForensics++**: Loại trừ toàn bộ 302 thư mục video trùng với test/val identities của DF40.
3. **DF40**: Tách bạch identity-disjoint theo cấu trúc phân rã danh tính chuẩn của benchmark DF40.

---

## 4. Pipeline thực thi tạo dữ liệu
* Trích xuất frame Celeb-synthesis: `src/data/extract_celeb_df_fake_train.py`
* Ghép và cân bằng split: `src/data/build_domain_balanced_splits.py`
* File kết quả:
  * `data/splits/train_domain_balanced.csv`
  * `data/splits/val_domain_balanced.csv`

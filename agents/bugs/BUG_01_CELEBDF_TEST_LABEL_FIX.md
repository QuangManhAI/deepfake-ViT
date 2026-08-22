# 🐛 BÁO CÁO KỸ THUẬT: SỬA LỖI GÁN NHÃN CELEB-DF TEST VÀ ĐỒNG BỘ BENCHMARK (BUG-01)

---

## 1. Tóm tắt sự cố (Executive Summary)

* **Hiện tượng:** Khi chạy đánh giá trên tập `test_balanced.csv`, mô hình ban đầu cho chỉ số Accuracy đạt mức **97.21%**, nhưng khi kiểm tra chi tiết theo từng phương pháp sinh ảnh thì phát hiện một số nhóm ảnh Fake bị dự đoán nhãn $0$ (Real) nhưng vẫn được tính là đoán đúng.
* **Nguyên nhân gốc rễ (Root Cause):** Lỗi logic trong script `src/data/prepare_df40_splits.py` (dòng 416). Điều kiện kiểm tra `fname.startswith("fake_")` đã bỏ sót toàn bộ 1,700 ảnh có tên `celeb_test_fake_...` (do bắt đầu bằng tiền tố `"celeb_"`), khiến toàn bộ 1,700 ảnh Celeb-DF test Fake bị gán nhãn nhầm thành `label = 0` (Real).
* **Mức độ nghiêm trọng:** Cao (Ảnh hưởng đến tính chính xác của chỉ số đánh giá trên tập Test tổng hợp).
* **Trạng thái:** **ĐÃ SỬA VÀ KIỂM ĐỊNH TOÀN DIỆN (RESOLVED & VERIFIED)**.

---

## 2. Chi tiết phân tích nguyên nhân kỹ thuật

### A. Đoạn mã gây lỗi trong `src/data/prepare_df40_splits.py`
```python
# MÃ NGUỒN CŨ BỊ LỖI:
elif celeb_test_dir.exists():
    celeb_imgs = list(celeb_test_dir.glob("*.png"))
    for img_p in celeb_imgs:
        fname = img_p.name
        is_fake = 1 if fname.startswith("fake_") else 0  # 🚨 LỖI: fname = "celeb_test_fake_..." -> trả về 0!
        v_name = fname.replace("fake_", "").replace("real_", "").rsplit("_frame", 1)[0]
        ...
```

### B. Cơ chế sai lệch (Failure Mechanism)
1. Thư mục `data/processed/celeb_df_test_extracted` chứa:
   - 890 ảnh `celeb_test_real_...` (Thực chất là Real, $y=0$).
   - 1,700 ảnh `celeb_test_fake_...` (Thực chất là Fake, $y=1$).
2. Vì `fname.startswith("fake_")` luôn trả về `False`, cả 1,700 ảnh Fake này bị gán `label = 0`.
3. Khi ghép vào `test_balanced.csv` và `test_full.csv`, 1,700 ảnh này được tính vào nhóm Real ($y=0$).
4. Trong tập Train, toàn bộ ảnh Real được lấy từ Celeb-DF & FF++, nên mô hình đã học nhận diện phong cách màu/camera của Celeb-DF là Real.
5. Khi test, mô hình đoán nhãn `0` cho các ảnh Celeb-DF này, trùng với nhãn `0` trong CSV, dẫn đến chỉ số accuracy bị thổi phồng lên **97.21%**.

---

## 3. Các bước khắc phục đã thực hiện

### Bước 1: Sửa mã nguồn logic gán nhãn
Đã cập nhật logic kiểm tra trong `src/data/prepare_df40_splits.py`:
```python
# MÃ NGUỒN ĐÃ SỬA:
is_fake = 1 if ("fake" in fname.lower() and "real" not in fname.lower()) or fname.startswith("fake_") or "test_fake" in fname.lower() else 0
v_name = fname.replace("celeb_test_fake_", "").replace("celeb_test_real_", "").replace("fake_", "").replace("real_", "").rsplit("_frame", 1)[0]
```

### Bước 2: Tái tạo toàn bộ các file split dữ liệu
Đã chạy lại pipeline dữ liệu để làm mới các file split:
* `data/splits/test_full.csv` (33,281 ảnh): 1,700 ảnh Celeb-DF Fake chuyển về đúng nhãn **`1` (Fake)**.
* `data/splits/test.csv` (33,281 ảnh): 1,700 ảnh Celeb-DF Fake chuyển về đúng nhãn **`1` (Fake)**.
* `data/splits/test_balanced.csv` (4,134 ảnh): Được tái tạo chuẩn 1:1 với **2,067 ảnh Real ($y=0$)** và **2,067 ảnh Fake ($y=1$)** với ground-truth chính xác 100%.

---

## 4. Bảng so sánh trước và sau khi sửa lỗi

| Tiêu chí | Trước khi sửa (Bị lỗi nhãn) | Sau khi sửa (Chuẩn hóa chính xác) |
| :--- | :---: | :---: |
| Nhãn của 1,700 ảnh `celeb_test_fake` | `label = 0` (Real - Sai lệch) | **`label = 1` (Fake - Chính xác 100%)** |
| Nhãn của 890 ảnh `celeb_test_real` | `label = 0` (Real) | **`label = 0` (Real)** |
| Cấu trúc tập `test_balanced.csv` | 3,767 Real (lẫn 1,700 fake) / 3,767 Fake | **2,067 Real / 2,067 Fake (1:1 chuẩn)** |
| Accuracy tổng thể trên `test_balanced.csv` | 97.21% (Bị làm sai lệch) | **93.57% (Chính xác thực tế)** |
| Specificity trên ảnh Real DF40 | 98.22% | **98.22%** (Rất cao và ổn định) |
| Recall trên ảnh Fake DF40 | 95.30% | **93.73%** |

---

## 5. Kết luận & Khuyến nghị
1. Lỗi gán nhãn đã được giải quyết hoàn toàn ở tầng tiền xử lý dữ liệu và split generation.
2. Mô hình hiện tại thể hiện năng lực phát hiện Deepfake rất mạnh trên benchmark DF40 (**>93.7% recall** trên hầu hết các dòng Diffusion và GAN).
3. Đã sẵn sàng bước tiếp theo: Mở rộng huấn luyện cân bằng đa miền (Domain-Balanced Training) và bổ sung Data Augmentation để tăng cường độ bền vững ngoài phân phối.

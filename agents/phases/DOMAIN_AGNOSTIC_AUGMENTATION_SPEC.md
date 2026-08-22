# 🎨 ĐẶC TẢ TĂNG CƯỜNG DỮ LIỆU ĐỘC LẬP MIỀN (DOMAIN-AGNOSTIC DATA AUGMENTATIONS - BƯỚC 3)

---

## 1. Lý do áp dụng (Rationale)
Vision Transformer (ViT) có xu hướng ghi nhớ các đặc trưng tần số cao đặc thù của chuẩn nén JPEG/H.264 và cảm biến camera. Để tăng cường khả năng tổng quát hóa ra ngoài phân phối (Out-of-Distribution Generalization), pipeline tiền xử lý cần phá vỡ các đặc trưng phong cách cục bộ.

---

## 2. Kiến trúc Data Augmentation Pipeline

```python
train_transform = transforms.Compose([
    # 1. Chuẩn hóa kích thước
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),

    # 2. Xóa bỏ thiên kiến màu sắc và ánh sáng camera
    transforms.ColorJitter(
        brightness=0.2,   # Thay đổi độ sáng ngẫu nhiên +/- 20%
        contrast=0.2,     # Thay đổi độ tương phản ngẫu nhiên +/- 20%
        saturation=0.2,   # Thay đổi độ bão hòa màu +/- 20%
        hue=0.05          # Thay đổi sắc độ màu ngẫu nhiên +/- 5%
    ),

    # 3. Triệt tiêu hạt nhiễu nén riêng (JPEG/Codec Artifact Invariance)
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
    ], p=0.3),

    # 4. Biến thiên độ sắc nét ngẫu nhiên (chống overfit ảnh quá nét/mờ)
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),

    # 5. Chuyển tensor và chuẩn hóa ImageNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # 6. Che phủ khuôn mặt ngẫu nhiên (Face Cutout / Random Erasing)
    # Ép Transformer chú ý vào toàn bộ khuôn mặt thay vì chỉ một vùng nhỏ
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value='random'),
])
```

---

## 3. Tác dụng kỹ thuật của từng phép biến đổi

| Phép biến đổi | Mục tiêu giải quyết | Hiệu quả đối với ViT |
| :--- | :--- | :--- |
| **ColorJitter** | Chống lại sự khác biệt tông màu giữa các studio quay video | Ép mô hình học hình học khuôn mặt thay vì màu da |
| **GaussianBlur** | Phá vỡ pattern nén video MP4/H.264 cục bộ | Buộc mạng tìm kiếm artifact ở tỷ lệ đa mức (multi-scale) |
| **RandomAdjustSharpness** | Mô phỏng các mức độ phân giải và chất lượng camera khác nhau | Cải thiện nhận diện trên video độ phân giải thấp/cao |
| **RandomErasing (Cutout)** | Ngăn mô hình chỉ tập trung vào một vị trí duy nhất (như miệng) | Kích hoạt nhiều attention heads phân tán trên toàn khuôn mặt |

---

## 4. Tích hợp vào Notebook và Training Scripts
* Notebook: [02_training_balanced_dataset.ipynb](../notebooks/02_training_balanced_dataset.ipynb)
* Script huấn luyện: [src/training/train.py](../src/training/train.py)

# Chạy dự án anti-deepfake trên RunPod (Ubuntu + GPU)

Mục tiêu: chạy eval (và fine-tune) DINOv3 trên GPU RunPod thay vì Mac 16GB.
Luồng chung: **local push lên Hugging Face Hub → RunPod pull về → cài → chạy**.

---

## 1. Trên máy LOCAL — push lên Hugging Face Hub (chỉ làm 1 lần)

### 1.1 Cài công cụ + đăng nhập

```bash
.venv/bin/pip install -U "huggingface_hub[cli]"
.venv/bin/hf login        # nhập token (https://huggingface.co/settings/tokens, quyền WRITE)
```

### 1.2 Tạo repo (1 lần)

```bash
.venv/bin/hf repo create dinov3-deepfake-detection --type model --private
.venv/bin/hf repo create df40-test-data-v3          --type dataset --private
```

### 1.3 Push (chạy script — tự exclude data lớn / cache)

```bash
bash src/utils/push_to_hub.sh              # code + weights (298MB) -> model repo
bash src/utils/push_dataset_to_hub.sh      # test_data_v3.zip (4.2GB) -> dataset repo
```

> Nếu repo đã tồn tại thì bỏ qua bước `hf repo create`, chạy thẳng script push.

---

## 2. Tạo pod trên RunPod

1. **Deploy → GPU pod → On-demand / Secure Cloud**.
2. **Template:** chọn **RunPod PyTorch** (Ubuntu 22.04 + CUDA + PyTorch) — nhanh nhất.
   Nếu muốn tự cài từ đầu: chọn **Ubuntu** rồi tự chạy `src/utils/setup_ubuntu.sh` (bước 3.3).
3. **GPU:** RTX 4090 (~$0.79/h) hoặc A10 (~$0.69/h) đủ cho eval + fine-tune nhỏ.
4. **Storage (tuỳ chọn):** gắn Network Volume ~10GB nếu muốn giữ data giữa các pod.
5. **Start pod → Connect → Web Terminal** (hoặc SSH).

---

## 3. TRONG POD — pull + cài + chạy

### 3.1 Pull code + models

```bash
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=hf_xxx            # hoặc hf login
git lfs install
git clone https://huggingface.co/ManhQuangAI/dinov3-deepfake-detection
cd dinov3-deepfake-detection
```

> Repo private → cần token. Hoặc dùng `hf download` thay cho git clone:
> `hf download ManhQuangAI/dinov3-deepfake-detection --local-dir .`

### 3.2 Pull dataset

```bash
hf download ManhQuangAI/df40-test-data-v3 --repo-type dataset --include test_data_v3.zip
unzip test_data_v3.zip -d .
```

### 3.3 Cài môi trường

```bash
# Nếu template là RunPod PyTorch (torch+CUDA đã có sẵn):
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Nếu template là Ubuntu thuần (tự cài mọi thứ):
bash src/utils/setup_ubuntu.sh
```

Kiểm tra GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3.4 Chạy eval (identity-disjoint, 40 method)

```bash
python src/eval/eval_identity_disjoint.py --model vit --root test_data_v3 --tag test_data_v3 --device cuda
python src/eval/eval_identity_disjoint.py --model cnn --root test_data_v3 --tag test_data_v3 --device cuda
```

Tạo report 40 method:

```bash
python src/experiments/make_method_report_md.py \
  --vit experiments/results/eval/identity_disjoint_v3_vit.json \
  --cnn experiments/results/eval/identity_disjoint_v3_cnn.json \
  --output experiments/results/eval/report_40_methods_v3.md
```

---

## 4. Fine-tune trên GPU (tuỳ chọn — đây là lý do chính thuê GPU)

Xem cách dùng của `src/training/train.py` và `src/training/finetune_compare.py`:

```bash
python src/training/train.py --help
```

Tham khảo `src/eval/eval_finetuned.py` để đánh giá checkpoint sau fine-tune.

---

## 5. Kiểm tra nhanh / gỡ lỗi

| Triệu chứng | Cách xử lý |
|---|---|
| `torch.cuda.is_available() = False` | Driver GPU thiếu / driver cũ → đổi `cu124` thành `cu121` trong `src/utils/setup_ubuntu.sh`, hoặc chọn template RunPod PyTorch |
| `OOM` khi extract | Giảm `--batch-size` (mặc định 16) |
| Lỗi thiếu file ảnh | Chưa `unzip test_data_v3.zip` hoặc chạy sai `--root` (phải là `test_data_v3`) |
| Pull repo private không được | Đã `export HF_TOKEN`? Token có quyền đọc repo đó? |

---

## 6. Tóm tắt kiến trúc

```
HF Hub
├── ManhQuangAI/dinov3-deepfake-detection   (model repo: code + weights 298MB)
└── ManhQuangAI/df40-test-data-v3           (dataset repo: test_data_v3.zip 4.2GB)
        │
RunPod (Ubuntu + CUDA)
└── pull → unzip → setup_ubuntu.sh → eval_identity_disjoint.py --device cuda
```

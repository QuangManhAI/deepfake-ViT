# Cách xây dựng `test_data_v3` và phân tích rò rỉ dữ liệu (data leakage)

> Bộ test chuẩn dùng cho đồ án anti-deepfake (DINOv3 ViT-S/16 vs ConvNeXt-Tiny).
> Mục tiêu của tài liệu: mô tả đúng quy trình tạo dữ liệu để người đọc có thể tin tưởng
> số liệu eval, đồng thời trả lời thẳng câu hỏi **"có rò rỉ hay không"**.

---

## 1. Tóm tắt

| | |
|---|---|
| Tổng ảnh | **30,691** (real = 1,177 · fake = 29,514) |
| Số method fake | **40** (đầy đủ DF40) |
| Nguồn real | Celeb-DF (178) + FF++ `original_sequences` (999) |
| Identity keys | **23,237** — mỗi người 1 key duy nhất |
| Identity PAIRED | **1,177** (mọi ảnh real đều có fake ghép) |
| Cấu trúc v3 | `real/` + `<method>/fake/` |
| Split | identity-disjoint, seed 42, 70/30 theo `identity` |

**Kết luận ngắn:** **KHÔNG có rò rỉ identity, không có ảnh lặp.** Nhưng protocol chỉ
đo khả năng tổng quát hóa **trong cùng dataset / cùng method** (khác người), *không*
phải zero-shot với method chưa từng thấy. Chi tiết ở mục 7.

---

## 2. Nguồn dữ liệu gốc

DF40 gồm 40 method làm giả, mỗi method có 2 họ:

- **cdf (Celeb-DF):** video gốc là các clip mặt người thật từ Celeb-DF
  (`Celeb-real` / `YouTube-real`), bị swap.
- **ff (FaceForensics++):** video gốc là 1,000 video `original_sequences` của FF++.
- Ngoài ra còn các method tổng hợp hoàn toàn (StyleGAN*, VQGAN, ddim…), expression
  (stargan, starganv2, styleclip), và các method không ghép với real (oth).

Ảnh real không có sẵn trong DF40 → lấy riêng: mỗi video real lấy **1 frame giữa**
(`mid-frame`) từ Celeb-DF và FF++. Tổng cộng 178 + 999 = **1,177 ảnh real**, mỗi ảnh
= 1 người duy nhất.

---

## 3. Quy trình xây dựng từng bước

Script: `src/data/build_test_data_v2.py` (scan + copy + manifest) →
`src/data/restructure_test_data_v3.py` (chia folder theo method bằng hard link).

### Bước 1 — Real pool (1,177 ảnh)
Với mỗi video real (178 Celeb-DF + 999 FF++), chọn `mid_frame` = frame giữa của video.
Identity = mã video (`cdc:idN_M` hoặc `ffc:<5 số>`).

### Bước 2 — Enumeration fake theo 40 method
Với mỗi method, đọc cấu trúc thư mục trên ổ Air rồi chọn **1 frame đại diện** mỗi
`(method, identity)` (không lấy nhiều frame cùng người cùng method → tránh trùng):

| Nhóm method | Cấu trúc | Quy tắc identity |
|---|---|---|
| 20 method cdf chuẩn | `cdf/frames/<idN_M>/` | identity = người bị mạo danh (`id16_0003`), frame ghép = frame real cùng chỉ số |
| 18 method ff chuẩn | `ff/frames/<A_B>/` | identity = người bị mạo danh (`A`), frame ghép theo chỉ số |
| DiT/RDDM/SiT | `cdf/Fake_from_{Celeb,YouTube}-real/<id>/` | trích id từ tên thư mục |
| StyleGAN3/StyleGANXL | frame `seed*.png` | **mặt tổng hợp** → domain `efs`, KHÔNG ghép real |
| pixart/sd2.1 | frame `Celeb-real_id13_0011_045.png` | trích id từ **tên frame** |
| e4e | `e4e/e4e/ff/<id>/`, frame `00001.jpg` | identity = id (ghép FF++) |
| mobileswap | `frames.zip` (giải nén local) | cdf `idN_M` + ff `A_B` chuẩn |
| stargan/starganv2/styleclip | `fe/` | domain `fe` |

### Bước 3 — Ghép real ↔ fake (PAIRED)
Với mỗi identity có real, tìm fake của chính người đó từ các method và chọn frame
có **cùng chỉ số frame** với real (matched frame). Kết quả: **1,177 identity paired,
7,454 fake paired** — mọi ảnh real đều có ít nhất 1 fake cùng người để test cặp.

### Bước 4 — Phân chia train/test (identity-disjoint)
- Gom ảnh theo cột `identity` thành 23,237 nhóm.
- Shuffle theo seed 42, lấy 70% nhóm → train (16,265 identity / 21,459 ảnh),
  30% → test (6,972 identity / 9,232 ảnh).
- **Toàn bộ real + fake của cùng 1 người luôn nằm cùng nhánh.**
- Model **không bao giờ nhìn thấy người đó trong train** khi test.

### Bước 5 — Sửa bug trùng path
Bản đầu dùng path phẳng `{identity}__{frame}.png` → bị **đè file** khi 2 method cùng
người cùng chọn đúng matched-frame (403 ảnh trùng). Đã sửa: path thêm tiền tố method
`{method}__{identity}__{frame}` → **path trùng = 0** (đã verify). Bug này từng làm số
eval bản 31-method sai (real cdc tụt 0.53); sau fix phục hồi đúng.

### Bước 6 — Tái cấu trúc v3
`test_data_v3/` = `real/` (1,177 ảnh dùng chung) + `<method>/fake/` (40 method).
Dùng **hard link** (cùng inode với v2, không tốn thêm ổ đĩa). Real để riêng top-level
vì real dùng **chung cho mọi method** — nhân bản vào từng method sẽ làm real bị tính
40 lần khi eval.

---

## 4. Protocol đánh giá

- **Probe:** backbone đông cứng (DINOv3 ViT-S/16 hoặc ConvNeXt-Tiny, tự-supervised),
  trích feature CLS 384-d / 768-d → LogisticRegression linear probe
  (`class_weight=balanced`).
- **Metric:** Acc / Prec / Rec / F1 / AUC, real acc, fake detection, theo domain,
  theo method, **paired-only** (chỉ identity có cả real lẫn fake — test nghiêm ngặt nhất).
- **Tái lập:** `src/eval/eval_identity_disjoint.py --root test_data_v3 --tag test_data_v3`.

---

## 5. Trả lời: CÓ LEAK KHÔNG?

### ✅ Đã chặn được (không leak)
1. **Không rò rỉ identity (người):** split theo `identity` — người bị làm giả trong
   test **không từng xuất hiện** (kể cả ảnh real) trong train.
2. **Không ảnh lặp:** mỗi ảnh xuất hiện đúng 1 lần trong toàn bộ dataset
   (identity-unique). Không có frame nào vừa ở train vừa ở test.
3. **Không lẫn index:** eval assert cache feature khớp label manifest trước khi fit.
4. **Paired-only test tách riêng:** khi test cặp real↔fake, frame fake là **cùng cảnh**
   với real (cùng chỉ số frame) → model không thể "ăn gian" bằng khác biệt bối cảnh,
   phải nhận ra thao tác làm giả thật sự.
5. **Bug trùng path đã sửa và verify** (path trùng = 0).

### ⚠️ Giới hạn quan trọng (đọc kỹ — KHÔNG phải leak, nhưng là phạm vi của protocol)
1. **Đây KHÔNG phải zero-shot với method mới.** Train và test đều gồm ảnh từ **cùng
   40 method, cùng nguồn dataset** (khác người). Model được huấn luyện feature trên
   các method này nên có thể học "chữ ký method". Số per-method đo khả năng tổng quát
   hóa **trong cùng method / cùng dataset** với người mới — không phải "gặp method lạ
   chưa từng thấy".
2. **Shortcut domain (method-signature):** probe có thể nhận fake bằng bề ngoài đặc
   trưng của từng method thay vì "hiểu" thao tác làm giả. Điều này **làm giảm độ khó**
   so với thực tế — nói cách khác, con số hiện tại có phần **lạc quan** cho ứng dụng
   thực (deepfake mới ra đời không có trong train).
3. **Ảnh real từ cùng dataset làm giả:** real lấy từ chính Celeb-DF/FF++ — nơi sinh ra
   fake. Điều này giúp ghép cặp chuẩn, nhưng real-world thì ảnh thật không cùng
   "phân phối camera" với ảnh trong dataset.

### 🔴 Không thể kết luận từ bộ test này
- Độ mạnh trước **method mới xuất hiện sau 2023** (không nằm trong DF40).
- Khả năng xử lý **video** (protocol hiện chỉ dùng **1 frame** — không có rò rỉ thời
  gian vì chỉ có 1 frame, nhưng cũng không tận dụng được chuyển động/môi trường).
- Khả năng chống **nén lại / upscale / screenshot** (ảnh giữ nguyên chất lượng gốc).

---

## 6. Bằng chứng đáng chú ý — ffc tụt (minh chứng protocol nghiêm túc)

Bảng tách theo (method, domain) cho thấy **FF++ paired-fake (domain `ffc`) tụt mạnh**
trong khi cùng method trên `cdc`/`oth` ≈ 1.0:

| Method | cdc | ffc (ViT / CNN) | oth |
|---|---|---|---|
| faceswap | 0.95 | **0.23 / 0.42** | 1.00 |
| facedancer | 0.84 | **0.30 / 0.37** | 0.98 |
| inswap | 0.93 | **0.42 / 0.53** | 0.96 |
| mobileswap | 1.00 | **0.75 / 0.62** (n=259) | 1.00 |

Đây là kịch bản khó nhất: người bị mạo danh **không xuất hiện trong train**
(identity-disjoint) và fake ghép **cùng cảnh** với real → model phải phát hiện thao
tác swap mà không có ngữ cảnh người đó. Con số này là bằng chứng protocol *không* bị
lỏng, và là phát hiện đáng viết trong đồ án.

---

## 7. Kết luận

- **Không leak**: không identity trùng, không ảnh lặp, split có seed, paired-frame
  cùng cảnh, bug trùng path đã sửa và verify.
- **Phạm vi**: kết quả đúng cho bài toán "nhận deepfake của 40 method DF40, trên
  người chưa từng thấy, trong cùng dataset" — đây là benchmark chuẩn của DF40.
- **Để khẳng định cho thế giới thực**, cần thêm: (1) test chéo dataset khác
  (VD: FaceShifter/DeeperForensics riêng), (2) method không nằm trong train
  (leave-one-method-out), (3) đánh giá trên video. Có thể là hướng mở rộng của đồ án.

---

## 8. Các file liên quan

| File | Vai trò |
|---|---|
| `src/data/build_test_data_v2.py` | scan + chọn frame + copy + manifest (nguồn gốc) |
| `src/data/restructure_test_data_v3.py` | chia v2 → v3 theo method (hard link) |
| `src/eval/eval_identity_disjoint.py` | protocol identity-disjoint + LR probe |
| `test_data_v3/manifest.csv` | 30,691 dòng: `method, video, path, identity, domain, label` |
| `experiments/results/eval/identity_disjoint_v3_{vit,cnn}.json` | kết quả eval |
| `experiments/results/eval/report_40_methods_v3.md` | bảng kết quả 40 method |
| `test_data_v3.zip` | bản đóng gói để giải nén/reproduce |

*Đọc dự án này 1 lần để nắm ngữ cảnh*

session 1: Thực hiện thao tác với dữ liệu
1. Thống kê và dữ liệu
- (1) số lượng data gốc DF40, mô tả thành cấu thành DF40 (nghĩa là từ real của FF++ và Celeb v2 - thống kê cả cái này), real fake ra sao.
- (2) Chia method của DF40 thành các nhóm deepFake data dễ so sánh.
- (3) EDA, toàn bộ gạch đầu hàng (1), mô tả size, pixel, biểu đồ hist, cột, grid visual
- Vấn đề của bộ data là gì? - theo tôi là data frames by frames có số lượng lớn nhưng dễ bị trùng identity - trùng identity thì cũng ko phải vấn đề lớn nếu data video nhưng mà ảnh đơn và muốn tổng quát hoá thì nên làm identity độc nhất.
- (4) Cách chia tập train, val, test - làm lại (1) (2) (3) với các tập đã chia (thông tin data lấy ở các file liên quan - tôi đã huấn luyện finetune 2 model mạnh nhất là DINO viT và DINO ConNext)
- Tôi nghĩ là data sẽ leak nhẹ nhưng ổn - quan trọng là train ko leak với test 

*Notice*
Cách thức: sinh hình ảnh và viết và .md kém hình. Viết có trình tự. 

---

## session 2: EDA lại với train/test mới + đánh giá "A1 có tác dụng thật không?"

### 1. Train finetune mới (`data/splits/finetune_plus_train.csv`) — 123,582 ảnh
- **Fake 94,025 ảnh (label 1) từ 42 method** | **Real 29,557 ảnh (label 0) từ 9 nguồn**:
  FF++ Real 12,515 · Celeb-DF Real 5,739 · ffhq_real (kaggle) 4,655 · celebvhq 3,800 · CollabDiff Real 690 · whichfaceisreal Real 683 · MidJourney Real 669 · DF40 Real 457 · starganv2 Real 349.
- **8 method "yếu" là mục tiêu A1** (pool ảnh): faceswap 11,953 · deepfake_faceswap 7,672 · facedancer 4,386 · sadtalker 4,383 · fsgan 3,401 · inswap 2,932 · wav2lip 2,927 · mobileswap 1,980.
- 34 method còn lại: 71,433 ảnh, phân phối đuôi dài (mỗi method ~350–5,739).
- Sampler mỗi epoch = 2×real = **59,114 ảnh**. Đây là điểm mấu chốt: cách chia slot quyết định mỗi ảnh được nhìn bao nhiêu lần/epoch.
- **Exposure A0 (sampler faceswap)** — faceswap 0.35 / real 0.35 / "other" 0.30 **chia đều theo method** (~433 slot/method):
  faceswap ~1.73× · real 0.70× · nhưng method yếu bị **đói**: deepfake_faceswap **0.056×**, facedancer 0.099×, sadtalker 0.099×, fsgan 0.127×, wav2lip 0.148×.
- **Exposure A1 (sampler weak_family)** — real 0.35 / **boost 0.45** cho 8 method yếu (theo pool, faceswap weight 2.0) / other 0.20:
  8 method yếu đều ~**0.52×** (nâng 3–9×), faceswap hạ còn 1.03×, real giữ 0.70×.

### 2. Val + Test mới
- **Val 6,302** (identity-disjoint, `finetune_plus_val.csv`): real 1,449 (9 nguồn như train) · fake 4,853.
- **Test cân bằng mới 21,446** (zero-leakage, `data/deepfake_test_suite_full_50k/`): **real 10,723** (ff++_real 1,728 · ffhq_real 8,787 · test_data_v3 208) / **fake 10,723** từ 38 method, hầu hết 300 ảnh/method (deepfacelab 25, heygen 11, MidJourney 187 là ngoại lệ).

### 3. Model trên test cân bằng 21,446 (fp32 MPS)
| model | acc% | prec | rec | f1 | AUC | real_acc | FP | FN |
|---|---|---|---|---|---|---|---|---|
| Plus **A0** (sampler cũ) | 97.91 | 98.05 | 97.77 | 97.91 | 0.9979 | 0.9805 | 209 | 239 |
| Plus **A1** (weak_family) | **98.47** | 97.94 | **99.02** | **98.48** | **0.9986** | 0.9792 | 223 | 105 |
| ConvNeXt (best, tham chiếu) | 99.22 | 99.79 | 98.64 | 99.21 | 0.9998 | 0.9979 | 22 | 146 |

### 4. A1 có tác dụng thật không? → **CÓ**, kiểm định cặp McNemar (cùng 21,446 ảnh)
- **Toàn test:** A0 đúng–A1 sai = 120 · A0 sai–A1 đúng = 240 → **χ²=39.3, p=3.6e-10** → cải thiện có ý nghĩa thống kê rất mạnh.
- **Nhóm FAKE (10,723):** 32 vs 166 → **p=3e-21** (A1 phát hiện fake tốt hơn hẳn).
- **Nhóm REAL (10,723):** 88 vs 74 → **p=0.31** → **không đủ bằng chứng real tụt** (98.05→97.92 chỉ là nhiễu). Điểm lo real_acc giảm −1.4 trên **val** không lan sang **test** → không phải overfit cục bộ mà do phân phối real trong val (FF++ frame) khó hơn.
- **Per-method (fake, 300/method)** — cải thiện có nghĩa (p<0.05):
  deepfake_faceswap **+61 ảnh** (p=1.6e-14) · wav2lip +21 (p=1.3e-5) · sadtalker +13 (p=0.003) · fsgan +9 (p=0.008) · inswap +9 (p=0.008) · faceswap +7 (p=0.023) · mobileswap +6 (p=0.041). facedancer +5 nhưng p=0.18 (mẫu chưa đủ).
- Không method nào tụt đáng kể (MRAA/stargan −4, p=0.13 — nhiễu). Một vài p đơn lẻ ~0.02–0.04 sẽ không qua Bonferroni chặt, nhưng **mẫu hình đồng nhất đúng 8 method được tăng exposure + McNemar tổng cực mạnh** ⇒ hiệu quả thật, đúng mục tiêu.
- Hình minh họa: `experiments/results/coursework_vs/eda_a0_vs_a1.png`

### 5. Cơ chế + kết luận
- A0 chia "other" **đều theo method** làm method có pool lớn (deepfake_faceswap 7,672 ảnh) chỉ được ~0.056×/ảnh/epoch → model hầu như không thấy → test det chỉ **77%**. A1 dành 45% epoch cho nhóm yếu → **0.52×** → test **77→97.3%**.
- A1: **+0.56 acc, −134 FN**, real giữ nguyên (không sig tụt) → **bước đi hợp lệ, hiệu quả thật, không overfit cục bộ**.
- Vẫn còn thua ConvNeXt (99.22): gap chính ở **real** (FP 223 vs 22) + vài method. Bước kế hoạch: **A2 = KD từ ConvNeXt** (sampler A0 + KD, cô lập biến) — code sẵn trong `scripts/finetune_plus_v3.py` (`--kd-teacher`), **chưa chạy** (chờ máy rảnh/reboot). A3 (gộp A1+A2) chỉ làm nếu A2 tự nó giúp.

*Notice session 2*
- Background task của Claude bị kill 2 lần giữa train → chạy training **detached** (`subprocess.Popen(start_new_session=True)`) + `--resume` từ `resume.pt` (lưu mỗi 800 steps). Đã sửa bug `resume_skip` (trộn batch/sample).
- Eval chuẩn mới: `scripts/eval_coursework_vs.py --tags Plus_viT_v3 plus_v3_s1 --extra-model plus_v3_s1=outputs/finetune/plus_v3_s1_best.pt` (thêm `--extra-model TAG=CKPT` cho model stage). Preds cặp lưu `.npz` → McNemar được.

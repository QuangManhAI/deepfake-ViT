# 3 Bộ Test Data Chuẩn: Ảnh / Video / Audio

*Ngày: 2026-06. Mục đích: test set uy tín cho từng modality, được paper hay dùng, kèm link tải. Mức xác minh: link và số liệu lấy trực tiếp từ README/trang chính thức.*

---

## 1. Bảng tổng: chọn nhanh

| Modality | Chính (chuẩn nhất) | Phụ (đa dạng / hiện đại) | Tải thẳng, không cần form? |
|---|---|---|---|
| **Ảnh (face forgery)** | **FF++ frames** c23/c40 [1901.08971] | **DF40** [2406.13495] (40 kỹ thuật), **GenImage** [2306.08571] (AI-gen ảnh thường), **iFakeFaceDB** [1911.05351] (StyleGAN anti-forensic), **Deepfake-Eval-2024 ảnh** [2503.02857] | GenImage ✅ (Drive); iFakeFaceDB ✅ (GitHub); FF++/DF40 cần form |
| **Video** | **FF++** c23/c40 [1901.08971] + **Celeb-DF v2** [1909.12962] (cross-dataset) | **DFDC subset** [2006.07397] (in-the-wild), **Deepfake-Eval-2024** [2503.02857] (thực tế 2024) | Deepfake-Eval-2024 ✅ (HF); DFDC ✅ (Kaggle); FF++/Celeb-DF cần form |
| **Audio** | **ASVspoof 2019 LA** [1911.01601] | **WaveFake** [2111.02813], **In-the-Wild** [2203.16263], **MLAAD** [2401.09512] | WaveFake ✅ (Zenodo), MLAAD ✅ (HF), In-the-Wild ✅ (HF); ASVspoof cần đăng ký licence |

---

## 2. Chi tiết + link tải

### 🖼️ A. Deepfake IMAGES

**A1. FaceForensics++ — frames (c23/c40/raw)** — *benchmark kinh điển cho ảnh mặt giả mạo (frame-level)*
- Nội dung: 1.000 video thật × 4 phương pháp (DeepFakes, Face2Face, FaceSwap, NeuralTextures) = 4.000 video fake → test frame-level ảnh mặt (đây là "image test" mà hầu hết paper dùng); kèm thêm FaceShifter + Google DFD.
- Link tải: **Google Form** → https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform (được duyệt → nhận script download, chọn EU/EU2 server)
- Repo: https://github.com/ondyari/FaceForensics

**A2. DF40** (NeurIPS 2024 D&B) — *test hiện đại nhất cho ảnh/video mặt: 40 kỹ thuật*
- Nội dung: 40 kỹ thuật = 10 face-swap + 12 face-reenactment + 10 entire-face-synthesis + 5 face-editing; gồm cả ảnh lẫn video; 2.000+ evaluation trong paper.
- Link tải: **Google Form** → https://docs.google.com/forms/d/1ESAWoWusOEGEEVnXCH_emv-wJqCYMhCbD6-85RMIoDk/edit
- Repo: https://github.com/YZY-stack/DF40 ; Paper: https://arxiv.org/abs/2406.13495

**A3. GenImage** (NeurIPS 2023 D&B) — *chuẩn cho ảnh AI-generated nói chung (không chỉ mặt)*
- Nội dung: ~1,33 triệu ảnh, 8 generator (ADM, BigGAN, GLIDE, Midjourney, SD v1.4, SD v1.5, VQDM, Wukong); real = ImageNet.
- Link tải: **Google Drive (tải thẳng, không cần form)** → https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS?usp=sharing
- Repo: https://github.com/GenImage-Dataset/GenImage ; Paper: https://arxiv.org/abs/2306.08571

### 🎞️ B. Deepfake VIDEOS

**B1. FaceForensics++ (video)** — *chuẩn in-domain: train c23 → test c23/c40/raw*
- Như A1 (cùng dataset, dùng ở mức video: mean frame score / majority vote).
- Link: form ở A1. Protocol chuẩn: chia 720 train / 140 val / 140 test video.

**B2. Celeb-DF v2** (CVPR 2020) — *chuẩn CROSS-DATASET (test độ tổng quát)*
- Nội dung: 590 real (Celeb-real) + 300 YouTube-real + 5.639 fake; **test list chính thức = 518 video** (`List_of_testing_videos.txt`).
- Link tải: **Google Form** → https://forms.gle/2jYBby6y1FBU3u6q9 (hoặc Tencent form https://wj.qq.com/s2/8540155/b5d9/) — link gửi qua email sau khi duyệt.
- Repo: https://github.com/yuezunli/celeb-deepfakeforensics

**B3. DFDC** (Kaggle) — *in-the-wild: audio + video, 8 kỹ thuật*
- Nội dung: 128.154 clip (23.654 real / 104.500 fake), diễn viên có đồng thuận.
- Link: https://www.kaggle.com/competitions/deepfake-detection-challenge/data (đăng nhập Kaggle + chấp nhận điều khoản; ~470GB bản đầy đủ — có thể tải subset).

**B4. Deepfake-Eval-2024** — *in-the-wild thực tế 2024 (video+audio+ảnh)*
- Nội dung: 44h video, 56,5h audio, 1.975 ảnh; 88 website, 52 ngôn ngữ; nhãn thủ công.
- Link: **Hugging Face (tải thẳng)** → https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024
- Paper: https://arxiv.org/abs/2503.02857

### 🔊 C. Deepfake AUDIO

**C1. ASVspoof 2019 LA (Logical Access)** — *benchmark kinh điển số 1 cho synthetic speech (TTS + voice conversion)*
- Nội dung: 3 tập con (LA/PA/DF); **LA** = 100% tấn công TTS + VC, train/dev/eval tách biệt có protocol keys → số của bạn so được trực tiếp với hàng trăm paper.
- Link tải: DataShare Edinburgh → https://datashare.ed.ac.uk/handle/10283/3336 (DOI 10.7488/ds/2555); phải đồng ý licence. Trang chính thức: https://www.asvspoof.org/database
- Paper: https://arxiv.org/abs/1911.01601

**C2. WaveFake** (NeurIPS 2021 D&B) — *phát hiện giọng GAN/neural vocoder*
- Nội dung: 11 vocoder (MelGAN, HiFiGAN, WaveGlow, Parallel WaveGAN, Multi-band/Full-band MelGAN...), ~117.000 mẫu (LJSpeech + JSUT).
- Link tải: **Zenodo (tải thẳng)** → https://zenodo.org/records/5642694
- Repo: https://github.com/RUB-SysSec/WaveFake ; Paper: https://arxiv.org/abs/2111.02813

**C3. In-the-Wild** — *audio deepfake ngoài đời thực (chính trị gia, người nổi tiếng)*
- Nội dung: 58 nhân vật, 20,8h bona-fide + 17,2h spoofed (từ YouTube/social); benchmark chuẩn để test generalization ngoài lab.
- Link tải: **Hugging Face (tải thẳng)** → https://huggingface.co/datasets/mueller91/In-The-Wild (~38GB; có bản Kaggle: https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-dataset)
- Paper: https://arxiv.org/abs/2203.16263

**C4. MLAAD** — *đa ngôn ngữ (11 ngôn ngữ), TTS/VC*
- Link tải: **Hugging Face (tải thẳng)** → https://huggingface.co/datasets/mueller91/MLAAD (CC-BY-NC-4.0)
- Paper: https://arxiv.org/abs/2401.09512

---

## 3. Bộ tối giản đề xuất cho project của bạn (3 test set)

| Modality | Test set | Lý do |
|---|---|---|
| Ảnh | **FF++ frames c23/c40** (+ DF40 nếu muốn 40 kỹ thuật) | Cùng pipeline align mặt sẵn có |
| Video | **FF++ c23/c40** (in-domain) + **Celeb-DF v2** (cross, 518 video test) | Đúng protocol các paper; so được số |
| Audio | **ASVspoof 2019 LA** (chuẩn so sánh) + **In-the-Wild** (thực tế) | LA để so với paper; ITW để test thật |

## 4. Lưu ý khi dùng làm TEST (tránh số ảo)

1. **Không train trên test set** — các tập trên đều có protocol riêng: FF++ (720/140/140), Celeb-DF (test list 518), ASVspoof (train/dev/eval + protocol keys), DF40 (challenge set riêng).
2. **Identity-disjoint** với tập train của bạn (đặc biệt FakeAVCeleb/VoxCeleb2 nếu dùng chung nguồn).
3. **Audio**: báo metric theo EER (Equal Error Rate) là chuẩn của cộng đồng ASVspoof, không chỉ ACC/AUC.
4. **Deepfake-Eval-2024** không có split sẵn → tự chia (khuyến nghị theo nguồn video để tránh rò rỉ).
5. DFDC đầy đủ ~470GB — tải subset cân bằng real/fake là đủ cho test (paper hay dùng 1.000–18.000 clip).

## 5. IMAGE-ONLY — danh sách đầy đủ (test bằng ảnh thuần)

| # | Dataset | Nội dung | Loại ảnh | Link tải | Form? |
|---|---|---|---|---|---|
| 1 | **FF++ frames** c23/c40/raw [1901.08971] | 1.000 thật + 4.000 fake × 4 phương pháp (mặt); cũng là nguồn ảnh frame-level chuẩn nhất | Ảnh mặt (face forgery) | form: https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform | ✅ chờ duyệt |
| 2 | **Celeb-DF v2 frames** [1909.12962] | 590 real + 5.639 fake (mặt); trích frame để test ảnh cross-dataset; test list 518 video | Ảnh mặt (face forgery) | form: https://forms.gle/2jYBby6y1FBU3u6q9 | ✅ chờ duyệt |
| 3 | **DF40 (ảnh)** [2406.13495] | 40 kỹ thuật (10 FS + 12 FR + 10 EFS + 5 FE), ảnh + video; NeurIPS 2024 | Ảnh mặt (hiện đại) | form: https://docs.google.com/forms/d/1ESAWoWusOEGEEVnXCH_emv-wJqCYMhCbD6-85RMIoDk/edit | ✅ chờ duyệt |
| 4 | **GenImage** [2306.08571] | ~1,33 triệu ảnh, 8 generator (ADM, BigGAN, GLIDE, Midjourney, SD1.4, SD1.5, VQDM, Wukong); real = ImageNet | Ảnh AI-gen nói chung (không chỉ mặt) | **tải thẳng**: https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS?usp=sharing | ❌ không |
| 5 | **iFakeFaceDB** [1911.05351] | ~87.000 ảnh mặt StyleGAN đã qua GANprintR (xóa fingerprint GAN → khó hơn, chống "ăn gian theo artifact") | Ảnh mặt GAN (anti-forensic) | **tải thẳng**: https://github.com/socialabubi/iFakeFaceDB | ❌ không |
| 6 | **Deepfake-Eval-2024 (ảnh)** [2503.02857] | 1.975 ảnh in-the-wild 2024 (lipsync, faceswap, diffusion), 52 ngôn ngữ | Ảnh thực tế 2024 | **tải thẳng**: https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024 | ❌ không |
| 7 | **UniversalFakeDetect** [2302.10174] | ~20 generator (StyleGAN2/3, ProGAN, DALL·E, Midjourney, SD, GLIDE, ADM...); test độ tổng quát chéo generator | Ảnh GAN + diffusion | script tải: https://github.com/WisconsinAIVision/UniversalFakeDetect (không có file gộp sẵn) | ❌ script |

**Lưu ý:** #1, #2 vốn là video nhưng **paper vẫn test frame-level** (trích frame → cắt mặt → predict) — đây là cách "ảnh deepfake mặt" được test phổ biến nhất. #3–#7 là ảnh thuần. Khi báo cáo: ghi rõ bạn test ảnh mặt (FF++/Celeb-DF/DF40) hay ảnh AI-gen nói chung (GenImage/UniversalFakeDetect) — hai bài toán khác nhau.

## 6. VIDEO TEST NHỎ — chạy nhanh, vẫn chuẩn paper (≤ ~11GB)

| Bộ | Số video | Dung lượng | Loại | Link tải | Form? |
|---|---|---|---|---|---|
| **UADFV** (ICASSP 2019) | 98 (49 thật + 49 fake) | **~155MB** | GAN faceswap, test nhanh (sanity check) | Drive: https://drive.google.com/drive/u/0/folders/1GEk1DSxmlV_61JtpEGzC9Fo_BffvyxpH ; Kaggle mirror: https://www.kaggle.com/datasets/anupriyakkumari/uadfv-new | ❌ tải thẳng |
| **DeepFake-TIMIT (DF-TIMIT)** | 960 (320 thật + 640 fake: LQ 64×64 + HQ 128×128) | **~0,2GB** | Faceswap (32 người, VidTIMIT); dùng làm test cross trong nhiều paper audio-visual | **Zenodo (tải thẳng)**: https://zenodo.org/records/4068245 ; trang chủ: https://www.idiap.ch/en/scientific-research/data/deepfaketimit | ❌ tải thẳng |
| **Celeb-DF v2** [1909.12962] | 6.229 (590 thật + 5.639 fake); test list 518 video | **10,16GB** (Kaggle mirror) | Chuẩn cross-dataset | Form: https://forms.gle/2jYBby6y1FBU3u6q9 ; **Kaggle mirror (tải thẳng)**: https://www.kaggle.com/datasets/reubensuju/celeb-df-v2 | ⏳/❌ |
| **DFDC Preview** [1910.08854] | 5.214 clip (2 kỹ thuật, diễn viên có đồng thuận) | chưa xác minh (cỡ ~10GB) | In-the-wild nhỏ (thay vì DFDC 470GB) | Meta: https://ai.meta.com/datasets/dfdc/ | ⏳ chờ duyệt |

**Đề xuất chạy nhanh (1 buổi):** UADFV (sanity, 155MB) → Celeb-DF v2 518 video test (10GB, số so được với paper) → thêm DF-TIMIT (Zenodo) nếu muốn bộ thứ 3 nhỏ. Muốn in-the-wild mà không tải 470GB thì dùng **DFDC Preview**.

---

## Nguồn (đã đọc trực tiếp README/trang chính thức)

- FF++: https://github.com/ondyari/FaceForensics (form + server EU) [Đã đọc]
- Celeb-DF v2: https://github.com/yuezunli/celeb-deepfakeforensics (form + cấu trúc + test list 518) [Đã đọc]
- Deepfake-Eval-2024: https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024 (44h/56,5h/1.975 ảnh) [Đã đọc]
- GenImage: https://github.com/GenImage-Dataset/GenImage (Drive link) [Đã đọc]
- DF40: https://github.com/YZY-stack/DF40 (form) [Đã đọc]
- ASVspoof 2019: https://datashare.ed.ac.uk/handle/10283/3336 + https://www.asvspoof.org/database [Đã đọc]
- WaveFake: https://zenodo.org/records/5642694 + https://github.com/RUB-SysSec/WaveFake [Đã đọc]
- In-the-Wild: https://huggingface.co/datasets/mueller91/In-The-Wild [Đã đọc]
- MLAAD: https://huggingface.co/datasets/mueller91/MLAAD [Đã đọc]
- iFakeFaceDB: https://github.com/socialabubi/iFakeFaceDB (~87.000 ảnh StyleGAN + GANprintR) [Đã đọc]
- UniversalFakeDetect: https://github.com/WisconsinAIVision/UniversalFakeDetect + https://arxiv.org/abs/2302.10174 [Đã đọc]
- UADFV: Drive https://drive.google.com/drive/u/0/folders/1GEk1DSxmlV_61JtpEGzC9Fo_BffvyxpH + Kaggle mirror 154,51MB https://www.kaggle.com/datasets/anupriyakkumari/uadfv-new [Đã đọc]
- DeepFake-TIMIT: Zenodo https://zenodo.org/records/4068245 + https://www.idiap.ch/en/scientific-research/data/deepfaketimit [Đã đọc]
- DFDC Preview: arXiv 1910.08854 + https://ai.meta.com/datasets/dfdc/ [Đã đọc]
- Celeb-DF v2 Kaggle mirror (10,16GB): https://www.kaggle.com/datasets/reubensuju/celeb-df-v2 [Đã đọc]

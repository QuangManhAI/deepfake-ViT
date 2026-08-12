# Danh mục Dataset Deepfake — Ảnh / Video / Audio / Audio-Visual (bản khảo sát)

> Mục đích: **tài liệu giao cho nhóm khảo sát & đánh giá model**. Liệt kê toàn bộ dataset tìm được (đã lọc bớt các bộ không tải được hoặc mờ ám), phân theo modality, kèm link tải và mức xác minh.
> Quy tắc dùng: **không tin số trên báo — tự chạy, tự test** (xem §H). Số liệu trong bảng là *mô tả dataset*, không phải kết quả model.

**Chú thích mức xác minh:**
- **[Đã xác minh]** = tôi đã đọc trực tiếp README/paper/supplement hoặc đo kiểm chứng (link, số liệu, cấu trúc).
- **[Web]** = tìm thấy qua tìm kiếm web với ≥2 nguồn khớp (ID arXiv đã kiểm tra, chưa đọc trực tiếp toàn văn).
- **[Tham khảo]** = có thật nhưng thông tin chi tiết còn thiếu, cần người khảo sát kiểm lại.

---

## 0. Bảng tổng quan nhanh
| Dataset | Modality | Quy mô | Dung lượng | Link chính | Ghi chú |
|---|---|---|---|---|---|
| **FF++** | Video | 1k real + 4k fake | ~13GB (c23)/30GB+ (RAW) | form (repo `ondyari/FaceForensics`) | **Không có audio** |
| **Celeb-DF v2** | Video | 5.639 fake + 890 real | ~10GB (Kaggle mirror) | form / Kaggle | Test list chuẩn 518 video |
| **DFDC** | Video (+audio) | 128.154 clip | ~470GB | Meta / Kaggle | Dataset AV chuẩn in-the-wild |
| **DFDC Preview** | Video | 5.214 clip | nhỏ | Meta | 2 kỹ thuật, test nhanh |
| **DeeperForensics-1.0** | Video | 60.000 video / 17,6M frame | lớn (đa phần) | GitHub `EndlessSora/DeeperForensics-1.0` | 100 diễn viên, 26 quốc gia |
| **KoDF** | Video (+audio thật) | 237.942 video | lớn | GitHub `deepbrainai-research/kodf` | Người Hàn, audio **không** fake |
| **ForgeryNet** | Ảnh+Video | 2,9M ảnh + 221.247 video | rất lớn | yinanhe.github.io/projects/forgerynet | 15 kỹ thuật, 4 task (có localization) |
| **WildDeepfake** | Video | 7.314 chuỗi face (707 video) | ~chục GB | GitHub `OpenTAI/wild-deepfake` | Video thu từ internet |
| **UADFV** | Video | 98 video (49/49) | **~155MB** | Drive / Kaggle | Test nhanh, 1 kỹ thuật |
| **DF-TIMIT** | Video | 960 video | **~0,2GB** | Zenodo 4068245 | LQ/HQ 64/128px |
| **DF40** | Ảnh+Video | 40 kỹ thuật | tùy chọn | Google Form | Benchmark hiện đại nhất (2024) |
| **Deepfake-Eval-2024** | AV | 44h video + 56,5h audio + 1.975 ảnh | vừa | HF `nuriachandra/Deepfake-Eval-2024` | 88 website, 52 ngôn ngữ |
| **GenVidBench** | Video | AI video gen | — | arXiv 2501.11340 | Benchmark video sinh (2025) |
| **GenImage** | Ảnh | 1,33M ảnh, 8 generator | ~60GB | Google Drive | Chuẩn AI-gen ảnh (ECCV 2024) |
| **iFakeFaceDB** | Ảnh | ~87.000 face StyleGAN | ~10GB | GitHub `socialabubi/iFakeFaceDB` | Có bản đã gỡ fingerprint GAN |
| **UniversalFakeDetect** | Ảnh | 20 generator | ~20GB | GitHub `WisconsinAIVision/...` | Tải bằng script |
| **DiffusionDB** | Ảnh | 14M ảnh (subset 2M) | 6,5TB (subset 1,6TB) | HF `poloclub/diffusiondb` | SD thật của user; dựng pretrain |
| **ArtiFact** | Ảnh | lớn (~1,3M, 25 kỹ thuật) | lớn | HF `bitmind/ArtiFact` | 13 GAN + 7 diffusion + 5 khác |
| **CIFAKE** | Ảnh | 120.000 (60k real + 60k fake) | **~1GB** | Kaggle `birdy654/...` | Sanity-test AI-gen |
| **ASVspoof 2019 LA** | Audio | train/dev/eval + keys | ~vài GB | DataShare Edinburgh | **Chuẩn audio** (EER) |
| **ASVspoof 2021** | Audio | LA + DF + PA | ~vài GB | Zenodo 4837263/4835108 | Có track deepfake riêng |
| **WaveFake** | Audio | ~117k mẫu, 11 vocoder | **~1GB** | Zenodo 5642694 | Vocoder/synthetic speech |
| **In-the-Wild** | Audio | 58 speaker, 38h | ~vài GB | HF `mueller91/In-The-Wild` | Audio deepfake in-the-wild |
| **MLAAD** | Audio | 11 ngôn ngữ | ~chục GB | HF `mueller91/MLAAD` | Đa ngôn ngữ, CC-BY-NC-4.0 |
| **ADD 2022** | Audio | 3 track (LF/PF/FG) | vừa | addchallenge.cn / Zenodo 12188127 | Partial-fake audio |
| **FoR** | Audio | 195k+ utterance | ~10GB | bil.eecs.yorku.ca/datasets/ | TTS cũ (Deep Voice 3, Wavenet) |
| **FakeAVCeleb** | **AV** | 20.000 video (500 real) | ~chục GB | GitHub `DASH-Lab/FakeAVCeleb` | Chuẩn train AV (NeurIPS 2021) |
| **AV-Deepfake1M** | **AV** | 1.886h, >1M video | ≥254GB (train) | HF `ControlNet/AV-Deepfake1M` (gated) | Localization; **không có face-swap**  |
| **LAV-DF** | **AV** | ~1.500 video | vừa | HF/`ControlNet/LAV-DF` | AV localization (DICTA Best Award) |
| **SWAN-DF** | **AV** | 30 cặp | nhỏ | Idiap | Thử nghiệm nhanh |
| **DigiFakeAV** | **AV** | (2025) | — | arXiv 2505.16512 | [Tham khảo] |

---

## A. VIDEO (face-swap / face-reenactment / tổng hợp)

### A1. FF++ — FaceForensics++ — arXiv **1901.08971** (ICCV 2019) [Đã xác minh]
- **Nội dung:** 1.000 video thật + 4.000 fake từ 4 kỹ thuật: DeepFakes (DF), Face2Face (F2F), FaceSwap (FS), NeuralTextures (NT); 4 mức nén c0 (RAW), c23, c40.
- **Dùng:** benchmark chính; split chuẩn **720/140/140 video**; test robustness nén (c23→c40/RAW).
- ** Không có audio** (README chính thức: *"We only downloaded the source video without audio"*) — chỉ dùng visual.
- **Tải:** form: https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform (server EU/EU2); script `download-FaceForensics.py` trong repo `ondyari/FaceForensics`. License: nghiên cứu.

### A2. Celeb-DF v2 — arXiv **1909.12962** (CVPR 2020) [Đã xác minh]
- **Nội dung:** 590 Celeb-real + 300 YouTube-real + **5.639 fake** (DeepFakes + FSGAN, chất lượng cao); **test list chuẩn = 518 video** (`List_of_testing_videos.txt`).
- **Dùng:** cross-dataset test sau khi train FF++ (đo generalization).
- **Tải:** form https://forms.gle/2jYBby6y1FBU3u6q9 (alt Tencent https://wj.qq.com/s2/8540155/b5d9/) ; **Kaggle mirror tải ngay, ~10,16GB:** https://www.kaggle.com/datasets/reubensuju/celeb-df-v2
- Celeb-DF v1: subset nhỏ hơn, ít dùng (các paper cũ báo trên v1).

### A3. DFDC — DeepFake Detection Challenge — arXiv **2006.07397** (2020) [Web]
- **Nội dung:** 128.154 clip (23.654 thật + 104.500 fake; 8 kỹ thuật; 3.426 diễn viên có đồng thuận); **có audio** (quay thực tế) → dataset AV chuẩn in-the-wild.
- **Dùng:** test generalization; hướng audio-visual (AVFF CVPR 2024 train FakeAVCeleb → test DFDC).
- **Tải:** https://ai.meta.com/datasets/dfdc/ ; Kaggle https://www.kaggle.com/competitions/deepfake-detection-challenge/data (~470GB bản đầy đủ; có subset).

### A4. DFDC Preview — arXiv **1910.08854** (2019) [Web]
- **Nội dung:** 5.214 clip, 2 kỹ thuật, tiền thân DFDC.
- **Tải:** https://ai.meta.com/datasets/dfdc/

### A5. DeeperForensics-1.0 — arXiv **2001.03024** (CVPR 2020) [Web]
- **Nội dung:** 60.000 video / 17,6M frame; 100 diễn viên trả phí, 26 quốc gia; 48.475 video nguồn + 11.000 manipulated; có biến dạng real-world (nén, blur, noise).
- **Tải:** GitHub `EndlessSora/DeeperForensics-1.0` → thư mục `dataset/` chứa link (Google Drive, nhiều phần). License: nghiên cứu phi thương mại.

### A6. KoDF — Korean DeepFake Dataset — arXiv **2103.10094** (ICCV 2021) [Web]
- **Nội dung:** **237.942 video** (62.166 real + 175.776 fake); khuôn mặt Hàn Quốc; **audio là audio thật** (không fake) → chỉ dùng cho visual hoặc ghép AV.
- **Tải:** GitHub `deepbrainai-research/kodf` ; site deepbrainai-research.github.io/kodf/

### A7. ForgeryNet — arXiv **2103.05630** (CVPR 2021) [Web]
- **Nội dung:** 2,9M ảnh + **221.247 video**; 15 kỹ thuật forgery (7 image-level + 8 video-level); 4 task: classification 2/3/n-way, **spatial localization**, **temporal localization**, face retrieval.
- **Dùng:** khi cần task localization hoặc n-way classification (không chỉ binary).
- **Tải:** project page https://yinanhe.github.io/projects/forgerynet (Google Drive tar parts, có MD5); repo `yinanhe/ForgeryNet`. Rất lớn — tải theo set.

### A8. WildDeepfake — arXiv **2101.01456** (ACM MM 2021) [Web]
- **Nội dung:** 7.314 chuỗi face từ 707 video **thu trên internet** (diverse, real-world); 3.809 real + 3.505 fake.
- **Dùng:** test in-the-wild (khác studio datasets).
- **Tải:** GitHub `OpenTAI/wild-deepfake` (Baidu/Google Drive).

### A9. UADFV — (paper đi kèm, không có arXiv độc lập) [Đã xác minh]
- **Nội dung:** 98 video (49 real + 49 fake bằng FakeApp/DeepFakes), 1 kỹ thuật, độ phân giải thấp.
- **Dùng:** **sanity test nhanh nhất** (155MB).
- **Tải:** https://drive.google.com/drive/u/0/folders/1GEk1DSxmlV_61JtpEGzC9Fo_BffvyxpH ; Kaggle mirror `uadfv-new` (154,51MB).

### A10. DeepFake-TIMIT (DF-TIMIT) — (Idiap) [Đã xác minh]
- **Nội dung:** 960 video (320 real + 640 fake); 2 chất lượng LQ (64×64) / HQ (128×128).
- **Dùng:** cross-test nhỏ; **~0,2GB**.
- **Tải:** https://zenodo.org/records/4068245 ; trang chính thức https://www.idiap.ch/en/scientific-research/data/deepfaketimit

### A11. DFD — DeepFakeDetection (Google) [Đã xác minh — nằm trong bộ FF++]
- **Nội dung:** video YouTube thật + fake (1 kỹ thuật DeepFakes nội bộ Google), 363 video.
- **Tải:** cùng form FF++ (thư mục `DeepFakeDetection` trong repo `ondyari/FaceForensics`).

### A12. GenVidBench — arXiv **2501.11340** (2025) [Web]
- **Nội dung:** benchmark cho **video do AI sinh** (text-to-video), nhiều generator hiện đại.
- **Dùng:** khi cần test model trên video generation 2024–2025 (không phải faceswap).
- **Tải:** theo repo/paper (chưa đọc trực tiếp chi tiết) [Tham khảo].

---

## B. ẢNH (face forgery + AI-generated images)

### B1. DF40 — arXiv **2406.13495** (NeurIPS 2024 D&B) [Web]
- **Nội dung:** **40 kỹ thuật tạo giả**: 10 face-swap + 12 face-reenactment + 10 EFS (tổng hợp toàn khuôn mặt) + 5 face-editing; gồm cả generator mới (diffusion).
- **Dùng:** benchmark generalization đa generator hiện đại nhất; có ảnh lẫn video.
- **Tải:** Google Form https://docs.google.com/forms/d/1ESAWoWusOEGEEVnXCH_emv-wJqCYMhCbD6-85RMIoDk/edit ; repo `YZY-stack/DF40`.

### B2. GenImage — arXiv **2306.08571** (ECCV 2024) [Đã xác minh]
- **Nội dung:** **1,33M ảnh** (1M fake + 0,33M real); 8 generator: ADM, BigGAN, GLIDE, Midjourney, Stable Diffusion v1.4/v1.5, VQDM, Wukong; kèm split train/val/test chuẩn.
- **Dùng:** benchmark chuẩn cho AI-generated image detection (không phải faceswap).
- **Tải:** Google Drive https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS ; repo `GenImage-Dataset/GenImage`.

### B3. iFakeFaceDB — arXiv **1911.05351** (GANprintR, 2019) [Web]
- **Nội dung:** ~87.000 ảnh face StyleGAN; **2 phiên bản: có và đã gỡ fingerprint GAN** → test khả năng bắt artifact "không-thuộc-về-phân-phối".
- **Tải:** GitHub `socialabubi/iFakeFaceDB`.

### B4. UniversalFakeDetect — arXiv **2302.10174** (CVPR 2023) [Web]
- **Nội dung:** ~20 generator (GAN + diffusion), ảnh tổng hợp nhiều loại (không chỉ face); dataset gốc từ nhiều nguồn.
- **Tải:** repo `WisconsinAIVision/UniversalFakeDetect` (script download).

### B5. DiffusionDB — arXiv **2210.14896** (IEEE VIS 2023) [Web]
- **Nội dung:** **14M ảnh Stable Diffusion + prompt thật của user** (6,5TB); subset DiffusionDB 2M (2M ảnh / 1,5M prompt / 1,6TB). License CC0/MIT.
- **Dùng:** nguồn pretrain/train cho AI-generated detection quy mô lớn.
- **Tải:** HF https://huggingface.co/datasets/poloclub/diffusiondb

### B6. ArtiFact — arXiv **2302.11970** (ICIP 2023) [Web]
- **Nội dung:** ảnh real + synthetic đa danh mục (human/face, animal, places, vehicles…); **25 kỹ thuật sinh** (13 GAN + 7 diffusion + 5 khác); quy mô lớn (~1,3M, cần xem card HF để chốt số).
- **Tải:** HF https://huggingface.co/datasets/bitmind/ArtiFact ; repo `awsaf49/artifact`.

### B7. CIFAKE — arXiv **2303.14126** (IEEE Access 2024) [Web]
- **Nội dung:** 120.000 ảnh = 60k real (CIFAR-10) + 60k fake (Stable Diffusion v1.4) — **nhỏ, tải nhanh** (~1GB) → sanity test.
- **Tải:** Kaggle https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images ; GitHub `jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images`.

### B8. Ảnh từ dataset video (extract frames)
- FF++ frames, Celeb-DF v2 frames, DFDC frames, DF40 images, Deepfake-Eval-2024 images (1.975 ảnh) — trích bằng ffmpeg theo pipeline cố định (xem §H).

---

## C. AUDIO (synthetic speech / voice cloning)

### C1. ASVspoof 2019 LA — arXiv **1911.01601**; challenge tại https://www.asvspoof.org/database [Đã xác minh]
- **Nội dung:** logical access — TTS/VC attacks (19 kỹ thuật); split **train/dev/eval + keys chính thức**; metric chuẩn **EER**.
- **Dùng:** benchmark audio chuẩn nhất, protocol sẵn để so sánh.
- **Tải:** https://datashare.ed.ac.uk/handle/10283/3336 (DOI 10.7488/ds/2555).

### C2. ASVspoof 2021 (LA + DF + PA) [Web]
- **Nội dung:** thêm **track deepfake (DF)** riêng + logical access (LA) khó hơn 2019 (có coding/transmission) + physical access (PA). License Open Data Commons Attribution.
- **Dùng:** test nâng cao; baseline chính thức gồm RawNet2 (PyTorch).
- **Tải:** LA: https://zenodo.org/records/4837263 ; DF: https://zenodo.org/records/4835108 ; keys: https://www.asvspoof.org/index2021.html ; code: github.com/asvspoof-challenge/2021

### C3. WaveFake — arXiv **2111.02813** (2021) [Đã xác minh]
- **Nội dung:** ~117.000 mẫu audio fake từ **11 vocoder** (WaveGAN, MelGAN, HiFi-GAN, MultiBand, Parallel WaveGAN…) + real (LJSpeech); nhỏ, tải nhanh (~1GB).
- **Dùng:** test theo vocoder (phân tích từng loại artifact).
- **Tải:** https://zenodo.org/records/5642694 ; repo `RUB-SysSec/WaveFake`.

### C4. In-the-Wild — arXiv **2203.16263** (IEEE SPL 2022) [Web]
- **Nội dung:** audio deepfake **in-the-wild**: 58 speaker, 20,8h bona-fide + 17,2h spoofed (11 TTS/VC).
- **Tải:** HF https://huggingface.co/datasets/mueller91/In-The-Wild

### C5. MLAAD — arXiv **2401.09512** (2024) [Web]
- **Nội dung:** **11 ngôn ngữ** (EN, DE, ES, FR, IT, PL, RU, UK, AR, ZH, HI…); synthetic speech đa TTS. License CC-BY-NC-4.0.
- **Dùng:** test đa ngôn ngữ / đa TTS.
- **Tải:** HF https://huggingface.co/datasets/mueller91/MLAAD

### C6. ADD 2022 — arXiv **2202.08433** (IJCAI 2022 challenge) [Web]
- **Nội dung:** 3 track: **LF** (low-quality fake), **PF** (partial fake — fake xen giữa audio thật), **FG** (fake game); tiếng Quan thoại (AISHELL-1/3/4 + TTS/VC).
- **Dùng:** bài toán partial-fake audio (sát thực tế nhất).
- **Tải:** http://addchallenge.cn/downloadADD2022 ; **train+dev mới lên Zenodo:** https://zenodo.org/records/12188127. ADD 2023: http://addchallenge.cn/add2023

### C7. FoR — Fake or Real (York University) [Web]
- **Nội dung:** >195.000 utterance (real + TTS cũ: Deep Voice 3, Google Wavenet…).
- **Tải:** https://bil.eecs.yorku.ca/datasets/ ; Kaggle mirror https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

### C8. Audio từ dataset AV (xem §D)
- FakeAVCeleb (voice-clone SV2TTS), AV-Deepfake1M (VITS/YourTTS), DFDC (audio thật), Deepfake-Eval-2024 (56,5h audio).

---

## D. AUDIO-VISUAL (video + audio — dùng khi cần detect 2 modality cùng lúc)

### D1. FakeAVCeleb — arXiv **2108.05080** (NeurIPS 2021 D&B) [Đã xác minh]
- **Nội dung:** **20.000 video** = 500 real + 19.500 fake **4 loại**: face-swap (DeepFakes/FSGAN), voice-clone (SV2TTS), face-swap+voice-clone, real; **nhãn fine-grained** → test được từng loại tấn công.
- **Dùng:** **dataset train chuẩn cho hướng audio-visual** (AVFF CVPR 2024 train trên nó).
- **Tải:** repo `DASH-Lab/FakeAVCeleb` (Google Form → script download, duyệt thủ công 1–2 tuần).

### D2. AV-Deepfake1M — arXiv **2311.15308** (ACM MM 2024 Best Award) [Web]
- **Nội dung:** quy mô lớn nhất: **1.886 giờ, >1M video, 2.068 chủ thể**; visual = **TalkLip** (lip-sync), audio = **VITS** (identity-dependent) + **YourTTS** (identity-independent); 3 loại (FA+FV / FA+RV / RA+FV); có fake xen real → thiết kế cho **localization**.
- ** Không có face-swap** → model chỉ học artifact lip-sync sẽ mù với faceswap. Có imbalance real/fake (tác giả thừa nhận).
- **Tải:** HF gated https://huggingface.co/datasets/ControlNet/AV-Deepfake1M (train ≥254GB, 254 file zip × 1GB); repo `ControlNet/AV-Deepfake1M`.

### D3. LAV-DF — arXiv **2204.06228** (DICTA 2022 Best Award; bản mở rộng "Glitch in the Matrix" arXiv **2305.01979**) [Web]
- **Nội dung:** dataset AV đầu tiên cho **temporal forgery localization** (video có đoạn fake xen real, audio có fake xen thật); ~1.500 video (xem README để chốt số).
- **Tải:** HF https://huggingface.co/datasets/ControlNet/LAV-DF ; repo `ControlNet/LAV-DF`.

### D4. SWAN-DF (Idiap) [Web]
- **Nội dung:** 30 cặp real/fake AV, thử nghiệm nhỏ.
- **Tải:** https://www.idiap.ch/en/scientific-research/data/swan-df

### D5. DigiFakeAV — arXiv **2505.16512** (2025) [Tham khảo]
- Nội dung/dung lượng chưa đọc trực tiếp; link tải cần kiểm lại.

### D6. Deepfake-Eval-2024 — arXiv **2503.02857** [Web]
- **Nội dung:** 44h video + 56,5h audio + 1.975 ảnh, thu từ **88 website, 52 ngôn ngữ**; nhãn gán thủ công; **chưa có split chính thức** → tự chia theo nguồn.
- **Tải:** HF https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024 ; repo `nuriachandra/Deepfake-Eval-2024`.

---

## E. Khung benchmark / công cụ tổng hợp

| Công cụ | arXiv | Nội dung |
|---|---|---|
| **DeepfakeBench** | **2307.01426** (NeurIPS 2023) | Chuẩn hoá 9 dataset (FF++, Celeb-DF v1/v2, DFD, DFDC-P, DFDC, UADFV, FaceShifter, DF-1.0) + code đánh giá thống nhất (frame-level); repo `sclbd/deepfakebench`.  Chỉ frame-level, chưa có temporal detector. |
| **DF40** | **2406.13495** | 40 kỹ thuật, tích hợp pipeline DeepfakeBench. |
| **GenVidBench** | **2501.11340** | Benchmark video AI-gen. |
| **Deepfake-Eval-2024** | **2503.02857** | AV, generator 2024. |

---

## F. Nguồn REAL (để tự dựng test set / trộn âm-thanh-video)

| Nguồn | arXiv | Dùng để |
|---|---|---|
| **VoxCeleb1/2** | **1706.08612** / **1806.05622** | Video + audio thật của người nổi tiếng → nền tảng ghép fake (cũng là nguồn gốc nhiều dataset AV) |
| **LRS3** | **1809.00496** | Video môi + audio thật (LipForensics pretrain trên nó) |
| **FFHQ** | **1812.04948** | 70k face thật độ phân giải cao → test nhầm lẫn face real |

---

## G. Hướng dẫn chọn nhanh theo nhu cầu

| Mục tiêu | Chọn | Vì sao |
|---|---|---|
| Sanity test nhanh (giờ đầu) | UADFV (155MB) → CIFAKE (~1GB) → WaveFake (~1GB) | Nhỏ, tải ngay, không cần form |
| Benchmark video chuẩn | FF++ c23/c40 + Celeb-DF v2 (test 518 video) | Protocol có sẵn, so sánh được với paper |
| Cross-dataset generalization | train FF++ → test Celeb-DF v2; train FakeAVCeleb → test DFDC/DF-TIMIT | Protocol của AVFF/RealForensics |
| Robustness với nén | FF++ c23→c40→RAW; ADD 2022 LF | Đo độ bền |
| AI-generated ảnh (diffusion/GAN) | GenImage, UniversalFakeDetect, ArtiFact, CIFAKE, DiffusionDB | 8–25 generator |
| Audio chuẩn | ASVspoof 2019 LA (train/dev/eval + keys, EER) | Protocol chuẩn nhất |
| Audio in-the-wild / đa ngôn ngữ | In-the-Wild, MLAAD | Realistic |
| Audio partial-fake | ADD 2022 PF | Sát thực tế |
| Video + audio cùng lúc | FakeAVCeleb (train) + DFDC subset (test) | AVFF protocol |
| AV localization | LAV-DF, AV-Deepfake1M, ForgeryNet (temporal) | Tìm đoạn giả |
| Đa kỹ thuật / đa quốc tịch | DF40, ForgeryNet, KoDF, Deepfake-Eval-2024 | Generalization rộng |
| Video AI-gen 2024–2025 | GenVidBench | Ngoài faceswap |

---

## H. Lưu ý protocol & giấy phép (đọc trước khi khảo sát)

1. **Tự chạy, đừng tin số báo:** số trên paper không so sánh được (protocol khác nhau). Ví dụ đã kiểm chứng: Xception FF++ LQ báo 81.00% (video-level) vs 86.86% (frame-level, F3-Net) — cùng model khác protocol. DeepfakeBench cảnh báo chính việc này.
2. **Frame-level ≠ video-level:** quy ước rõ metric bạn báo là gì; frame-level (mỗi frame 1 dự đoán) hay video-level (gộp majority vote/mean). Báo cả hai.
3. **Split chuẩn:** FF++ = 720/140/140 video (không để lọt video giữa train/test); Celeb-DF v2 = test 518 video; ASVspoof = train/dev/eval + keys chính thức; audio báo **EER**.
4. **Form duyệt thủ công 1–2 tuần:** FF++, Celeb-DF, DF40, FakeAVCeleb → **xin sớm**. Tải ngay không cần form: UADFV, DF-TIMIT, GenImage (Drive), CIFAKE, WaveFake, In-the-Wild, MLAAD, Deepfake-Eval-2024, ASVspoof 2019 (DataShare), Celeb-DF v2 (Kaggle mirror).
5. **Audio có hay không (đã kiểm chứng):** FF++ **không** audio; KoDF audio thật (không fake); DFDC/FakeAVCeleb/AV-Deepfake1M/LAV-DF/Deepfake-Eval-2024 **có** audio.
6. **Giấy phép:** phần lớn dùng cho nghiên cứu phi thương mại (FF++, Celeb-DF, DeeperForensics, MLAAD CC-BY-NC-4.0, ADD). Kiểm license từng bộ trước khi publish code/weights. ASVspoof 2021: Open Data Commons Attribution.
7. **Công bố dataset trên HF** cần đọc lại card license (DiffusionDB CC0/MIT; gated repos cần accept license).

---

## Nguồn (URLs đã dùng khi xác minh)

- FF++: https://arxiv.org/abs/1901.08971 ; repo https://github.com/ondyari/FaceForensics
- Celeb-DF: https://arxiv.org/abs/1909.12962 ; https://forms.gle/2jYBby6y1FBU3u6q9 ; Kaggle https://www.kaggle.com/datasets/reubensuju/celeb-df-v2
- DFDC: https://arxiv.org/abs/2006.07397 ; https://ai.meta.com/datasets/dfdc/ ; DFDC-P https://arxiv.org/abs/1910.08854
- DeeperForensics-1.0: https://arxiv.org/abs/2001.03024 ; https://github.com/EndlessSora/DeeperForensics-1.0
- KoDF: https://arxiv.org/abs/2103.10094 ; https://github.com/deepbrainai-research/kodf
- ForgeryNet: https://arxiv.org/abs/2103.05630 ; https://yinanhe.github.io/projects/forgerynet
- WildDeepfake: https://arxiv.org/abs/2101.01456 ; https://github.com/OpenTAI/wild-deepfake
- DF-TIMIT: https://zenodo.org/records/4068245 ; https://www.idiap.ch/en/scientific-research/data/deepfaketimit
- UADFV: https://drive.google.com/drive/u/0/folders/1GEk1DSxmlV_61JtpEGzC9Fo_BffvyxpH ; Kaggle `uadfv-new`
- DF40: https://arxiv.org/abs/2406.13495 ; https://docs.google.com/forms/d/1ESAWoWusOEGEEVnXCH_emv-wJqCYMhCbD6-85RMIoDk/edit ; https://github.com/YZY-stack/DF40
- GenImage: https://arxiv.org/abs/2306.08571 ; https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS
- iFakeFaceDB: https://arxiv.org/abs/1911.05351 ; https://github.com/socialabubi/iFakeFaceDB
- UniversalFakeDetect: https://arxiv.org/abs/2302.10174 ; https://github.com/WisconsinAIVision/UniversalFakeDetect
- DiffusionDB: https://arxiv.org/abs/2210.14896 ; https://huggingface.co/datasets/poloclub/diffusiondb
- ArtiFact: https://arxiv.org/abs/2302.11970 ; https://huggingface.co/datasets/bitmind/ArtiFact
- CIFAKE: https://arxiv.org/abs/2303.14126 ; https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- ASVspoof 2019: https://arxiv.org/abs/1911.01601 ; https://datashare.ed.ac.uk/handle/10283/3336
- ASVspoof 2021: https://zenodo.org/records/4837263 ; https://zenodo.org/records/4835108 ; https://www.asvspoof.org
- WaveFake: https://arxiv.org/abs/2111.02813 ; https://zenodo.org/records/5642694
- In-the-Wild: https://arxiv.org/abs/2203.16263 ; https://huggingface.co/datasets/mueller91/In-The-Wild
- MLAAD: https://arxiv.org/abs/2401.09512 ; https://huggingface.co/datasets/mueller91/MLAAD
- ADD 2022: https://arxiv.org/abs/2202.08433 ; http://addchallenge.cn/downloadADD2022 ; https://zenodo.org/records/12188127
- FoR: https://bil.eecs.yorku.ca/datasets/ ; https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset
- FakeAVCeleb: https://arxiv.org/abs/2108.05080 ; https://github.com/DASH-Lab/FakeAVCeleb
- AV-Deepfake1M: https://arxiv.org/abs/2311.15308 ; https://huggingface.co/datasets/ControlNet/AV-Deepfake1M
- LAV-DF: https://arxiv.org/abs/2204.06228 ; bản mở rộng https://arxiv.org/abs/2305.01979 ; https://huggingface.co/datasets/ControlNet/LAV-DF
- Deepfake-Eval-2024: https://arxiv.org/abs/2503.02857 ; https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024
- DigiFakeAV: https://arxiv.org/abs/2505.16512 ; SWAN-DF: https://www.idiap.ch/en/scientific-research/data/swan-df
- GenVidBench: https://arxiv.org/abs/2501.11340
- DeepfakeBench: https://arxiv.org/abs/2307.01426 ; https://github.com/sclbd/deepfakebench
- VoxCeleb1/2: https://arxiv.org/abs/1706.08612 ; https://arxiv.org/abs/1806.05622
- LRS3: https://arxiv.org/abs/1809.00496 ; FFHQ: https://arxiv.org/abs/1812.04948

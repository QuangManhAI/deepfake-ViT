# Danh sách bài nghiên cứu: Phát hiện Deepfake bằng CNN & Vision Transformer

**Ngày tạo:** 2025 (phiên bản đầu tiên)
**Phạm vi:** Bài nghiên cứu về *anti-deepfake* — tập trung phát hiện deepfake ảnh/video khuôn mặt bằng CNN, Vision Transformer (ViT) và các mô hình lai CNN+ViT, kèm dataset chuẩn và survey tổng quan.
**Nguồn tra cứu:** arXiv / alphaXiv (đọc trực tiếp abstract hoặc toàn văn qua `alpha_get_paper`), xác minh chéo bằng web search cho các bài ngoài arXiv.
**Mức xác minh:** Mỗi bài có ghi chú mức độ: `Đã đọc trực tiếp` (đã mở abstract/toàn văn qua alphaXiv) hoặc `Tìm thấy qua tìm kiếm` (metadata + abstract từ kết quả tìm kiếm, chưa đọc toàn văn).

---

## Tóm tắt nhanh (TL;DR)

- **Nếu chỉ đọc 3 bài đầu tiên:** đọc survey ViT (2405.08463), dataset FaceForensics++ (1901.08971), và bài hybrid CNN+ViT kinh điển CViT (2102.11126).
- **Nếu làm thực nghiệm:** bắt đầu từ dataset FaceForensics++ + Celeb-DF, baseline Xception (trong bài FF++), rồi so sánh với CViT/M2TR.
- **Xu hướng chính trong tài liệu:** (1) CNN thuần tốt cho artifact cục bộ nhưng yếu generalization; (2) ViT nắm quan hệ toàn cục, tốt hơn khi kết hợp CNN ở đầu pipeline; (3) vấn đề lớn nhất hiện nay là *generalization* — model chết khi gặp phương pháp giả mạo mới, nén video mạnh, hoặc deepfake "in the wild".

---

## 1. Survey tổng quan (nên đọc trước)

### 1.1 DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection
- **Tác giả:** R. Tolosana, R. Vera-Rodriguez, J. Fierrez, A. Morales, J. Ortega-Garcia (BiDA Lab, UAM)
- **Nơi đăng:** Information Fusion, vol. 64, 2020 | arXiv: [2001.00179](https://arxiv.org/abs/2001.00179) — `Đã đọc trực tiếp`
- **Nội dung:** Phân loại thao tác khuôn mặt thành 4 nhóm (face synthesis, identity swap, attribute manipulation, expression swap); tổng hợp dataset công khai, phương pháp phát hiện (GAN-pipeline features, steganalysis, deep learning, behavioral/temporal cues) và benchmark kèm kết quả.
- **Điểm chốt:** Phát hiện tốt trên deepfake "thế hệ 1" (UADFV, FF++) nhưng AUC tụt dưới 60% trên Celeb-DF (thế hệ 2) — generalization là điểm yếu cố hữu.

### 1.2 A Timely Survey on Vision Transformer for Deepfake Detection
- **Tác giả:** Z. Wang, Z. Cheng, J. Xiong, X. Xu, T. Li, B. Veeravalli, X. Yang (A*STAR I2R / NUS / SWJTU)
- **Nơi đăng:** arXiv, 2024 | [2405.08463](https://arxiv.org/abs/2405.08463) — `Đã đọc trực tiếp`
- **Nội dung:** Survey chuyên sâu đầu tiên về ViT cho deepfake detection, cập nhật đến 02/2024. Phân loại kiến trúc thành 3 nhóm: **standalone ViT** (ICT, UIA-ViT, ViT-distillation...), **hybrid tuần tự CNN→ViT** (CViT, ISTVT...), **hybrid song song** (EfficientNet+ViT, M2TR, GenConViT...). Kèm benchmark tự chạy lại 5 model mã nguồn mở trên FF++ và Celeb-DF.
- **Điểm chốt:** M2TR bền nhất với nén (FF++ LQ: ACC 87.19%, AUC 0.904); GenConViT giảm mạnh khi nén (97.68% RAW → 48.56% LQ); CNN+Với ViT tuần tự (Khan et al.) tăng AUC từ 77.10% → 99.28% trên FF++ so với ViT thuần.

### 1.3 Deepfake Detection: A Comprehensive Survey from the Reliability Perspective (bổ sung)
- arXiv: [2211.10881](https://arxiv.org/abs/2211.10881) — `Tìm thấy qua tìm kiếm`
- Góc nhìn "độ tin cậy" của detector: sai số, tấn công đối nghịch, đánh giá trong thực tế. Hữu ích nếu bạn quan tâm khía cạnh *anti-deepfake* theo nghĩa an toàn/đáng tin cậy.

---

## 2. Dataset chuẩn (nền tảng cho mọi thực nghiệm)

### 2.1 FaceForensics++: Learning to Detect Manipulated Facial Images
- **Tác giả:** A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, M. Nießner (TUM / Napoli / Erlangen)
- **Nơi đăng:** ICCV 2019 | arXiv: [1901.08971](https://arxiv.org/abs/1901.08971) — `Đã đọc trực tiếp`
- **Nội dung:** 1.000 video gốc từ YouTube + 4 phương pháp thao tác (DeepFakes, Face2Face, FaceSwap, NeuralTextures), 3 mức chất lượng (RAW/HQ/LQ). Đánh giá 5 CNN + đặc trưng steganalysis + người.
- **Điểm chốt:** Baseline mạnh nhất là **XceptionNet fine-tune + face tracking**: 99.26% (RAW), 95.73% (HQ), 81.00% (LQ); người chỉ đạt 68.69%/58.73%. Xception trở thành baseline chuẩn của hầu hết bài sau này.

### 2.2 Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics
- **Tác giả:** Y. Li, X. Yang, P. Sun, H. Qi, S. Lyu (UAlbany / UCAS)
- **Nơi đăng:** CVPR 2020 | arXiv: [1909.12962](https://arxiv.org/abs/1909.12962) — `Đã đọc trực tiếp`
- **Nội dung:** 590 video thật + 5.639 deepfake chất lượng cao (256×256, giảm color mismatch, mask mượt, lọc Kalman chống flicker). Đánh giá 9 detector cũ.
- **Điểm chốt:** AUC trung bình của 9 phương pháp chỉ **56.9%** — dataset khó nhất thời điểm đó; benchmark bắt buộc để kiểm tra generalization.

### 2.3 The DeepFake Detection Challenge (DFDC) Dataset
- **Tác giả:** B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang, C. Canton Ferrer (Facebook AI)
- **Nơi đăng:** arXiv, 2020 | [2006.07397](https://arxiv.org/abs/2006.07397) — `Đã đọc trực tiếp`
- **Nội dung:** >100k clip giả từ 3.426 diễn viên trả phí (có đồng thuận), nhiều phương pháp (DFAE, MM/NN, NTH, FSGAN, StyleGAN), tập private test chứa 50% deepfake thật "in the wild".
- **Điểm chốt:** Model top challenge dùng **ensemble CNN (EfficientNet-B7, Xception, ResNet) + 3D CNN** (SlowFast/I3D); best model đạt avg precision 0.753 trên phần "real videos" của private test — đủ chứng minh huấn luyện trên DFDC chuyển được ra ngoài đời.

---

## 3. Phương pháp dựa trên CNN

### 3.1 MesoNet: a Compact Facial Video Forgery Detection Network
- **Tác giả:** D. Afchar, V. Nozick, J. Yamagishi, I. Echizen
- **Nơi đăng:** IEEE WIFS 2018 | arXiv: [1809.00888](https://arxiv.org/abs/1809.00888) — `Đã đọc trực tiếp`
- **Nội dung:** CNN nhỏ (Meso-4, MesoInception-4) học đặc trưng *mesoscopic* để phát hiện Deepfake/Face2Face.
- **Điểm chốt:** Bài CNN kinh điển đầu tiên; AUC 84.3% trên UADFV validation, tốt trên FF++ raw nhưng kém trên nén — là baseline tham khảo lịch sử.

### 3.2 Face Forgery Detection by Mining Frequency-aware Clues (F3-Net)
- **Tác giả:** Y. Qian, G. Yin, L. Sheng, Z. Chen, J. Shao (SenseTime)
- **Nơi đăng:** ECCV 2020 | arXiv: [2007.09355](https://arxiv.org/abs/2007.09355) — `Tìm thấy qua tìm kiếm, đã xác minh ID`
- **Nội dung:** Hai nhánh: (1) khai thác tín hiệu frequency-domain (DCT) để tìm artifact; (2) học chênh lệch phân bố tần số giữa thật/giả.
- **Điểm chốt:** Đại diện cho hướng *frequency-domain* — quan trọng vì artifact bị phá hủy bởi JPEG compression nhưng vẫn sót lại trong miền tần số. Được M2TR và nhiều bài sau so sánh.

### 3.3 DeepRhythm: Exposing DeepFakes with Attentional Visual Heartbeat Rhythms
- **Tác giả:** H. Qi et al.
- **Nơi đăng:** ACM MM 2020 | arXiv: [2006.07634](https://arxiv.org/abs/2006.07634) — `Đã đọc trực tiếp (abstract)`
- **Nội dung:** Dùng tín hiệu sinh học — photoplethysmography (PPG) từ video — để nhận diện "nhịp tim" bất thường trong deepfake.
- **Điểm chốt:** Đại diện hướng *biological signal*: detection dựa trên đặc tính vật lý không phụ thuộc phương pháp tạo fake cụ thể.

---

## 4. Phương pháp dựa trên Vision Transformer

### 4.1 Protecting Celebrities from DeepFake with Identity Consistency Transformer (ICT)
- **Tác giả:** W. Dong, Z. Wang, M. Wang, J. Hu, J. Hu, Z. Guo, B. Zhang
- **Nơi đăng:** arXiv, 2022 | [2203.01318](https://arxiv.org/abs/2203.01318) — `Tìm thấy qua tìm kiếm, đã xác minh ID`
- **Nội dung:** ViT thuần (standalone) học *high-level semantics* — so sánh độ nhất quán danh tính giữa các vùng mặt thay vì artifact pixel.
- **Điểm chốt:** Được survey ViT đánh giá là model standalone tiêu biểu, generalization tốt vì dựa trên bất nhất ngữ nghĩa cấp cao thay vì pattern nhiễu của từng generator.

> Lưu ý: survey ViT (2405.08463) liệt kê thêm các standalone model khác (UIA-ViT, ViT-distillation, Shallow ViT) — đọc survey để có bức tranh đầy đủ hơn.

---

## 5. Phương pháp lai CNN + ViT (trọng tâm cho hướng của bạn)

### 5.1 Deepfake Video Detection Using Convolutional Vision Transformer (CViT)
- **Tác giả:** D. Wodajo, S. Atnafu (Jimma University / Addis Ababa University)
- **Nơi đăng:** arXiv, 2021 | [2102.11126](https://arxiv.org/abs/2102.11126) — `Đã đọc trực tiếp (toàn văn)`
- **Nội dung:** CNN 17 lớp (kiểu VGG, không FC) trích đặc trưng cục bộ → ViT (8 head) phân loại bằng attention toàn cục. Tiền xử lý kỹ (BlazeFace + MTCNN + face_recognition, augmentation 90%).
- **Kết quả:** DFDC: **91.5% accuracy, AUC 0.91** (400 video test); FF++ Deepfake 93%, nhưng FaceShifter chỉ 46% — minh chứng rõ giới hạn generalization.
- **Điểm chốt:** Bài "CNN + ViT" kinh điển, kiến trúc tuần tự, dễ đọc, dễ tái hiện — khởi điểm tốt cho đồ án.

### 5.2 M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection
- **Tác giả:** J. Wang, Z. Wu, W. Ouyang, X. Han, J. Chen, S.-N. Lim, Y.-G. Jiang (Fudan / Huya / Meta AI)
- **Nơi đăng:** ACM ICMR 2022 | arXiv: [2104.09770](https://arxiv.org/abs/2104.09770) — `Đã đọc trực tiếp (toàn văn)`
- **Nội dung:** Transformer đa tỉ lệ (patch 80/40/20/10 px ở các head khác nhau) + nhánh frequency (FFT với filter học được) + cross-modality fusion; học multi-task (phân loại + mask vùng giả) và contrastive loss. Kèm dataset SR-DF.
- **Kết quả:** FF++ LQ: ACC 92.89%, AUC 0.9531 (SOTA so với F3-Net, MaDD); Celeb-DF AUC 99.8 (trained FF++ → test Celeb-DF: 68.2 AUC cross-dataset).
- **Điểm chốt:** Model hybrid ViT đại diện nhất trong survey — kết hợp đúng hai ý tưởng bạn quan tâm: CNN backbone (EfficientNet-b4) + ViT đa tỉ lệ + frequency.

### 5.3 GenConViT: Deepfake Video Detection Using Generative Convolutional Vision Transformer
- **Tác giả:** P. H. S. Toro et al. (nhóm GenConViT)
- **Nơi đăng:** arXiv, 2023 | [2307.07036](https://arxiv.org/abs/2307.07036) — `Tìm thấy qua tìm kiếm, đã xác minh ID`
- **Nội dung:** Kết hợp ConvNeXt (CNN hiện đại) + Swin Transformer (hierarchical ViT) theo cấu trúc song song, tận dụng generator-based training.
- **Kết quả:** Theo benchmark của survey ViT: rất mạnh trên FF++ RAW (97.68% ACC) nhưng sụp đổ khi nén (48.56% LQ) — ví dụ điển hình của overfitting artifact chất lượng cao.

### 5.4 Combining EfficientNet and Vision Transformers for Video Deepfake Detection
- **Tác giả:** D. A. Coccomini, N. Messina, C. Gennaro, F. Falchi (ISTI-CNR, Pisa)
- **Nơi đăng:** ICIAP 2022 | arXiv: [2107.02612](https://arxiv.org/abs/2107.02612) — `Xác minh qua web search`
- **Nội dung:** ViT + EfficientNet kết hợp cho video deepfake; là model "David et al." mà survey ViT gọi là hybrid song song.
- **Điểm chốt:** Minh họa cách gắn CNN hiệu quả (EfficientNet) với ViT mà không cần kiến trúc quá phức tạp.

---

## 6. Generalization & phương pháp cải thiện độ bền (hướng hot nhất 2023–2025)

### 6.1 Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection (RealForensics)
- **Tác giả:** A. Haliassos, R. Mira, S. Petridis, M. Pantic (Imperial College London)
- **Nơi đăng:** CVPR 2022 | arXiv: [2201.07131](https://arxiv.org/abs/2201.07131) — `Xác minh qua web search`; code: github.com/ahaliassos/RealForensics
- **Nội dung:** Self-supervised học từ video khuôn mặt thật (tự nhiên về appearance + behavior) làm tiền đề cho detector, không cần fake data lúc pretrain; chống nén tốt.
- **Điểm chốt:** Hướng *self-supervision / natural consistency* — hiện là một trong các trục nghiên cứu mạnh nhất về generalization.

### 6.2 UCF: Uncovering Common Features for Generalizable Deepfake Detection
- **Tác giả:** Z. Yan, Y. Zhang, Y. Fan, B. Wu (CUHK-SZ)
- **Nơi đăng:** ICCV 2023 | arXiv: [2304.13949](https://arxiv.org/abs/2304.13949) — `Xác minh qua web search`
- **Nội dung:** Phân rã đặc trưng thành *common feature* (dùng chung mọi loại fake) và *specific feature* (riêng từng generator) để tránh overfit pattern riêng của từng phương pháp.
- **Điểm chốt:** Bài nền tảng cho dòng "forgery-specific vs common features" — thường xuyên xuất hiện trong ablation của các bài 2024–2025.

### 6.3 Các bài mới liên quan (tìm thấy qua alpha_search, chưa đọc toàn văn)
- **Wavelet-Driven Generalizable Framework for Deepfake Face Detection** — [2409.18301](https://arxiv.org/abs/2409.18301), 2024.
- **Generalized Face Forgery Detection via Adaptive Learning for Pre-trained ViT** — [2309.11092](https://arxiv.org/abs/2309.11092), 2023 (fine-tune hiệu quả ViT pretrained cho face forgery).
- **Detecting Face Forgeries with Domain-adversarial Triplet Learning** — [2506.23189](https://arxiv.org/abs/2506.23189), 2025 (ViT + domain-adversarial, mới nhất).

---

## Bảng so sánh nhanh

| Bài | Năm | Loại | Điểm nổi bật | Kết quả tham chiếu (nguồn gốc) |
|---|---|---|---|---|
| MesoNet [1809.00888] | 2018 | CNN | CNN mesoscopic đầu tiên | AUC 84.3% UADFV (bài báo) |
| FF++ / Xception [1901.08971] | 2019 | CNN | Baseline chuẩn + dataset | 99.26% RAW / 81.00% LQ (bài báo) |
| Tolosana survey [2001.00179] | 2020 | Survey | Phân loại 4 nhóm manipulation | AUC <60% trên Celeb-DF (bài báo) |
| F3-Net [2007.09355] | 2020 | CNN + frequency | Frequency-aware | FF++ LQ AUC 93.3 (M2TR trích dẫn) |
| Celeb-DF [1909.12962] | 2020 | Dataset | Deepfake chất lượng cao | AUC trung bình 9 model: 56.9% (bài báo) |
| DFDC [2006.07397] | 2020 | Dataset | >100k clip, có consent | Best AP 0.753 in-the-wild (bài báo) |
| DeepRhythm [2006.07634] | 2020 | CNN + PPG | Tín hiệu sinh học | — |
| CViT [2102.11126] | 2021 | CNN→ViT | Kiến trúc lai kinh điển | 91.5% ACC, AUC 0.91 DFDC (bài báo) |
| M2TR [2104.09770] | 2022 | ViT đa tỉ lệ + freq | SOTA benchmark survey ViT | FF++ LQ ACC 92.89 (bài báo) |
| ICT [2203.01318] | 2022 | ViT standalone | Identity consistency | — |
| RealForensics [2201.07131] | 2022 | Self-supervised | Generalization + nén | CVPR 2022 |
| EfficientNet+ViT [2107.02612] | 2022 | CNN+ViT | Hybrid song song | ICIAP 2022 |
| GenConViT [2307.07036] | 2023 | CNN+ViT | ConvNeXt+Swin | FF++ RAW 97.68% / LQ 48.56% (survey ViT) |
| ViT survey [2405.08463] | 2024 | Survey | Phân loại 3 nhóm ViT | Benchmark 5 model (survey) |
| UCF [2304.13949] | 2023 | Common/specific features | Generalization | ICCV 2023 |

---

## Trình tự đọc đề xuất (nếu bạn mới bắt đầu)

1. **Tolosana 2020** (2001.00179) — bức tranh toàn cảnh: 4 nhóm manipulation, dataset, phương pháp.
2. **FaceForensics++** (1901.08971) — hiểu dataset + baseline Xception (thứ bạn sẽ so sánh mọi thứ với).
3. **CViT** (2102.11126) — đọc kỹ toàn văn: pipeline CNN→ViT đơn giản, dễ làm lại.
4. **Survey ViT 2024** (2405.08463) — chọn hướng ViT nào phù hợp; đọc phần benchmark để biết model nào bền với nén.
5. **M2TR** (2104.09770) — mẫu kiến trúc hybrid hiện đại hơn (đa tỉ lệ + frequency + multi-task).
6. **UCF / RealForensics** — trước khi viết related work về generalization.

## Gợi ý thực nghiệm kế tiếp (nếu bạn đang làm đồ án)

- **Setup tối thiểu:** FF++ (RAW + LQ) và Celeb-DF làm test generalization; mặt cắt bằng RetinaFace/MTCNN; metric ACC + AUC + cross-dataset AUC (train FF++ → test Celeb-DF).
- **So sánh công bằng:** luôn kèm baseline Xception (c23) và CViT; ghi rõ cấu hình nén vì kết quả thay đổi mạnh theo LQ/HQ.
- **Câu hỏi mở còn bỏ ngỏ trong tài liệu:** (1) model drift — detector cũ chết khi generator mới (diffusion) ra đời; (2) tấn công đối nghịch lên detector (anti-forensics); (3) đánh giá đa phương thức (audio + visual) — RealForensics và các bài 2025 đang đi theo hướng này; (4) chuẩn benchmark thống nhất để so sánh công bằng giữa các paper.

---

## Nguồn (Sources)

Tất cả link arXiv/alphaXiv trực tiếp đã ghi ở từng mục. Tóm tắt:

- https://arxiv.org/abs/2001.00179 (Tolosana survey)
- https://arxiv.org/abs/2405.08463 (ViT survey)
- https://arxiv.org/abs/1901.08971 (FaceForensics++)
- https://arxiv.org/abs/1909.12962 (Celeb-DF)
- https://arxiv.org/abs/2006.07397 (DFDC)
- https://arxiv.org/abs/1809.00888 (MesoNet)
- https://arxiv.org/abs/2007.09355 (F3-Net)
- https://arxiv.org/abs/2006.07634 (DeepRhythm)
- https://arxiv.org/abs/2203.01318 (ICT)
- https://arxiv.org/abs/2102.11126 (CViT)
- https://arxiv.org/abs/2104.09770 (M2TR)
- https://arxiv.org/abs/2307.07036 (GenConViT)
- https://arxiv.org/abs/2107.02612 (EfficientNet+ViT)
- https://arxiv.org/abs/2201.07131 (RealForensics; code: https://github.com/ahaliassos/RealForensics)
- https://arxiv.org/abs/2304.13949 (UCF)
- https://arxiv.org/abs/2211.10881 (Survey độ tin cậy)
- https://arxiv.org/abs/2409.18301 | https://arxiv.org/abs/2309.11092 | https://arxiv.org/abs/2506.23189 (các bài 2023–2025 bổ sung)

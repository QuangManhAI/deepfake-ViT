# Dataset Video + Audio cho Deepfake Detection (Audio-Visual)

*Ngày: 2026-06. Mục đích: các bộ data có **cả video và audio** để làm phát hiện deepfake đa phương thức. Mức xác minh: [README] = số liệu đọc trực tiếp từ README chính thức; [Web] = xác minh qua tìm kiếm/arXiv.*

---

## 1. Dataset có FAKE (video + audio) — dùng để train/test detection

| Dataset | Năm / Venue | Real | Fake | Tổng | Loại giả mạo | Audio thật | Audio giả | Fine-grained | Truy cập |
|---|---|---|---|---|---|---|---|---|---|
| **FakeAVCeleb** [2108.05080, NeurIPS 2021 D&B] | 2021 | 500 | 19.500 | 20.000 | Face-swap (DeepFakes/FSGAN), voice clone (SV2TTS), FS+VC | ✅ | ✅ | ✅ (duy nhất trong bảng so sánh) | Google form → download script [README] |
| **DFDC** [2006.07397] | 2020 | 23.654 | 104.500 | 128.154 | 8 kỹ thuật, in-the-wild | ✅ | ✅ | ❌ | Kaggle (chấp nhận điều khoản); bản đầy đủ ~470GB [README] |
| **AV-Deepfake1M** [2311.15308, ACM MM 2024 Best Award] | 2023 | 2.068 chủ thể, **1.886 giờ** | **>1 triệu video giả** | Content-driven (LLM); **KHÔNG có face-swap** — visual = **TalkLip** (lip-sync), audio = **VITS** (TTS phụ thuộc định danh) + **YourTTS** (độc lập); 3 loại: FA+FV / FA+RV / RA+FV | ✅ | ✅ | ✅ (segments — thiết kế cho **localization**, không phải classification) | HF gated `ControlNet/AV-Deepfake1M`; **train ≈ 254GB (254 phần zip × 1GB, từ listing)**; authors thừa nhận **mất cân bằng real/fake** [arXiv] |
| **Deepfake-Eval-2024** [2503.02857] | 2024 | — | — | video+audio+ảnh in-the-wild | lipsync, faceswap, diffusion | ✅ | ✅ | ❌ | GitHub `nuriachandra/Deepfake-Eval-2024` [Web] |
| **DigiFakeAV** [2505.16512] | 2025 | — | — | Digital human diffusion, multimodal control | ✅ | ✅ | ❌ | arXiv [Web] |
| **SWAN-DF** (Idiap) | 2020 | — | — | 30 cặp người; face swap + voice swap (autoencoder) | ✅ | ✅ | ❌ | Trang Idiap [Web] |
| **KoDF** | 2021 | 62.166 | 175.776 | 237.942 | 6 kỹ thuật (visual) | ✅ | ❌ | ❌ | GitHub — *audio là thật, chỉ visual bị sửa* [README] |

**Ghi chú quan trọng:**
- **FakeAVCeleb là chuẩn de facto cho train audio-visual** — bài AVFF (CVPR 2024) train trên FakeAVCeleb rồi cross-test DF-TIMIT + DFDC subset [Web]. Có **fine-grained labeling** (biết clip nào FS, VC, hay FS+VC) → test được riêng từng loại giả mạo.
- **DFDC** là lựa chọn in-the-wild (clip quay thực tế có lời nói, nhiều kỹ thuật); audio nền ồn, khó hơn FakeAVCeleb.
- **AV-Deepfake1M** to nhất (1.886h), kèm localization — nhưng **chỉ có lip-sync (TalkLip) + voice-clone (VITS/YourTTS), không có face-swap** → không dùng làm dataset finetune duy nhất nếu bạn muốn detector tổng quát.
- **KoDF** thú vị ở chỗ: audio **thật** + visual **fake** → dùng để test "bất nhất chéo modality" (audio nói đúng, mặt không khớp môi/khớp giọng).

## 2. Dataset THẬT (không fake) — dùng để pretrain / học prior audio-visual

| Dataset | Nội dung | Dùng cho |
|---|---|---|
| **VoxCeleb2** | ~1M câu nói, 6.112 người nổi tiếng (video phỏng vấn YouTube) | Pretrain audio-visual; FakeAVCeleb xây dựng trên này [README] |
| **LRS3** | ~400h audio-visual speech (Ted talks) | Lip-sync consistency (như LipForensics-style) [Web] |
| **LRW** | Lip Reading in the Wild — 1.000 từ | Prior khớp môi–giọng |

## 3. Khuyến nghị theo mục đích

| Nếu bạn muốn... | Chọn |
|---|---|
| Test audio-visual với 4 nhóm rõ ràng (thật / FS / VC / FS+VC) | **FakeAVCeleb** (nhỏ, sạch, fine-grained) |
| In-the-wild thực tế, scale vừa | **DFDC** subset (Kaggle) |
| Scale lớn + localization | **AV-Deepfake1M** (HF gated) |
| Generator 2024–2025 (diffusion, lipsync) | **Deepfake-Eval-2024** / **DigiFakeAV** |
| Phát hiện bất nhất audio–video (visual fake + audio thật) | **KoDF**, DFDC |

## 4. Cảnh báo / pitfall khi tự chạy

1. **FakeAVCeleb chỉ có 500 video thật** (1/người nổi tiếng, ~6,4s) — tập real nhỏ; phần lớn fake là FS+VC đồng thời. Đừng chia train/test theo video ngẫu nhiên — hãy chia **theo danh tính** (identity-disjoint) để tránh rò rỉ.
2. **Nguồn gốc VoxCeleb2** (CC-BY-4.0, cần kiểm tra license khi tải) — identity có thể trùng với dữ liệu YouTube khác.
3. **Pipeline audio bắt buộc**: trích 16kHz mono → feature (MFCC / Wav2Vec2 / AST); đồng bộ âm–hình cần kiểm tra (AVFF-style).
4. **Đừng dùng FF++/Celeb-DF cho audio** — không có audio (xác minh README FF++), audio không phân biệt được fake/thật.
5. Metric nên báo **theo từng loại giả mạo** (FS / VC / FS+VC riêng) — AUC chung dễ che giấu điểm yếu (voice-clone thường khó hơn face-swap).

## 5. FINETUNE: chọn dataset nào (trả lời chi tiết)

**AV-Deepfake1M = lớn nhất về video+audio (1.886h, >1 triệu video). Nhưng "lớn" ≠ "đúng" cho finetune detector của bạn — 4 điểm phải biết trước khi tải ~254GB:**

| Tiêu chí | AV-Deepfake1M | FakeAVCeleb | DFDC (full) |
|---|---|---|---|
| Quy mô | 1.886h, >1M video | 20.000 video | 128.154 clip |
| Face-swap | ❌ **KHÔNG có** (chỉ TalkLip lip-sync) | ✅ DeepFakes/FSGAN | ✅ 8 kỹ thuật |
| Voice-clone | ✅ VITS/YourTTS | ✅ SV2TTS | ✅ (audio thật + giả) |
| Thiết kế cho | **Localization** (real+fake xen kẽ trong video) | Classification 4 nhóm | Classification in-the-wild |
| Cân bằng class | ⚠️ Mất cân bằng (authors thừa nhận) | Cân bằng 4 nhóm | Tự chọn subset |
| Split | ✅ identity-disjoint: 1.657 train / 411 test (test chỉ VITS identity-dependent) | Tự chia | Có sẵn |
| Lưu trữ | **≥254GB train + test/val** | ~chục GB | ~470GB |

**Kết luận theo mục tiêu:**
- Muốn **scale + bài toán localization audio-visual** (tìm đoạn giả trong video dài) → **AV-Deepfake1M** là lựa chọn duy nhất ở quy mô này; phù hợp nếu hướng của bạn là temporal localization hoặc cần nhiều dữ liệu TTS/lip-sync.
- Muốn **detector tổng quát** (bắt cả face-swap lẫn voice-clone) → AV-Deepfake1M **thiếu hẳn nhánh face-swap** — model chỉ học artifact lip-sync, sẽ mù với faceswap. Dùng **FakeAVCeleb (chính) + DFDC subset**.
- Recipe "khủng mà an toàn": **FakeAVCeleb + DFDC subset + AV-Deepfake1M subset (~50–100GB)** → test cross-dataset: FakeAVCeleb 4-way, DFDC, Deepfake-Eval-2024 (generator 2024).

## 6. Liên hệ project hiện tại

- Đây chính là trục "đánh giá đa phương thức (audio + visual)" được liệt kê là câu hỏi mở trong reading list — tạm gọi **Gap G (audio-visual)**.
- Mở rộng tự nhiên từ Gap A: giữ nguyên 2 model (DINOv3 ViT-S+ vs ConvNeXt-Tiny) → thêm nhánh audio (Wav2Vec2/AST) → so 3 chế độ: visual-only / audio-only / fused, trên FakeAVCeleb 4-way + cross-test DFDC subset.
- Chi phí: trung bình (cần thêm encoder audio + fusion); dữ liệu FakeAVCeleb tải qua form — xin trước vì duyệt thủ công.

---

## Nguồn

- FakeAVCeleb: https://arxiv.org/abs/2108.05080 ; repo https://github.com/DASH-Lab/FakeAVCeleb (README — bảng so sánh dataset, số [README]) [Đã đọc README]
- AV-Deepfake1M: https://arxiv.org/abs/2311.15308 ; HF https://huggingface.co/datasets/ControlNet/AV-Deepfake1M ; repo https://github.com/ControlNet/AV-Deepfake1M [Web]
- Deepfake-Eval-2024: https://arxiv.org/abs/2503.02857 ; repo https://github.com/nuriachandra/Deepfake-Eval-2024 [Web]
- DigiFakeAV: https://arxiv.org/abs/2505.16512 [Web]
- SWAN-DF: https://www.idiap.ch/en/scientific-research/data/swan-df [Web]
- AVFF (CVPR 2024, train FakeAVCeleb → cross-test): https://openaccess.thecvf.com/content/CVPR2024/supplemental/Oorloff_AVFF_Audio-Visual_Feature_CVPR_2024_supplemental.pdf [Web]
- DFDC: https://ai.meta.com/datasets/dfdc/ ; Kaggle https://www.kaggle.com/competitions/deepfake-detection-challenge/data [Web]

# Các nghiên cứu Deepfake Detection dùng CẢ Video + Audio (Audio-Visual)

> Trả lời câu hỏi: *"Liệt kê các nghiên cứu mà model phân loại dựa trên cả video + audio"*.
> Đây là danh mục **phương pháp (model)**, không phải dataset. Dataset AV đã có ở `outputs/deepfake-datasets-catalog.md` §D và `outputs/audio-visual-datasets.md`.
> Mọi arXiv ID đều đã xác minh qua tìm kiếm/fetch trực tiếp. Số liệu định lượng chỉ đưa khi có nguồn gốc rõ (xem §8).

**Chú thích mức xác minh:**
- **[Đã xác minh]** = đã đọc trực tiếp paper/supplement/repo.
- **[Web]** = xác minh qua ≥2 nguồn web khớp (tiêu đề, ID arXiv, venue).
- Code: **[có code]** = đã thấy repo chính thức; **[code 1 phần]** = repo nhưng thiếu mô-đun; **—** = chưa xác minh.

---

## 1. TL;DR

1. **Audio-visual deepfake detection LÀ một hướng nghiên cứu thật, đang phát triển mạnh**: từ 2021 (DefakeHop-style, POI-Forensics 2023) đến 2024–2026 có tối thiểu **~20 phương pháp** ở CVPR/ICCV/NeurIPS/ACM MM/TIFS + hàng loạt preprint.
2. Chia 3 nhóm kiến trúc chính:
   - **Fusion/ensemble**: học đặc trưng chung của 2 modality (AVFF, AVTENet, AVT²-DWF, X-AVDT…).
   - **Consistency/mismatch**: tận dụng **sự lệch môi–giọng / lệch thời gian** (Lips Are Lying, Voice-Face Homogeneity, Lost in Translation, LIPINC…).
   - **Identity verification**: so khớp người nói–khuôn mặt (POI-Forensics).
3. Bằng chứng định lượng mạnh nhất đã đọc trực tiếp: **AVFF (CVPR 2024)** — cross-dataset train FakeAVCeleb → test DFDC: **AP/AUC 87.0/86.2**, thắng visual-only tốt nhất (RealForensics 82.9/83.7) **+4.1 AP**; test DF-TIMIT đạt **100/100**.
4. **Có code chạy được ngay** (đã xác minh repo): AVT²-DWF, X-AVDT (CVPR 2026), LIPINC, ICS-AV (ICCV 2025). AVoiD-DF chỉ mở code 1 phần.
5. ⚠️ Hiệu chỉnh quan trọng: **HAMMER thường bị nhầm là video+audio nhưng thật ra là ảnh+text**; DefakeHop là visual-only; HM-Conformer là audio-only → đừng trích dẫn nhầm.

---

## 2. Bảng tổng quan

| # | Phương pháp | Venue / Năm | Ý tưởng cốt lõi | Code | Xác minh |
|---|---|---|---|---|---|
| 1 | **AVFF** (2406.02951) | CVPR 2024 | Fusion audio-visual; train FakeAVCeleb, generalizable | — | [Đã xác minh] |
| 2 | **AVTENet** (2310.13103) | IEEE TIFS 2025 (preprint 2023) | Ensemble transformer, lấy cảm hứng nhận thức con người | — | [Web] |
| 3 | **AVT²-DWF** (2403.14974) | 2024 | 2 transformer (âm + hình) + dynamic weight fusion | [có] `raining-dev/AVT2-DWF` | [Web] |
| 4 | **X-AVDT** (2603.08483) | CVPR 2026 | Cross-attention giữa audio-visual, góc nhìn generator | [có] `youngseo0526/X-AVDT` | [Web] |
| 5 | **Explicit Correlation Learning** (2404.19171) | 2024 | Học tương quan chéo tường minh → generalizable | — | [Web] |
| 6 | **Contextual Cross-Modal Attention** (2408.01532) | 2024 | Attention chéo theo ngữ cảnh, có localization | — | [Web] |
| 7 | **DiMoDif** (2411.10193) | 2024 | Phân biệt thông tin theo modality, detection + localization | — | [Web] |
| 8 | **Self-supervised AV reps** (2511.17181) | 2025 | Dùng biểu diễn tự-supervised cho AV detection | — | [Web] |
| 9 | **ERF-BA-TFD+** (2508.17282) | 2025 | Receptive field mở rộng + fusion âm/hình | — | [Web] |
| 10 | **HOLA** (2507.22781) | 1M-Deepfakes Challenge 2025 | Video-level, hierarchical aggregation + pretrain | — | [Web] |
| 11 | **FauForensics** (2505.08294) | 2025 | Facial Action Units sinh học bất biến | — | [Web] |
| 12 | **Forgery-aware AV Adaptation** (2511.19080) | 2025 | Variational Bayesian, adaptation đa dataset | — | [Web] |
| 13 | **AVoiD-DF** (IEEE 10081373) | IEEE TIFS 2023 | Joint learning, khai thác AV inconsistency | [1 phần] `SYSU-DISG/AVoiD-DF` | [Web] |
| 14 | **Cross-/Within-Modality Regularization** (2401.05746) | 2024 | Regularization giữ bản sắc từng modality | — | [Web] |
| 15 | **MIS-AVoiDD** (2310.02234) | 2023 | Biểu diễn modality-invariant + specific | — | [Web] |
| 16 | **Voice-Face Homogeneity** (2203.02195) | ACM TOMM 2024 | Độ thuần nhất giọng–mặt | — | [Web] |
| 17 | **Lips Are Lying** (2401.15668) | NeurIPS 2024 | Lệch thời gian môi–audio trong lip-sync | — | [Web] |
| 18 | **Lost in Translation** (CVPRW 2024) | CVPR 2024 WMF | Lip-sync từ audio–video mismatch (Bohacek & Farid) | — | [Web] |
| 19 | **LIPINC** (2401.10113) | 2024 | Inconsistency vùng miệng theo thời gian | [có] `skrantidatta/LIPINC` | [Web] |
| 20 | **Fine-Grained Inconsistencies** (2408.06753) | BMVC 2024 | Artifact tinh (spatial + temporal) AV | — | [Web] |
| 21 | **ICS-AV: Intra/Cross-modal Sync** (ICCV 2025) | ICCV 2025 | Đồng bộ intra + cross-modal, temporal localization | [có] `AshutoshAnshul/ics-av-deepfake` | [Web] |
| 22 | **POI-Forensics** (2204.03083) | CVPR 2023 WMF | Person-of-interest: so khớp danh tính âm–hình (contrastive) | — | [Web] |
| 23 | **LAV-DF** (2204.06228 / 2305.01979) | DICTA 2022 / journal | Detection + temporal localization AV | — | [Web] |
| — | **Survey AV detection** (2411.07650) | 2024 | Survey đầu tiên chuyên audio-visual | — | [Web] |

---

## 3. Nhóm FUSION / ENSEMBLE (học đặc trưng chung 2 modality)

**3.1. AVFF — Audio-Visual Feature Fusion** — arXiv **2406.02951** (CVPR 2024, pp. 27102–27112) [Đã xác minh — đọc trực tiếp supplemental]
- Tác giả: Oorloff, Koppisetti, Bonettini, Solanki, Colman, Yacoob, Shahriyari, Bharaj (UMD + Reality Defender).
- Ý tưởng: fusion audio-visual features; **train trên FakeAVCeleb, test cross-dataset** (DFDC, DF-TIMIT, FF++, Celeb-DF).
- Lý do thắng (đọc trực tiếp): face-swap giữ nguyên audio thật → lộ qua **AV mismatch**; lip-sync/voice-clone giữ nguyên video thật → lộ qua mismatch ngược; single-modality chỉ bắt được nửa số đó.
- Số liệu cross-dataset (train FakeAVCeleb → test, bảng trong supplemental — đã đọc):
  - **DFDC**: Xception 68.0/67.6 (AP/AUC), LipForensics 76.8/77.4, FTCN 70.5/71.1, RealForensics 82.9/83.7, **AVFF 87.0/86.2** (+4.1 AP so với visual tốt nhất).
  - **DF-TIMIT**: Xception 86.0/90.5, RealForensics 99.2/99.5, **AVFF 100/100**.
- Link: https://arxiv.org/abs/2406.02951 ; supplemental: https://openaccess.thecvf.com/content/CVPR2024/supplemental/Oorloff_AVFF_Audio-Visual_Feature_CVPR_2024_supplemental.pdf

**3.2. AVTENet** — arXiv **2310.13103** (IEEE TIFS 2025; preprint 10/2023) [Web]
- Human-cognition-inspired audio-visual transformer ensemble; nhiều expert khai thác các tín hiệu khác nhau. IEEE Xplore 10938399.

**3.3. AVT²-DWF** — arXiv **2403.14974** [Web] **[có code]**
- Audio-Visual dual Transformers + **Dynamic Weight Fusion**; dual-stage: spatial + temporal. Repo: https://github.com/raining-dev/AVT2-DWF

**3.4. X-AVDT** — arXiv **2603.08483** (CVPR 2026, KAIST Visual Media Lab) [Web] **[có code + dataset]**
- Audio-Visual Cross-Attention; nhìn từ phía generator: cross-attention nội bộ của model sinh mã hoá alignment speech–motion → tận dụng làm tín hiệu phát hiện. Repo: https://github.com/youngseo0526/X-AVDT

**3.5. Explicit Correlation Learning** — arXiv **2404.19171** [Web]
- Học tường minh tương quan chéo giữa modality để generalizable với cross-modal deepfake.

**3.6. Contextual Cross-Modal Attention** — arXiv **2408.01532** [Web]
- Attention chéo theo ngữ cảnh; detection + localization.

**3.7. DiMoDif** — arXiv **2411.10193** [Web]
- Discourse Modality-information Differentiation; detection + localization AV.

**3.8. Self-supervised representations for AV deepfake detection** — arXiv **2511.17181** [Web]
- Nghiên cứu có hệ thống biểu diễn tự-supervised (vision + speech) cho bài AV detection — gần với thiết kế "DINOv3 + audio branch" mà chúng ta đang định làm (xem `outputs/research-gaps.md` Gap G).

**3.9–3.12. Các phương pháp 2025 (preprint):**
- **ERF-BA-TFD+** — arXiv **2508.17282** [Web]: enhanced receptive field + fusion.
- **HOLA** — arXiv **2507.22781** [Web]: giải pháp video-level track của 1M-Deepfakes Detection Challenge 2025.
- **FauForensics** — arXiv **2505.08294** [Web]: dùng Facial Action Units (sinh học, bất biến) + AV.
- **Forgery-aware Audio-Visual Adaptation** — arXiv **2511.19080** [Web]: Variational Bayesian, tăng generalization đa dataset.

**3.13. AVoiD-DF** — IEEE TIFS 2023 (Xplore 10081373) [Web] **[code 1 phần]**
- Audio-Visual Joint Learning khai thác AV inconsistency. ⚠️ Tác giả ghi rõ *"due to protocol restrictions we cannot release the complete source code and models"* → chỉ mở 1 phần; repo: https://github.com/SYSU-DISG/AVoiD-DF

**3.14. Cross-Modality & Within-Modality Regularization** — arXiv **2401.05746** [Web]
- Regularization giữ sự phân biệt modality (cross + within) khi học biểu diễn chung — trực tiếp liên quan vấn đề "2 branch triệt tiêu nhau".

**3.15. MIS-AVoiDD** — arXiv **2310.02234** [Web]
- Modality-Invariant + Modality-Specific representations cho AV detection.

---

## 4. Nhóm CONSISTENCY / MISMATCH (tận dụng lệch môi–giọng, lệch thời gian)

**4.1. Voice-Face Homogeneity Tells Deepfake** — arXiv **2203.02195** (ACM TOMM 2024) [Web]
- Đo độ thuần nhất giữa giọng và khuôn mặt làm tín hiệu forgery; hướng đến generalizability.

**4.2. Lips Are Lying** — arXiv **2401.15668** (NeurIPS 2024) [Web]
- Lệch **thời gian** giữa môi và audio trong lip-sync forgery; mạnh với loại fake "không đổi danh tính, không có artifact nhìn thấy" (thách thức detector hiện tại).

**4.3. Lost in Translation: Lip-Sync Deepfake Detection from Audio-Video Mismatch** — CVPR 2024 Workshops (WMF), Bohacek & Farid, pp. 4315–4323 [Web]
- Phát hiện lip-sync qua audio–video mismatch; có PDF mở: https://openaccess.thecvf.com/content/CVPR2024W/WMF/papers/Bohacek_Lost_in_Translation_Lip-Sync_Deepfake_Detection_from_Audio-Video_Mismatch_CVPRW_2024_paper.pdf

**4.4. LIPINC** — arXiv **2401.10113** [Web] **[có code]**
- LIP-syncing detection via mouth INConsistency: bất nhất vùng miệng theo thời gian. Repo: https://github.com/skrantidatta/LIPINC

**4.5. Fine-Grained Inconsistencies** — arXiv **2408.06753** (BMVC 2024) [Web]
- Bắt artifact **tinh** (spatial + temporal) mà các phương pháp high-level bỏ sót.

**4.6. ICS-AV: Intra-modal and Cross-modal Synchronization** — ICCV 2025 (Anshul, Gopal, Rajan, Chng — NTU), pp. 13826–13836 [Web] **[có code]**
- Đồng bộ intra-modal + cross-modal; detection **và temporal localization**. Repo: https://github.com/AshutoshAnshul/ics-av-deepfake ; PDF: https://openaccess.thecvf.com/content/ICCV2025/papers/Anshul_Intra-modal_and_Cross-modal_Synchronization_for_Audio-visual_Deepfake_Detection_and_Temporal_ICCV_2025_paper.pdf

---

## 5. Nhóm IDENTITY VERIFICATION

**5.1. POI-Forensics: Audio-Visual Person-of-Interest DeepFake Detection** — arXiv **2204.03083** (CVPR 2023 Workshops, Cozzolino et al.) [Web]
- Contrastive learning: embedding của reference video gần với embedding cùng chủ thể, xa chủ thể khác → phát hiện deepfake bằng **kiểm chứng danh tính** âm–hình. Gần với bài toán "người nói có đúng là người trong video không".

---

## 6. Nhóm DETECTION + TEMPORAL LOCALIZATION (tìm đoạn giả)

- **LAV-DF** — arXiv **2204.06228** (DICTA 2022 Best Award; bản mở rộng "Glitch in the Matrix" **2305.01979**) [Web]: dataset + phương pháp, temporal forgery localization AV (audio fake xen thật, video fake xen thật).
- **Contextual Cross-Modal Attention** (2408.01532), **DiMoDif** (2411.10193), **ICS-AV** (ICCV 2025) — đều có localization (xem trên).

---

## 7. Survey (đọc trước khi vào chi tiết)

| Survey | arXiv | Ghi chú |
|---|---|---|
| Understanding Audiovisual Deepfake Detection: Techniques, Challenges, and Human-machine Alignment | **2411.07650** | Survey chuyên về AV deepfake detection — điểm khởi đầu tốt nhất |
| Passive Deepfake Detection Across Multi-modalities | **2411.17911** | Toàn diện multi-modality (AV + text) |
| Evolving from Single-modal to Multi-modal Facial Deepfake Detection | **2406.06965** | Lộ trình single → multi-modal |

---

## 8. Hiệu chỉnh quan trọng (tránh trích dẫn nhầm)

| Cái tên hay bị nhầm | Sự thật (đã xác minh) |
|---|---|
| **HAMMER** (2304.02556) | Là **ảnh + text** ("Detecting and Grounding Multi-Modal Media Manipulation", CVPR 2023) — **KHÔNG phải** video+audio |
| **DefakeHop** (2103.06929) | ICME 2021, trích đặc trưng từ **ảnh mặt** (SSL/Saab) — visual-only, không phải AV |
| **HM-Conformer** (2309.08208) | Audio-only (Conformer cho ADD) — không phải AV |
| **WaveFake / ASVspoof / MLAAD / In-the-Wild** | Dataset **audio-only** — đừng gắn nhãn "audio-visual" |

---

## 9. Gợi ý cho project (kế thừa quyết định trước)

- **Baseline AV đề xuất cho Gap G** (xem `outputs/research-gaps.md`): giữ cặp matched **DINOv3 ViT-S+/16 vs ConvNeXt-Tiny** làm visual branch + branch audio (Wav2Vec2 2.0 / AST) + fusion; so sánh visual-only / audio-only / fused trên **FakeAVCeleb 4-way** và DFDC subset — đúng protocol cross-dataset kiểu AVFF.
- Các phương pháp **có code sẵn để đối chiếu (baseline thứ cấp)**: AVT²-DWF, LIPINC, ICS-AV, X-AVDT (nặng nhất, CVPR 2026).
- **Chỗ trống rõ ràng** (khớp Gap A–G): chưa thấy phương pháp nào so sánh **CNN vs ViT cùng kích thước, cùng pretraining** trong khung AV; hầu hết chỉ báo số riêng lẻ trên dataset khác nhau → vẫn là khoảng trống để project của anh tạo số riêng.

---

## Nguồn (URLs)

- AVFF: https://arxiv.org/abs/2406.02951 ; supplemental https://openaccess.thecvf.com/content/CVPR2024/supplemental/Oorloff_AVFF_Audio-Visual_Feature_CVPR_2024_supplemental.pdf ; openaccess https://openaccess.thecvf.com/content/CVPR2024/html/Oorloff_AVFF_Audio-Visual_Feature_Fusion_for_Video_Deepfake_Detection_CVPR_2024_paper.html
- AVTENet: https://arxiv.org/abs/2310.13103 ; https://ieeexplore.ieee.org/document/10938399
- AVT²-DWF: https://arxiv.org/abs/2403.14974 ; https://github.com/raining-dev/AVT2-DWF
- X-AVDT: https://arxiv.org/abs/2603.08483 ; https://github.com/youngseo0526/X-AVDT ; https://openaccess.thecvf.com/content/CVPR2026/html/Kim_X-AVDT_Audio-Visual_Cross-Attention_for_Robust_Deepfake_Detection_CVPR_2026_paper.html
- Explicit Correlation Learning: https://arxiv.org/abs/2404.19171
- Contextual Cross-Modal Attention: https://arxiv.org/abs/2408.01532
- DiMoDif: https://arxiv.org/abs/2411.10193
- Self-supervised AV reps: https://arxiv.org/abs/2511.17181
- ERF-BA-TFD+: https://arxiv.org/abs/2508.17282 ; HOLA: https://arxiv.org/abs/2507.22781 ; FauForensics: https://arxiv.org/abs/2505.08294 ; Forgery-aware AV Adaptation: https://arxiv.org/abs/2511.19080
- AVoiD-DF: https://ieeexplore.ieee.org/document/10081373 ; https://github.com/SYSU-DISG/AVoiD-DF
- Cross-/Within-Modality Regularization: https://arxiv.org/abs/2401.05746 ; MIS-AVoiDD: https://arxiv.org/abs/2310.02234
- Voice-Face Homogeneity: https://arxiv.org/abs/2203.02195 ; dblp https://dblp.org/rec/journals/tomccap/ChengGWLCN24.html
- Lips Are Lying: https://arxiv.org/abs/2401.15668 ; NeurIPS https://proceedings.neurips.cc/paper_files/paper/2024/file/a5a5b0ff87c59172a13342d428b1e033-Paper-Conference.pdf
- Lost in Translation: https://openaccess.thecvf.com/content/CVPR2024W/WMF/html/Bohacek_Lost_in_Translation_Lip-Sync_Deepfake_Detection_from_Audio-Video_Mismatch_CVPRW_2024_paper.html
- LIPINC: https://arxiv.org/abs/2401.10113 ; https://github.com/skrantidatta/LIPINC
- Fine-Grained Inconsistencies: https://arxiv.org/abs/2408.06753 ; BMVC https://bmvc2024.org/proceedings/695/
- ICS-AV: https://openaccess.thecvf.com/content/ICCV2025/html/Anshul_Intra-modal_and_Cross-modal_Synchronization_for_Audio-visual_Deepfake_Detection_and_Temporal_ICCV_2025_paper.html ; https://github.com/AshutoshAnshul/ics-av-deepfake
- POI-Forensics: https://arxiv.org/abs/2204.03083 ; https://openaccess.thecvf.com/content/CVPR2023W/WMF/papers/Cozzolino_Audio-Visual_Person-of-Interest_DeepFake_Detection_CVPRW_2023_paper.pdf
- LAV-DF: https://arxiv.org/abs/2204.06228 ; https://arxiv.org/abs/2305.01979
- Surveys: https://arxiv.org/abs/2411.07650 ; https://arxiv.org/abs/2411.17911 ; https://arxiv.org/abs/2406.06965
- HAMMER (để hiệu chỉnh): https://arxiv.org/abs/2304.02556 ; DefakeHop: https://arxiv.org/abs/2103.06929 ; HM-Conformer: https://arxiv.org/abs/2309.08208

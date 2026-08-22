# Khoảng trống nghiên cứu (Research Gaps) — Deepfake Detection

*Ngày: 2026-06 (mùa xuân); nguồn: bộ bài đã đọc trong `deepfake-detection-reading-list.md` + `model-and-dataset-selection.md`, bổ sung kiểm tra web 2024–2025.*
*Nguyên tắc: mọi số đều trích từ bài báo đã xác minh (ghi rõ). Mức độ xác minh: [Đã đọc] = đã đọc trực tiếp / [Web] = xác minh qua tìm kiếm.*

---

## Gap A — Nghiên cứu đối chứng kiến trúc CNN vs ViT với pretrain "matched" (phương pháp luận)

**Bằng chứng khoảng trống:**
- Mọi so sánh CNN vs ViT hiện có đều **nhiễu bởi pretrain khác nhau**: CViT [2102.11126] pretrain ImageNet; M2TR [2104.09770] dùng backbone EfficientNet-B4 (ImageNet) + ViT đa tỉ lệ; GenConViT [2307.07036] ViT-B/16 ImageNet; ICT [2203.01318] train *từ đầu* trên MS-Celeb-1M (khuôn mặt) — không con nào cùng dữ liệu pretrain.
- Bài FF++ gốc [1901.08971] so sánh **chỉ CNN** (Xception, ResNet-50, ... tất cả ImageNet-1k → matched cho CNN-CNN, nhưng không có ViT nào trong bảng gốc).
- Survey ViT [2405.08463] benchmark 5 model mã nguồn mở nhưng **không kiểm soát pretrain** (mỗi model một nguồn weights khác nhau).
- **Chưa có paper nào** so CNN vs ViT với: cùng dữ liệu pretrain + cùng recipe SSL + cùng teacher + cùng pipeline đánh giá. → Câu hỏi "ViT tốt hơn CNN vì kiến trúc, hay vì pretrain to hơn?" **vẫn chưa được trả lời một cách sạch.**

**Cách khai thác (khả thi với setup hiện tại):**
- Cặp chính: **DINOv3-ViT-S+/16 (29M) vs DINOv3-ConvNeXt-Tiny (29M)** — cùng LVD-1689M, cùng recipe (DINO+iBOT+KoLeo+Gram), cùng teacher ViT-7B, khớp param (mục 6.2, đã xác minh).
- Cặp đối chứng thứ hai (cổ điển, để nối với literature): **Xception vs ViT-B/16** — cả hai ImageNet-1k.
- Giao thức: FF++ c23 train (split 720/140/140), test c23/c40/RAW + Celeb-DF v2 cross-domain, frame-level + video-level (majority vote), ACC + AUC, 3 seeds (đã thiết kế trong mục 6.3).
- **Kết quả mới mà bộ tài liệu chưa có:** (1) bảng DINOv3 đầu tiên trên FF++/Celeb-DF; (2) tách được hiệu ứng "kiến trúc" vs "dữ liệu pretrain".

**Giá trị:** bài phương pháp luận/benchmark; trả lời câu hỏi mở lâu năm của lĩnh vực. Độ khó: thấp–trung bình. Rủi ro: kết quả có thể là "không có khác biệt lớn" — nhưng bản thân kết luận đó vẫn đáng công bố.

---

## Gap B — Testbed generalization trên generator hiện đại (AIGC/diffusion) — era gap

**Bằng chứng khoảng trống:**
- Generalization sụp là vấn đề được ghi nhận nhiều nhất trong tài liệu: Celeb-DF [1909.12962] — AUC trung bình 9 detector = **56.9%**; survey [2001.00179] — AUC <60% trên Celeb-DF; CViT [2102.11126] — FaceShifter chỉ **46%** (trong khi DFDC 91.5%); M2TR FF++→Celeb-DF **68.2 AUC**; UCF [2304.13949] — Xception cross AUC 0.672.
- Nhưng **tất cả các số trên đều từ thế hệ GAN cũ (2018–2021)**. Benchmark 2024–2025 xác nhận detectors cũ không theo kịp generator mới:
  - **DF40** [arXiv:2406.13495, NeurIPS 2024] — 40 kỹ thuật (10 face-swap, 12 reenactment, 10 full-face AIGC, 5 editing), 2000+ đánh giá → detectors fail trên EFS/talking-head/diffusion [Web].
  - **GenVidBench** [2501.11340] — phát hiện video AI-generated [Web].
  - **Celeb-DF++** [2507.18015] — benchmark cho generalizable forensics [Web].
  - **Deepfake-Eval-2024** [2503.02857] — deepfake thật lan truyền trên mạng xã hội năm 2024 [Web].
  - **DigiFakeAV** [2505.16512] — digital human dùng diffusion, multimodal [Web].
- **Chưa có paper nào report DINOv3 trên FF++/Celeb-DF/DFDC/DF40** (đã ghi trong FAQ 5.1); 2511.22471 [Web] mới test frozen DINOv3 trên *image* forgery cross-generator — **chưa ai làm cho face deepfake video**.

**Cách khai thác:**
- Dùng chính 2 model của Gap A (ViT-S+ vs ConvNeXt-Tiny) → thêm cột test: **DF40 subset** (tải qua Google form) hoặc GenVidBench + dữ liệu diffusion tự sinh (công cụ mã nguồn mở hiện có).
- 2 chế độ: frozen + linear probe (theo khuyến nghị DINOv3) vs full fine-tune — so xem foundation-model features có "thắng" cả trên generator cũ lẫn mới không.
- **Kết quả mới:** bảng cross-generator đầu tiên cho cặp matched CNN/ViT trên generator 2024–2025.

**Giá trị:** cao nhất về tính thời sự; đúng hướng "model drift" mà tài liệu liệt kê là câu hỏi mở. Độ khó: trung bình (chủ yếu eval; cần tải DF40, có thể phải xin phép). Rủi ro: cạnh tranh tăng nhanh (2025 có nhiều bài theo hướng này).

---

## Gap C — Robustness với chuỗi nén thực tế (không chỉ c40)

**Bằng chứng khoảng trống:**
- Nén làm sụp detector: GenConViT [2307.07036] **97.68% RAW → 48.56% LQ**; M2TR [2104.09770] bền nhất survey nhưng vẫn tụt (LQ ACC 87.19% / AUC 0.904).
- Nhưng FF++ c40 chỉ là **proxy thô** cho nén thực tế (re-encode nhiều vòng qua nền tảng mạng xã hội). DeeperForensics-1.0 có bộ 7 perturbation nhưng ít bài dùng (ICT, RealForensics dùng).
- Chưa có nghiên cứu hệ thống: **cùng model, cùng pipeline, so sánh CNN vs ViT trên chuỗi re-encode thực tế** (H.264/H.265 bitrate thấp, 2–3 vòng re-encode kiểu platform).

**Cách khai thác:** thêm bước FFmpeg vào pipeline của Gap A (đã có sẵn thiết kế); chạy trên 3 mức nén + chuỗi re-encode; đo cả frame/video-level. Kết quả: bảng "ai mất ít điểm hơn khi nén" — câu trả lời thực dụng cho triển khai. Độ khó: thấp (chỉ thêm transform). Rủi ro: thấp.

---

## Gap D — Bias theo sắc tộc / dữ liệu khuôn mặt Việt (equity gap)

**Bằng chứng khoảng trống:**
- Toàn bộ dataset chuẩn lấy người nổi tiếng phương Tây / diễn viên Mỹ (FF++ YouTube, DFDC 3.426 diễn viên đồng thuận, Celeb-DF).
- Chính MODEL_CARD DINOv3 [2508.10104] thừa nhận **tụt điểm ở nhóm thu nhập thấp / chênh lệch vùng miền** (bias section) — nhưng **chưa có bài deepfake nào trong bộ tài liệu đo bias theo sắc tộc** cho detector mặt.
- Không thấy dataset/đánh giá deepfake mặt Việt trong tài liệu đã đọc [cần kiểm tra thêm trước khi khẳng định "chưa có ai làm"].

**Cách khai thác:** đánh giá ROC theo nhóm (khuôn mặt Á vs Âu) của chính 2 model Gap A; xây bộ test nhỏ khuôn mặt Việt (thu thập có đồng thuận + tự sinh fake bằng công cụ mã nguồn mở). Ý nghĩa xã hội rõ với bối cảnh lừa đảo deepfake. Độ khó: trung bình (cần thu thập dữ liệu + cân nhắc đạo đức/đồng thuận). Rủi ro: chậm hơn các gap khác vì khâu dữ liệu.

---

## Gap E — Calibration + loại bỏ unknown generator (open-set có "kiểm soát rủi ro")

**Bằng chứng khoảng trống:**
- ICT [2203.01318] là bài open-set hiếm hoi (open-set AUC trung bình 87.01, ICT-Ref 96.34) — nhưng **không release training code** (đã xác minh README) → không tái lập được; và **không có phân tích calibration** (ECE, rejection threshold).
- Không bài nào trong bộ tài liệu báo **Expected Calibration Error (ECE)** hoặc đường cong "reject unknown" cho detector.

**Cách khai thác:** thêm metric calibration vào protocol Gap A (temperature scaling, ECE, precision-recall khi threshold theo phần trăm reject) — phép đo rẻ, bổ sung phần "deployment readiness" mà literature bỏ trống. Độ khó: thấp. Rủi ro: thấp.

---

## Gap F — Kết hợp tín hiệu sinh học (nhịp tim) + kiến trúc hiện đại (deep dive)

**Bằng chứng khoảng trống:**
- DeepRhythm [2006.07634] chứng minh nhịp tim (rPPG) phát hiện deepfake với AUC cao — nhưng trên generator cũ, điều kiện gần như không nén; chưa ai kết hợp temporal rhythm + spatial ViT hiện đại trong điều kiện nén/generator mới.

**Cách khai thác:** fusion spatial (ViT-S+) + temporal (DeepRhythm-style) — cần compute nhiều hơn, pipeline phức tạp hơn. Độ khó: cao. Rủi ro: cao (tín hiệu rPPG nhạy với nén). Chỉ nên làm sau Gap A.

---

## Khuyến nghị chiến lược

| Gap | Tính mới | Khả thi với setup hiện tại | Chi phí | Rủi ro | Đề xuất |
|---|---|---|---|---|---|
| **A — Kiến trúc matched** | Cao (chưa ai làm sạch) | **Rất cao** (protocol đã thiết kế xong) | Thấp | Thấp | ⭐ **Làm trước** |
| **B — AIGC cross-generator** | Rất cao | Cao (chủ yếu eval) | Trung bình | Trung bình | ⭐ Kết hợp ngay với A |
| **C — Chuỗi nén thực tế** | Trung bình | Rất cao (chỉ thêm FFmpeg) | Thấp | Thấp | Thêm vào A |
| **D — Bias sắc tộc/VN** | Cao (độc đáo) | Trung bình (cần thu thập dữ liệu) | Trung bình | Trung bình | Luận văn / mở rộng |
| **E — Calibration** | Trung bình | Rất cao | Thấp | Thấp | Thêm metric vào A |
| **F — Sinh học + ViT** | Trung bình | Thấp (compute) | Cao | Cao | Sau cùng |

**Lộ trình đề xuất:** A + B + C + E trong một bài: "CNN vs ViT với pretrain matched: ai bền với generator mới, nén thực tế, và khi cần từ chối unknown?" — chạy trên chính bộ protocol tự-test của bạn, số liệu 100% tự sinh.

---

## Nguồn

- Bộ bài đã xác minh: `deepfake-detection-reading-list.md`, `model-and-dataset-selection.md` (đầy đủ arXiv ID bên trong).
- DF40: https://arxiv.org/abs/2406.13495 (NeurIPS 2024), repo https://github.com/YZY-stack/DF40 [Web]
- GenVidBench: https://arxiv.org/abs/2501.11340 [Web]
- Celeb-DF++: https://arxiv.org/abs/2507.18015 [Web]
- Deepfake-Eval-2024: https://arxiv.org/abs/2503.02857 [Web]
- DigiFakeAV: https://arxiv.org/abs/2505.16512 [Web]
- DINOv3 MODEL_CARD (bias section): https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md

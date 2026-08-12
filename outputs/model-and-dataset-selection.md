# Lựa chọn Model & Dataset: Deepfake Detection (CNN vs ViT)

**Ngày tạo:** 2025
**Mục đích:** Chọn 1 model CNN, 1 model ViT và danh sách dataset đáng tin cậy (được nhiều bài báo dùng) cho dự án phát hiện deepfake.
**Cách xác minh:** Mọi thông tin dataset×bài báo bên dưới đều đã được kiểm tra trực tiếp trong **toàn văn** các bài báo qua alphaXiv (`alpha_ask_paper`), không dựa trên trích dẫn thứ cấp. Mỗi con số kèm nguồn bài báo gốc.

---

## 1. Model CNN được khuyến nghị: **XceptionNet (fine-tuned)**

**Nguồn gốc:** Xception — F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions", CVPR 2017. Được FaceForensics++ (Rössler et al., ICCV 2019, arXiv:1901.08971) đưa vào làm **baseline chuẩn** cho bài toán phát hiện giả mạo mặt.

**Vì sao chọn Xception:**
- Là **baseline bắt buộc** trong gần như mọi bài báo deepfake detection 2019–2025 — chọn nó giúp bạn so sánh kết quả với hầu hết literature mà không cần giải thích thêm.
- Kiến trúc thuần CNN, nhẹ, fine-tune nhanh, có pretrain ImageNet — phù hợp cho đồ án/khởi điểm thực nghiệm.

**Kết quả tham chiếu (đã xác minh):**
| Nguồn | Kết quả |
|---|---|
| FF++ paper (1901.08971), video-level + face tracking | 99.26% (RAW), 95.73% (HQ), 81.00% (LQ) |
| F3-Net bảng kết quả (2007.09355), frame-level | FF++ LQ: ACC 86.86% / AUC 0.893; HQ: 95.73% / 0.963; RAW: 99.26% / 0.992 |

**Các bài báo dùng Xception (đã xác minh trong toàn văn):**
- **F3-Net** (2007.09355): Xception là backbone của model đề xuất.
- **M2TR** (2104.09770): so sánh backbone Xception vs EfficientNet-B4; Xception cũng là baseline trong bảng Celeb-DF (AUC 97.6), ForgeryNet (80.78 2-way), cross-dataset.
- **UCF** (2304.13949): Xception là backbone + baseline chính (FF++→CelebDF AUC 0.672; bản UCF nâng lên 0.824).
- **ICT** (2203.01318): baseline Xcep-c0 (FF++ AUC 99.26) và Xcep-c23 (95.60) trong bảng open-set.
- **RealForensics** (2201.07131): baseline chính trong mọi bảng (FF++ in-dist 99.0% raw / 97.0% c23; cross-dataset avg 75.3).
- **GenConViT** (2307.07036): trong bảng so sánh DFDC/FF++.
- **Tolosana survey** (2001.00179), **ViT survey** (2405.08463): trích dẫn baseline Xception.

> **Tùy chọn mạnh hơn (nếu muốn CNN hiện đại):** EfficientNet-B4 (backbone của M2TR, 2104.09770) hoặc EfficientNet-B7 (giải DFDC của Selim, 97.2% accuracy — được GenConViT trích dẫn). Nhưng để so sánh công bằng với literature, **Xception vẫn là lựa chọn mặc định**.

---

## 2. Model ViT được khuyến nghị: **ICT — Identity Consistency Transformer**

**Nguồn gốc:** X. Dong, J. Bao, D. Chen, et al. (Microsoft Research Asia + USTC), "Protecting Celebrities from DeepFake with Identity Consistency Transformer", arXiv 2203.01318 (2022). Code: `github.com/LightDXY/ICT_DeepFake`.

> **⚠️ Cập nhật quan trọng theo tiêu chí "tự test lại" (chi tiết ở mục 6):** ICT **có** pretrained weights (`ICT_Base.pth` + `ref.pkl` — link tải đã kiểm tra hoạt động, HTTP 200) nhưng repo **chỉ phát hành inference code, KHÔNG có training code** (README TODO: "[x] Release inference code. [ ] Release training code") và yêu cầu dependency cũ (`timm==0.3.4`, `torch>=1.4`, RetinaFace align riêng) → **không thể tự huấn luyện lại**, chỉ dùng được ở chế độ inference làm mốc literature. Nếu cần model ViT tự train được với pretrain "công bằng" với CNN → dùng **ViT-B/16 (timm, ImageNet-1k)** hoặc **DINOv2-ViT-B** (Apache 2.0).

**Vì sao chọn ICT:**
- Là **ViT thuần** (standalone): 12 blocks, 12 attention heads, patch 14×14, input 112×112, train từ đầu trên MS-Celeb-1M — không có CNN backbone, đúng nghĩa "1 model ViT".
- Khác các ViT dựa trên artifact pixel: ICT học **high-level semantics** (độ nhất quán danh tính giữa vùng mặt trong/ngoài) → **generalization tốt nhất lớp standalone ViT** trong survey ViT (2405.08463), bền với suy giảm ảnh (nén, noise, blur).
- Code chính thức công khai.

**Kết quả tham chiếu (open-set AUC, train MS-Celeb-1M, test dataset chưa từng thấy — bảng 1, 2203.01318):**
| Dataset | ICT | ICT-Ref |
|---|---|---|
| DFD (Google) | 84.13 | 93.17 |
| FF++ | 90.22 | 98.56 |
| DeeperForensics | 93.57 | 99.25 |
| Celeb-DF v1 | 81.43 | 96.41 |
| Celeb-DF v2 | 85.71 | 94.43 |
| **Trung bình** | **87.01** | **96.34** |

So với baseline low-level tốt nhất cùng bảng (Face X-ray avg 79.16; Xcep-c23 79.56) → ICT vượt ~8 điểm AUC trung bình. Đặc biệt: video deepfake thật trên YouTube ("Ctrl Shift Face"): ICT 94.36% AUC, ICT-Ref 100%.

**Hạn chế cần biết:** ICT chỉ phát hiện **face swap** (lệch danh tính trong/ngoài); không bắt được reenactment giữ nguyên danh tính (tự bài báo thừa nhận ở Limitations).

> **Tùy chọn thay thế theo mục tiêu:**
> - Muốn model ViT **được trích dẫn nhiều nhất** cho deepfake → **CViT** (2102.11126, CNN→ViT, 91.5% DFDC) — nhưng đây là hybrid CNN+ViT, không phải ViT thuần.
> - Muốn ViT **mạnh nhất theo benchmark survey** (2405.08463) → **M2TR** (2104.09770, ViT đa tỉ lệ + nhánh frequency, FF++ LQ ACC 92.89/AUC 95.31; Celeb-DF in-set AUC 99.8) — hybrid với backbone EfficientNet-B4.
> - Muốn ViT "sách giáo khoa" → fine-tune **ViT-B/16** (Dosovitskiy et al., ICLR 2021) với pipeline tiền xử lý giống CViT.
> - Muốn ViT foundation model (mới 08/2025) → **DINOv3-ViT** (bản ViT là ViT thuần) — xem FAQ 5.1: mạnh cross-generator nhưng license thương mại, chưa có số chuẩn trên FF++/Celeb-DF, và pretrain 1,7 tỷ ảnh làm lệch phép so sánh CNN vs ViT. **Cách hóa giải lệch pretrain: dùng cặp DINOv3-ViT-S+/16 (29M) vs DINOv3-ConvNeXt-Tiny (29M)** — cùng 1,7 tỷ ảnh, cùng recipe SSL, cùng teacher ViT-7B, khớp cỡ param, chỉ khác kiến trúc (mục 6.2). Lưu ý: ViT-S/16 (21M) KHÔNG khớp cỡ với ConvNeXt-Tiny (29M).

---

## 3. Danh sách dataset đáng tin cậy × các bài báo sử dụng

Bảng dưới đây chỉ liệt kê các dataset **có trong benchmark thực nghiệm** của các bài đã kiểm tra toàn văn (không tính "chỉ nhắc đến").

### 3.1 Nhóm "kinh điển" — nên dùng (được gần như mọi bài dùng)

| Dataset | Năm | Quy mô (đã xác minh) | Các bài báo đã dùng (kiểm tra toàn văn) |
|---|---|---|---|
| **FaceForensics++** (1901.08971) | 2019 | 1.000 video thật + 4.000 fake (4 phương pháp: DeepFakes, Face2Face, FaceSwap, NeuralTextures); 3 mức nén RAW/HQ/LQ; **⚠️ không có audio** (README chính thức: "downloaded the source video without audio"; audio YouTube phải tự tải lại — và 4 phương pháp không sửa audio nên audio không phân biệt được fake/thật) | Bài gốc FF++; **CViT** (test 4 nhóm); **M2TR** (RAW/HQ/LQ + SR-DF); **F3-Net** (RAW/HQ/LQ); **GenConViT** (ACC 97.0); **ICT** (test open-set); **UCF** (train HQ, cross-test); **RealForensics** (train + in-dist raw/c23/c40); survey ViT (benchmark 5 model) |
| **Celeb-DF (v2)** (1909.12962) | 2020 | 590 video thật + 5.639 deepfake (face swap chất lượng cao) | Bài gốc (9 detector, AUC trung bình chỉ 56.9%); **M2TR** (in-set AUC 99.8; cross FF++→Celeb-DF 68.2); **GenConViT** (ACC 90.94); **ICT** (CD1 + CD2 open-set); **UCF** (cross AUC 0.824); **RealForensics** (cross AUC 86.9) |
| **DFDC** (2006.07397) | 2020 | 128.154 clip: 23.654 thật + 104.500 fake (8 kỹ thuật, 3.426 diễn viên có đồng thuận); ~470GB bản đầy đủ; **có audio** (clip quay thực tế — dataset chuẩn cho hướng audio-visual: AVFF CVPR 2024, ACM MM 2020) | Bài gốc (challenge, best AP 0.753 in-the-wild); **CViT** (train + test, 91.5% ACC); **GenConViT** (train + test, ACC 98.5); **UCF** (cross-test AUC 0.805); **RealForensics** (cross-test AUC 75.9) |

### 3.2 Nhóm bổ sung (dùng cho test generalization / chuyên sâu)

| Dataset | Năm | Quy mô (đã xác minh) | Các bài báo đã dùng |
|---|---|---|---|
| **DFD (Google Deepfake Detection)** | 2019 | 363 thật + 3.068 fake | **ICT** (AUC 84.13), **UCF** (cross AUC 0.945), M2TR (bảng so sánh dataset) |
| **DeeperForensics-1.0** | 2020 | 1.000 fake dựa trên FF++ + bộ 7 loại perturbation (saturation, noise, blur, pixelation, nén...) | **ICT** (AUC 93.57), **RealForensics** (cross AUC 99.3; dùng bộ perturbation làm protocol robustness) |
| **ForgeryNet** | 2021 | 99.630 thật + 121.617 fake, 15 phương pháp, 36 loại distortion | **M2TR** (2-way 82.52 / 3-way 75.12 / 16-way 69.12), **RealForensics** (generalization AUC 71.8) |
| **UADFV** | 2018 | 49 thật + 49 fake | **CViT** (AUC 93.75), **MesoNet** (AUC 84.3), M2TR (bảng so sánh) |
| **DF-TIMIT (DeepfakeTIMIT)** | 2018 | 320 cặp video (LQ + HQ) | **GenConViT** (ACC 98.28), M2TR (bảng so sánh) |
| **WildDeepfake** | 2020 | 3.805 thật + 3.509 fake (từ Internet) | M2TR (chỉ trong bảng so sánh dataset, không phải benchmark chính) |
| **SR-DF** | 2022 | 1.000 thật + 4.000 fake (4 phương pháp hiện đại, post-process DoveNet) | **M2TR** (tự giới thiệu; Celeb-DF-level quality, khó hơn: avg AUC 86.7 so với 95.5 của Celeb-DF) |

### 3.3 Dataset phụ trợ (không phải benchmark, dùng cho training/pretrain)

| Dataset | Quy mô | Dùng cho |
|---|---|---|
| **MS-Celeb-1M** | 10M ảnh / 1M danh tính | ICT (train identity consistency, không cần fake data) |
| **LRW** (Oxford) | 500k video khuôn mặt nói chuyện | RealForensics (nguồn video thật tự nhiên, self-supervised stage 1) |
| **VoxCeleb2** | ~1M video | RealForensics (ablation: thay LRW vẫn đạt AUC 82.9→98.8) |

---

## 4. Combo đề xuất cho đồ án/thực nghiệm

**Cấu hình tối giản và đúng chuẩn literature (giống protocol M2TR / UCF / RealForensics):**

```
Huấn luyện:      FF++ (bản HQ/c23)         ← dataset chuẩn nhất để train
Test in-domain:  FF++ (RAW, HQ, LQ)         ← đánh giá độ bền với nén
Test cross-domain: Celeb-DF (v2)            ← metric quan trọng nhất: generalization
Mở rộng (nếu đủ tài nguyên): DFDC (train) hoặc chỉ test trên DFDC
Baseline:        Xception (c23) + CViT      ← bắt buộc để so sánh công bằng
Model chính:     Xception (CNN) vs ICT (ViT)
Metric:          ACC + AUC (frame-level); cross-dataset AUC (train FF++ → test Celeb-DF)
Tiền xử lý:      RetinaFace/MTCNN + crop mặt 1.3×, resize 224×224 (CViT) hoặc 256×256 (UCF)
```

**Lưu ý thực tế:**
- Kết quả thay đổi rất mạnh theo mức nén — **luôn ghi rõ c23/c40/RAW** khi báo cáo.
- DFDC bản đầy đủ ~470GB; có thể dùng DFDC Preview (1910.08854) nếu hạn chế tài nguyên.
- Celeb-DF + FF++ là cặp "bắt buộc" trong hầu hết paper 2021–2025 (M2TR, UCF, RealForensics, GenConViT đều dùng cặp này).

---

## 5. Phản biện & làm rõ (FAQ)

### 5.1 Vì sao không chọn DINOv3? DINOv3 có phải ViT thuần?

**Trả lời ngắn:** bản ViT của DINOv3 đúng là ViT thuần (patch embedding + self-attention, không CNN backbone) — nhưng "DINOv3" là **cả gia đình model gồm ViT lẫn ConvNeXt**, và có 4 lý do thực dụng để không đặt nó làm "model ViT chính" cho đồ án so sánh CNN vs ViT:

**Sự thật về DINOv3 (đã xác minh):**
- arXiv:2508.10104, Meta AI, công bố 13/08/2025. SSL trên LVD-1689M (1,7 tỷ ảnh), tối đa 6,7B tham số (ViT-7B). Kỹ thuật mới: Gram anchoring, RoPE 2D, register tokens.
- Bộ 12 model (MODEL_CARD): 5 ViT-S/S+/B/L/H+ (21M–840M, distilled từ ViT-7B) + ViT-7B + **4 ConvNeXt-T/S/B/L** trên dữ liệu web; thêm 2 model dữ liệu vệ tinh. → "DINOv3" ≠ "ViT thuần": nhánh ConvNeXt là kiến trúc conv.
- License: **license thương mại riêng của Meta** (phải điền form access request) — khác DINOv2 (Apache 2.0, cả code lẫn weights).

**4 lý do không chọn làm model chính:**
1. **Không có mốc so sánh literature trên benchmark chuẩn.** Chưa có paper nào report DINOv3 trên FF++/Celeb-DF/DFDC. Các bài dùng DINO family cho forgery mới ra cuối 2025 (2511.22471 — cross-generator *image* forgery, không phải face deepfake video; 2511.12107 — DINOv2+LoRA, AAAI 2026). Số của bạn sẽ không đối chiếu được với bảng kết quả của paper khác.
2. **Nhiễu giữa "kiến trúc" và "dữ liệu pretrain".** DINOv3 học trên 1,7 tỷ ảnh; Xception/ResNet pretrain ImageNet (1,28 triệu). So CNN vs ViT theo cách này đo lệch "lợi thế dữ liệu", không đo thuần kiến trúc. ICT (train từ đầu trên MS-Celeb-1M — dữ liệu khuôn mặt) giữ phép so sánh sạch hơn.
3. **License thương mại** — bất tiện khi công bố code/đồ án; DINOv2 (Apache 2.0) thoải mái hơn nếu bạn nhất định dùng DINO family.
4. **Chi phí:** fine-tune ViT-7B không thực tế; dùng ViT-S/16 (21M) thì lợi thế foundation model giảm nhiều.

**Khi nào DINOv3 là lựa chọn ĐÚNG:**
- Mục tiêu = generalization với generator mới (AIGC/image forgery): 2511.22471 chứng minh **frozen DINOv3 không cần fine-tune** đã generalizes mạnh (dựa vào cấu trúc low-frequency toàn cục thay vì artifact high-frequency theo từng generator).
- Muốn thí nghiệm **CNN vs ViT đối xứng hoàn hảo**: dùng cặp **DINOv3-ViT-S+/16 vs DINOv3-ConvNeXt-Tiny (cả hai 29M)** — cùng 1,7 tỷ ảnh, cùng recipe SSL, cùng teacher, khớp cỡ param (xem mục 6.2). Điều này **giải quyết luôn lo ngại "nhiễu pretrain" ở điểm 2**. Lưu ý: ViT-S/16 (21M) không khớp cỡ với ConvNeXt-Tiny (29M).
- Đồ án muốn đóng góp mới (linear probe / token-ranking / LoRA trên DINOv3) thay vì tái lập baseline literature.
- Nếu chọn DINOv3: dùng **ViT-S/16 (21M)** hoặc **ViT-B/16 (86M)**, fine-tune nhẹ (LoRA/linear probe), và **chạy song song ICT + Xception làm mốc** để vẫn có điểm neo so sánh với literature.

### 5.2 Xception có phải "CNN thuần"?

**Bạn nói đúng một chi tiết quan trọng:** Xception dùng **depthwise separable convolution** (depthwise conv theo từng kênh + pointwise conv 1×1) — Chollet gọi đây là "extreme Inception". Tuy nhiên:
- Depthwise conv vẫn là **phép tích chập**; Xception không chứa bất kỳ module attention/transformer nào → theo phân loại CNN vs ViT, Xception là **CNN 100%**.
- Lý do chọn Xception không phải vì "thuần" mà vì **tính so sánh được**: nó là baseline/backbone của FF++, F3-Net, M2TR, UCF, ICT, RealForensics — bạn có cột đối chiếu trong mọi paper.
- Nếu bạn cần CNN "thuần" theo nghĩa khắt khe (chỉ standard convolution): **ResNet-50** — cũng là baseline trong chính bài FF++ (1901.08971) và nhiều paper khác; vẫn đối chiếu được. (VGG-16 làm khung trong MesoNet.)
- Lưu ý: hầu hết CNN hiện đại (MobileNet, EfficientNet, ConvNeXt — kể cả nhánh ConvNeXt của DINOv3) đều dùng depthwise conv. "CNN thuần" ngày nay gần như đồng nghĩa ResNet/VGG cổ điển; đổi sang ResNet mất chút sức mạnh nhưng giữ được tính "sạch" trong bài viết của bạn.

---

## 6. Kế hoạch tự đánh giá lại (reproducible protocol)

> Bạn nói đúng: nếu làm nghiêm túc thì phải **tự chạy lại trên bộ test của mình**, không tin số trên báo. Dưới đây là ma trận "model nào thực sự chạy được" (đã xác minh trực tiếp 2025-08) và protocol đề xuất.

### 6.1 Ma trận khả dụng (weights / training code / license — đã xác minh)

| Model | Pretrained weights | Training code | License | Thực tế khi chạy |
|---|---|---|---|---|
| **Xception** (timm) | ✓ ImageNet | ✓ (tự viết fine-tune) | Apache-2.0 | Dễ nhất; baseline chuẩn literature |
| **ResNet-50** (timm) | ✓ ImageNet | ✓ | Apache-2.0 | CNN standard-conv (nếu cần "thuần") |
| **ViT-B/16** (timm/HF) | ✓ ImageNet-1k | ✓ | Apache-2.0 | ViT thuần; **matched pretrain với Xception** → so sánh kiến trúc sạch nhất |
| **DINOv2 ViT-B** | ✓ (torch.hub/HF) | ✓ (nhưng đắt: 142M ảnh) | Apache-2.0 | SSL mạnh; lưu ý pretrain lệch với CNN |
| **DINOv3 ViT-B** | ✓ (HF, safetensors 342MB) | ✓ (rất đắt: 1,7 tỷ ảnh) | Meta commercial | Mới 08/2025; license hạn chế |
| **DINOv3 ConvNeXt-T** (29M) | ✓ (HF `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`) | ✓ (rất đắt) | Meta commercial | **CNN cùng recipe với DINOv3-ViT**: cùng LVD-1689M, cùng SSL, cùng teacher ViT-7B |
| **DINOv3 ConvNeXt-S** (50M) | ✓ (HF `facebook/dinov3-convnext-small-pretrain-lvd1689m`) | ✓ (rất đắt) | Meta commercial | Bản to hơn nếu muốn chứng minh scaling |
| **ICT** | ✓ `ICT_Base.pth` + `ref.pkl` (link test OK, HTTP 200) | ✗ **CHỈ inference** | không có LICENSE trong repo | `timm==0.3.4`, `torch>=1.4` cũ; cần RetinaFace align; không tự train lại được |
| **CViT** | ✓ `cvit_*.pth` trong repo + HF mirror | ✓ | không rõ | Hybrid CNN→ViT; train trên DFDC |
| **GenConViT** | ✓ HF `Deressa/GenConViT` (ed+vae, state dict) | ✓ | không rõ | Hybrid ConvNeXt+Swin; script `prediction.py` có preset `[dfdc, faceforensics, timit, celeb]`; VAE ~2,6GB |
| **M2TR** | ✓ Model Zoo (Google Drive) | ✓ (`tools/train.py`) | MIT | Hybrid EfficientNet+transformer; **Model Zoo ghi rõ đánh giá trên subset FF-DF** (FF-C23 97.93 / FF-C40 92.89 / CelebDF 99.76) — không phải full FF++ 4 phương pháp |

**Kết luận quan trọng:** lo ngại của bạn về ICT là có cơ sở — weights thì có, nhưng **không có training code** nên không thể tái lập/retrain; và nhiều bộ weights phát hành (M2TR Model Zoo, GenConViT state dict) chỉ tương ứng subset/protocol riêng, không tái lập số trên báo khi chạy full benchmark.

### 6.2 Cặp model "đối xứng hoàn hảo" cho thí nghiệm (đã xác minh từ MODEL_CARD chính thức)

**Câu trả lời cho "CNN nào train trên nhiều ảnh, cùng cỡ DINOv3 Small": DINOv3-ConvNeXt-Tiny — NHƯNG không khớp cỡ chính xác với ViT-S/16.** Cặp khớp cỡ đúng là **ViT-S+ vs ConvNeXt-Tiny (cả hai 29M)**.

**Đo trực tiếp từ file weights (safetensors, fp32) trên Hugging Face:**

| Model | Tham số đo được (bytes ÷ 4) | Tham số Meta công bố | File model.safetensors |
|---|---|---|---|
| DINOv3 **ViT-S/16** | **≈ 21,6M** | 21M | 86.406.384 bytes |
| DINOv3 **ViT-S+/16** | **≈ 28,7M** | 29M | 114.794.096 bytes |
| DINOv3 **ConvNeXt-Tiny** | **≈ 27,8M** | 29M | 111.299.216 bytes |

→ **ViT-S/16 (21M) vs ConvNeXt-Tiny (29M): lệch ~29–38% — KHÔNG cùng cỡ.**
→ **ViT-S+/16 (29M) vs ConvNeXt-Tiny (29M): khớp cỡ** (lệch <3% theo phép đo; Meta công bố cả hai = 29M).

Meta đã train sẵn **cả ViT lẫn ConvNeXt bằng đúng một recipe** — bảng so sánh cặp khớp cỡ:

| | DINOv3 **ViT-S+/16** (khuyến nghị) | DINOv3 **ConvNeXt-Tiny** | (tùy chọn) DINOv3 ViT-S/16 |
|---|---|---|---|
| Tham số | **29M** | **29M** | 21M |
| Pretrain data | LVD-1689M (1,7 tỷ ảnh) | LVD-1689M (1,7 tỷ ảnh) | LVD-1689M |
| Thuật toán SSL | DINO + iBOT + KoLeo + Gram anchoring | giống hệt | giống hệt |
| Teacher distill | ViT-7B (đông cứng) | **cùng teacher ViT-7B** | cùng teacher |
| Tải | `torch.hub` (`dinov3_vits16plus`) / HF | `torch.hub` (`dinov3_convnext_tiny`) / HF | `torch.hub` (`dinov3_vits16`) |
| Kiến trúc | ViT thuần (patch 16, attention, RoPE, SwiGLU) | **CNN thuần** (conv, không attention) | ViT thuần |

→ Chỉ còn **một biến số duy nhất: kiến trúc (CNN vs ViT)**. Đây là thiết kế thí nghiệm sạch nhất có thể; DINOv2 không có nhánh ConvNeXt (chỉ ViT), nên DINOv3 là gia đình duy nhất hiện có cặp ViT+CNN matched ở quy mô này.

**Nguồn xác minh ViT-S+ tồn tại:** (1) README chính thức github.com/facebookresearch/dinov3 — bảng "ViT-S+/16 distilled, 29M, LVD-1689M"; (2) hubconf.py — hàm `dinov3_vits16plus` có thật; (3) HF repo facebook/dinov3-vits16plus-pretrain-lvd1689m — model.safetensors 114.794.096 bytes (đã liệt kê file). Đọc thêm: https://github.com/facebookresearch/dinov3/blob/main/hubconf.py, https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m

**Hai lưu ý trung thực:**
1. ConvNeXt dùng depthwise conv (7×7 depthwise + 1×1 pointwise) như Xception/EfficientNet — vẫn là CNN (không attention), nhưng không phải "standard convolution" kiểu ResNet. Nếu bạn yêu cầu strict standard-conv thì **không có** CNN nào pretrain ở quy mô tỷ ảnh để matched — ResNet-50 (ImageNet 1,28M) vẫn là lựa chọn cổ điển duy nhất.
2. License: cả hai model đều thuộc **DINOv3 License (commercial)** — dùng cho nghiên cứu ổn, nhưng khi công bố code/phân phối weights cần xem kỹ điều khoản.

**Ngoài DINO family (nếu muốn độc lập hơn):** CLIP ResNet-50 (~25M params, pretrain 400M cặp ảnh–text, license MIT) — nhưng học theo contrastive image-text chứ không phải SSL thuần, nên "matched" kém hơn.

### 6.3 Cặp cổ điển (literature) và protocol

**Cặp cổ điển (đối chiếu literature, nếu cần):** Xception (ImageNet) vs ViT-B/16 (ImageNet) — cùng pretrain ImageNet-1k, nhưng quy mô nhỏ hơn DINOv3 ~1000×.

**Literature anchors (inference-only):** ICT (weights chính thức), M2TR (Model Zoo), CViT.

**Protocol đề xuất:**

1. **Dữ liệu:** train/val/test trên FF++ c23, split chuẩn **720/140/140** video (dùng list có sẵn trong repo `ondyari/FaceForensics`); test thêm **FF++ c40 + RAW** (đo robustness với nén); **Celeb-DF v2 test-only** (đo cross-domain). Tùy chọn: DFDC (test subset).
2. **Tiền xử lý:** RetinaFace detect + align, resize 224×224, cố định N frames/video (vd 10) — **GIỮ NGUYÊN pipeline cho cả hai model** (đừng dùng pipeline riêng của từng paper, vì đó là lý do số trên báo không so sánh được với nhau).
3. **Metric:** frame-level ACC + AUC; video-level (majority vote) ACC + AUC; cross-dataset AUC (train FF++ → test Celeb-DF); robustness (train c23 → test c40/RAW); chạy **3 seeds**, báo mean ± std.
4. **Pitfall đã kiểm chứng cần ghi vào bài:**
   - Cùng model Xception nhưng số lệch nhau vì protocol: F3-Net báo FF++ LQ **86.86%** (frame-level) trong khi bài FF++ báo **81.00%** (video-level + face tracking) → luôn tự chạy baseline Xception của bạn làm thước đo chuẩn hóa.
   - M2TR Model Zoo công bố trên subset FF-DF, không phải full FF++ → weights phát hành ≠ số paper trên benchmark đầy đủ.
   - Không để lọt video giữa train/test; không dùng Celeb-DF để train nếu muốn đo generalization.
5. **Tải dữ liệu:** FF++ và Celeb-DF cần điền form (theo repo `ondyari/FaceForensics` và `yuezunli/celeb-deepfakeforensics`); DFDC full ~470GB. Bản FF++ RAW rất lớn — nếu hạn chế tài nguyên, test robustness bằng c40 là đủ.
6. **Tài nguyên ước lượng:** fine-tune Xception/ViT-B/16 trên FF++ c23 (~3.600 video train) mất vài giờ trên 1 GPU; linear probe DINOv2/DINOv3 nhanh hơn nhiều.

---

## 7. Nguồn (Sources)

- Xception: Chollet, CVPR 2017 — https://arxiv.org/abs/1610.02357
- FF++: Rössler et al., ICCV 2019 — https://arxiv.org/abs/1901.08971
- Celeb-DF: Li et al., CVPR 2020 — https://arxiv.org/abs/1909.12962
- DFDC: Dolhansky et al., 2020 — https://arxiv.org/abs/2006.07397
- MesoNet: Afchar et al., 2018 — https://arxiv.org/abs/1809.00888
- F3-Net: Qian et al., ECCV 2020 — https://arxiv.org/abs/2007.09355
- CViT: Wodajo & Atnafu, 2021 — https://arxiv.org/abs/2102.11126
- M2TR: Wang et al., ICMR 2022 — https://arxiv.org/abs/2104.09770
- ICT: Dong et al., 2022 — https://arxiv.org/abs/2203.01318 ; code: https://github.com/LightDXY/ICT_DeepFake
- GenConViT: Wodajo et al., 2023 — https://arxiv.org/abs/2307.07036 ; code: https://github.com/erprogs/GenConViT
- UCF: Yan et al., ICCV 2023 — https://arxiv.org/abs/2304.13949
- RealForensics: Haliassos et al., CVPR 2022 — https://arxiv.org/abs/2201.07131 ; code: https://github.com/ahaliassos/RealForensics
- ViT survey: Wang et al., 2024 — https://arxiv.org/abs/2405.08463
- ViT gốc: Dosovitskiy et al., ICLR 2021 — https://arxiv.org/abs/2010.11929
- DINOv3: Siméoni et al., 2025 — https://arxiv.org/abs/2508.10104 ; model card: https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md ; license: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
- DINOv2: Oquab et al., 2024 — https://arxiv.org/abs/2304.07193 ; repo (Apache 2.0): https://github.com/facebookresearch/dinov2
- DINOv3 cho cross-generator forgery: 2025 — https://arxiv.org/abs/2511.22471
- DINOv2+LoRA cho face forgery (AAAI 2026): 2025 — https://arxiv.org/abs/2511.12107
- SSL ViT cho deepfake detection (gồm DINO family): 2024 — https://arxiv.org/abs/2405.00355
- DF40 benchmark (40 kỹ thuật tạo giả, NeurIPS 2024 D&B): https://arxiv.org/abs/2406.13495

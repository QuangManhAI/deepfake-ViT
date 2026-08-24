# Eval `exp02_weak_max_v3.pt` trên test_balanced (BICUBIC)

- **Test set:** `experiments/results/error_analysis_lora/test_balanced.csv` — 2354 ảnh (1177 real / 1177 fake), **40 method giả**.
- **Model:** `outputs/finetune/exp02_weak_max_v3.pt` (finetune 152K + gif MidJourney)
- **Accuracy 98.05%** · ROC-AUC 0.9980 · Precision 97.56% · Recall 98.56% · F1 98.06%
- **Confusion:** TN 1148 · FP 29 · FN 17 · TP 1160

## 1. Phát hiện fake theo method

| Method | n | TP | FN | Detection rate |
|---|---|---|---|---|
| 🔴 heygen | 1 | 0 | 1 | 0.0% |
| 🟡 facedancer | 27 | 25 | 2 | 92.6% |
| 🟢 StyleGANXL | 40 | 38 | 2 | 95.0% |
| 🟢 DiT | 41 | 39 | 2 | 95.1% |
| 🟢 wav2lip | 22 | 21 | 1 | 95.5% |
| 🟢 fsgan | 26 | 25 | 1 | 96.2% |
| 🟢 sadtalker | 26 | 25 | 1 | 96.2% |
| 🟢 faceswap | 27 | 26 | 1 | 96.3% |
| 🟢 lia | 27 | 26 | 1 | 96.3% |
| 🟢 tpsm | 27 | 26 | 1 | 96.3% |
| 🟢 CollabDiff | 31 | 30 | 1 | 96.8% |
| 🟢 pixart | 37 | 36 | 1 | 97.3% |
| 🟢 SiT | 40 | 39 | 1 | 97.5% |
| 🟢 e4e | 40 | 39 | 1 | 97.5% |
| 🟢 MRAA | 29 | 29 | 0 | 100.0% |
| 🟢 MidJourney | 26 | 26 | 0 | 100.0% |
| 🟢 RDDM | 22 | 22 | 0 | 100.0% |
| 🟢 StyleGAN2 | 25 | 25 | 0 | 100.0% |
| 🟢 StyleGAN3 | 40 | 40 | 0 | 100.0% |
| 🟢 VQGAN | 24 | 24 | 0 | 100.0% |
| 🟢 blendface | 27 | 27 | 0 | 100.0% |
| 🟢 danet | 27 | 27 | 0 | 100.0% |
| 🟢 ddim | 24 | 24 | 0 | 100.0% |
| 🟢 deepfacelab | 1 | 1 | 0 | 100.0% |
| 🟢 e4s | 15 | 15 | 0 | 100.0% |
| 🟢 facevid2vid | 27 | 27 | 0 | 100.0% |
| 🟢 fomm | 27 | 27 | 0 | 100.0% |
| 🟢 hyperreenact | 27 | 27 | 0 | 100.0% |
| 🟢 inswap | 19 | 19 | 0 | 100.0% |
| 🟢 mcnet | 27 | 27 | 0 | 100.0% |
| 🟢 mobileswap | 56 | 56 | 0 | 100.0% |
| 🟢 one_shot_free | 27 | 27 | 0 | 100.0% |
| 🟢 pirender | 27 | 27 | 0 | 100.0% |
| 🟢 sd2.1 | 64 | 64 | 0 | 100.0% |
| 🟢 simswap | 27 | 27 | 0 | 100.0% |
| 🟢 stargan | 40 | 40 | 0 | 100.0% |
| 🟢 starganv2 | 40 | 40 | 0 | 100.0% |
| 🟢 styleclip | 40 | 40 | 0 | 100.0% |
| 🟢 uniface | 27 | 27 | 0 | 100.0% |
| 🟢 whichfaceisreal | 30 | 30 | 0 | 100.0% |

**Tóm tắt:** 38/40 method ≥95% · 1 method 80–95% · 1 method <80%.

## 2. Real accuracy theo domain

| Domain | n | Đúng | Real acc |
|---|---|---|---|
| cdc | 178 | 178 | 100.0% |
| ffc | 999 | 970 | 97.1% |

## 3. Nhận xét

- **Method yếu nhất (miss nhiều):** `heygen` (0.0%, FN 1).
- **MidJourney:** n=26 → detection 100.0% (FN 0) — trước finetune là 38.5%.
- **Real giữ ổn:** tổng real acc 97.5% — không sụp khi vá method yếu.

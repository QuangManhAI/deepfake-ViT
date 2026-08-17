# Kết quả 40 method — DINOv3 ViT-S/16 vs ConvNeXt-Tiny

- **Dataset:** `test_data_v3/` — cấu trúc `real/` + `<method>/fake/`
- **Protocol:** identity-disjoint (split seed 42, train_ratio 0.7, theo cột `identity`) — model test chỉ trên identity chưa từng thấy trong train
- **Probe:** frozen backbone (feature extractor) + LogisticRegression linear probe (class_weight=balanced)
- **Train/test:** 21,459 / 9,232 ảnh, 23,237 identity keys

## Tổng quan

| Metric | DINOv3 ViT-S/16 | ConvNeXt-Tiny |
|---|---|---|
| Accuracy | 0.9519 | 0.9441 |
| Precision | 0.9947 | 0.9906 |
| Recall | 0.9551 | 0.9509 |
| F1 | 0.9745 | 0.9704 |
| AUC | 0.9746 | 0.9531 |
| Nhận đúng REAL (acc) | 0.8711 (FPR 12.9%) | 0.7708 (FPR 22.9%) |
| Bắt FAKE (det) | 0.9551 | 0.9509 |
| Paired-only Acc | 0.8586 | 0.8348 |

## Theo domain (test)

| Domain | n | ViT acc | ViT real | ViT fake | CNN acc | CNN real | CNN fake |
|---|---|---|---|---|---|---|---|
| cdc | 583 | 0.9691 | 0.9149 | 0.9739 | 0.9794 | 0.9787 | 0.9795 |
| efs | 1796 | 0.9894 | nan | nan | 0.9905 | nan | nan |
| fe | 894 | 0.9966 | nan | nan | 0.9810 | nan | nan |
| ffc | 2026 | 0.8268 | 0.8642 | 0.8202 | 0.7932 | 0.7384 | 0.8028 |
| oth | 3933 | 0.9865 | nan | nan | 0.9870 | nan | nan |

## 40 method — detection rate (fake) theo (method, domain)

Domain: **cdc**=Celeb-DF · **ffc**=FF++ · **efs**=tổng hợp · **oth**=không ghép · **fe**=expression. Một method có thể trải nhiều domain.

| Method | domain | n | ViT det | CNN det |
|---|---|---:|---:|---:|
| blendface | cdc | 19 | 1.0000 | 0.9474 |
| blendface | ffc | 43 | 0.5814 | 0.4884 |
| blendface | oth | 150 | 0.9867 | 0.9800 |
| CollabDiff | efs | 196 | 0.9898 | 0.9949 |
| danet | cdc | 19 | 1.0000 | 1.0000 |
| danet | ffc | 43 | 0.9302 | 0.8140 |
| danet | oth | 141 | 1.0000 | 0.9929 |
| ddim | efs | 183 | 1.0000 | 1.0000 |
| deepfacelab | efs | 7 | 0.8571 | 0.7143 |
| DiT | cdc | 47 | 0.9574 | 0.9574 |
| DiT | ffc | 39 | 0.6410 | 0.6667 |
| DiT | oth | 223 | 0.9596 | 0.9686 |
| e4e | ffc | 302 | 0.9834 | 0.9934 |
| e4s | cdc | 10 | 1.0000 | 1.0000 |
| e4s | ffc | 32 | 0.8125 | 0.5938 |
| e4s | oth | 73 | 0.9589 | 0.9726 |
| facedancer | cdc | 19 | 0.8421 | 1.0000 |
| facedancer | ffc | 43 | 0.3023 | 0.3721 |
| facedancer | oth | 133 | 0.9774 | 0.9925 |
| faceswap | cdc | 19 | 0.9474 | 1.0000 |
| faceswap | ffc | 43 | 0.2326 | 0.4186 |
| faceswap | oth | 139 | 1.0000 | 1.0000 |
| facevid2vid | cdc | 19 | 1.0000 | 1.0000 |
| facevid2vid | ffc | 43 | 1.0000 | 1.0000 |
| facevid2vid | oth | 159 | 1.0000 | 1.0000 |
| fomm | cdc | 19 | 1.0000 | 1.0000 |
| fomm | ffc | 43 | 1.0000 | 1.0000 |
| fomm | oth | 141 | 1.0000 | 1.0000 |
| fsgan | cdc | 18 | 1.0000 | 1.0000 |
| fsgan | ffc | 43 | 0.5349 | 0.5116 |
| fsgan | oth | 146 | 1.0000 | 0.9932 |
| heygen | efs | 7 | 0.8571 | 0.7143 |
| hyperreenact | cdc | 19 | 1.0000 | 1.0000 |
| hyperreenact | ffc | 43 | 1.0000 | 0.9767 |
| hyperreenact | oth | 137 | 1.0000 | 1.0000 |
| inswap | cdc | 14 | 0.9286 | 1.0000 |
| inswap | ffc | 38 | 0.4211 | 0.5263 |
| inswap | oth | 85 | 0.9647 | 1.0000 |
| lia | cdc | 19 | 0.8947 | 0.9474 |
| lia | ffc | 43 | 0.7442 | 0.8140 |
| lia | oth | 140 | 0.9429 | 0.9714 |
| mcnet | cdc | 19 | 1.0000 | 1.0000 |
| mcnet | ffc | 43 | 0.8837 | 0.8605 |
| mcnet | oth | 156 | 0.9936 | 0.9936 |
| MidJourney | efs | 188 | 0.9574 | 0.9947 |
| mobileswap | cdc | 19 | 1.0000 | 1.0000 |
| mobileswap | ffc | 259 | 0.7529 | 0.6216 |
| mobileswap | oth | 148 | 1.0000 | 1.0000 |
| MRAA | cdc | 19 | 1.0000 | 1.0000 |
| MRAA | ffc | 43 | 0.7674 | 0.9302 |
| MRAA | oth | 136 | 0.9926 | 1.0000 |
| one_shot_free | cdc | 19 | 1.0000 | 1.0000 |
| one_shot_free | ffc | 43 | 0.9535 | 0.9535 |
| one_shot_free | oth | 152 | 1.0000 | 1.0000 |
| pirender | cdc | 19 | 1.0000 | 0.9474 |
| pirender | ffc | 43 | 1.0000 | 0.9767 |
| pirender | oth | 149 | 1.0000 | 0.9933 |
| pixart | cdc | 26 | 1.0000 | 1.0000 |
| pixart | ffc | 39 | 0.6154 | 0.8718 |
| pixart | oth | 209 | 0.9378 | 0.9665 |
| RDDM | cdc | 23 | 1.0000 | 1.0000 |
| RDDM | ffc | 39 | 1.0000 | 1.0000 |
| RDDM | oth | 93 | 1.0000 | 1.0000 |
| sadtalker | oth | 201 | 1.0000 | 1.0000 |
| sd2.1 | cdc | 26 | 1.0000 | 1.0000 |
| sd2.1 | ffc | 249 | 0.9839 | 0.9759 |
| sd2.1 | oth | 212 | 1.0000 | 0.9906 |
| simswap | cdc | 19 | 0.8947 | 0.9474 |
| simswap | ffc | 43 | 0.5349 | 0.4186 |
| simswap | oth | 137 | 1.0000 | 0.9489 |
| SiT | cdc | 47 | 0.9574 | 0.9574 |
| SiT | ffc | 39 | 0.7179 | 0.6667 |
| SiT | oth | 220 | 0.9636 | 0.9591 |
| stargan | fe | 280 | 1.0000 | 0.9536 |
| starganv2 | fe | 313 | 1.0000 | 1.0000 |
| styleclip | fe | 301 | 0.9900 | 0.9867 |
| StyleGAN2 | efs | 179 | 0.9888 | 1.0000 |
| StyleGAN3 | efs | 309 | 1.0000 | 0.9903 |
| StyleGANXL | efs | 305 | 1.0000 | 0.9869 |
| tpsm | cdc | 19 | 1.0000 | 1.0000 |
| tpsm | ffc | 43 | 0.9070 | 0.8837 |
| tpsm | oth | 153 | 0.9935 | 0.9935 |
| uniface | cdc | 19 | 1.0000 | 0.9474 |
| uniface | ffc | 43 | 0.6977 | 0.5814 |
| uniface | oth | 138 | 1.0000 | 0.9855 |
| VQGAN | efs | 195 | 1.0000 | 0.9949 |
| wav2lip | cdc | 21 | 0.9524 | 0.9048 |
| wav2lip | oth | 162 | 0.9938 | 0.9877 |
| whichfaceisreal | efs | 227 | 0.9780 | 0.9868 |

## REAL theo domain

| Domain | n real | ViT real acc | CNN real acc |
|---|---|---:|---:|
| cdc | 47 | 0.9149 | 0.9787 |
| ffc | 302 | 0.8642 | 0.7384 |


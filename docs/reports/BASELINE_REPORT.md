# Baseline Report — Phase 3

**Protocol:** `identity_clean_v1` (identity-disjoint, seed 42)
**Backbone:** DINOv3 ViT-S/16 (`experiments/checkpoints/weights/model.safetensors`, 82.4 MB, from `ManhQuangAI/dinov3-deepfake-detection`)
**Head:** `Linear(384, 2)` on the CLS token
**Test set:** 4,499 images (171 real / 4,328 fake)

FP is defined as REAL predicted as FAKE. FN is defined as FAKE predicted as REAL.

---

## 1. Experiments

| | Baseline A | Baseline B |
|---|---|---|
| Loss | `CrossEntropyLoss` | `CrossEntropyLoss(weight=[12.7064, 0.5205])` |
| Epochs | 1 | 1 |
| Batch size | 32 | 32 |
| Optimizer | AdamW (backbone 1e-5, head 1e-3) | same |
| Scheduler | CosineAnnealingLR | same |
| Seed | 42 | 42 |
| Input | 256×256 | same |
| Augmentation | Resize + RandomHorizontalFlip | same |
| Device | MPS | MPS |
| Best-checkpoint criterion | highest validation accuracy | same |

Both runs used the identical protocol, seed, architecture, and hyperparameters. The only difference is class weighting.

---

## 2. Test Metrics

| metric | baseline_a_standard | baseline_b_class_weighted |
|---|---|---|
| accuracy | 0.9773 | 0.9624 |
| balanced_accuracy | 0.7326 | 0.9018 |
| precision | 0.9794 | 0.9934 |
| recall | 0.9975 | 0.9674 |
| f1 | 0.9883 | 0.9802 |
| mcc | 0.6320 | 0.6319 |
| roc_auc | 0.9795 | 0.9783 |
| pr_auc | 0.9992 | 0.9991 |
| real_precision | 0.8791 | 0.5035 |
| real_recall | 0.4678 | 0.8363 |
| fake_precision | 0.9794 | 0.9934 |
| fake_recall | 0.9975 | 0.9674 |
| tn | 80.0000 | 143.0000 |
| fp | 91.0000 | 28.0000 |
| fn | 11.0000 | 141.0000 |
| tp | 4317.0000 | 4187.0000 |

---

## 3. How Good Is the Baseline?

**Accuracy is misleading.** Baseline A reaches **0.9773** accuracy but only **0.7326** balanced accuracy. With a 25:1 fake:real test set, a model that predicts "fake" almost always scores high accuracy while barely detecting real images.

The honest picture:

| | Baseline A | Baseline B |
|---|---|---|
| Accuracy | 0.9773 | 0.9624 |
| **Balanced Accuracy** | **0.7326** | **0.9018** |
| MCC | 0.6320 | 0.6319 |
| ROC-AUC | 0.9795 | 0.9783 |
| PR-AUC | 0.9992 | 0.9991 |
| **Real Recall** | **0.4678** | **0.8363** |
| Fake Recall | 0.9975 | 0.9674 |

**FACT:** Baseline A misses **53.2%** of real images (91 of 171 real images labelled fake). Baseline B reduces that to **16.4%** (28 of 171).

**FACT:** ROC-AUC is nearly identical between A (0.9795) and B (0.9783), and MCC is nearly identical (0.6320 vs 0.6319). The ranking quality of the two models is comparable; the difference is where the decision threshold effectively sits.

**INFERENCE:** Class weighting mainly moves the operating point rather than improving the underlying representation. Balanced accuracy improves by +0.1692 at the cost of +130 additional false negatives.

**PR-AUC near 0.999 is not impressive here** — the positive class is 96% of the test set, so a trivial classifier already gets high average precision. ROC-AUC and MCC are the more informative headline numbers.

---

## 4. Recall Balance

| Baseline | Real Recall | Fake Recall | Gap |
|---|---|---|---|
| A | 0.4678 | 0.9975 | 0.5296 |
| B | 0.8363 | 0.9674 | 0.1312 |

**FACT:** Neither baseline is balanced. The real class remains substantially harder in both.

---

## 5. Dominant Error Types

| Baseline | TN | FP (real→fake) | FN (fake→real) | TP |
|---|---|---|---|---|
| A | 80 | 91 | 11 | 4317 |
| B | 143 | 28 | 141 | 4187 |

**Baseline A:** dominated by **false positives** — 91 real images misclassified as fake against only 11 false negatives. The model has largely learned "predict fake".

**Baseline B:** more symmetric — 28 FP and 141 FN. Class weighting shifted errors from the real class onto the fake class.

**INFERENCE:** The real class is the harder class in both settings. With 826 real training images versus 20,165 fake, this is consistent with the training distribution, though this analysis does not prove causality.

---

## 6. Weakest Fake Methods (Baseline B)

Methods with F1 below 1.0, sorted ascending:

| method | sample_count | accuracy | recall | f1 | false_negative_rate | average_confidence |
|---|---|---|---|---|---|---|
| heygen | 3 | 0.6667 | 0.6667 | 0.8000 | 0.3333 | 0.5984 |
| e4s | 51 | 0.8039 | 0.8039 | 0.8913 | 0.1961 | 0.8151 |
| mobileswap | 199 | 0.8191 | 0.8191 | 0.9006 | 0.1809 | 0.8046 |
| real | 171 | 0.8363 | 0.8363 | 0.9108 | 0.1637 | 0.2220 |
| faceswap | 89 | 0.8652 | 0.8652 | 0.9277 | 0.1348 | 0.8709 |
| facedancer | 100 | 0.8900 | 0.8900 | 0.9418 | 0.1100 | 0.8702 |
| simswap | 91 | 0.9341 | 0.9341 | 0.9659 | 0.0659 | 0.8987 |
| SiT | 152 | 0.9408 | 0.9408 | 0.9695 | 0.0592 | 0.9298 |
| fsgan | 87 | 0.9425 | 0.9425 | 0.9704 | 0.0575 | 0.9290 |
| mcnet | 94 | 0.9468 | 0.9468 | 0.9727 | 0.0532 | 0.9424 |
| danet | 98 | 0.9490 | 0.9490 | 0.9738 | 0.0510 | 0.9373 |
| inswap | 65 | 0.9538 | 0.9538 | 0.9764 | 0.0462 | 0.9245 |

Full table: `experiments/results/baseline/eval_b/per_method_metrics.csv`

**FACT:** `real` has the lowest F1 in both baselines. Among fake methods, the weakest are face-swap style methods.

**FACT:** Most methods reach F1 = 1.0 on the test set. Per-method test counts are small (typically 65–225 images), so individual method F1 values have wide confidence intervals and should not be over-interpreted.

---

## 7. Errors vs Dataset Properties (Associations Only)

Join coverage against `sample_quality.csv`: **100.0%** of test predictions matched.

| metric | median_correct | median_error | mean_correct | mean_error | n_correct | n_error |
|---|---|---|---|---|---|---|
| width | 256.0000 | 256.0000 | 391.9215 | 289.3254 | 4330 | 169 |
| height | 256.0000 | 256.0000 | 391.9215 | 289.3254 | 4330 | 169 |
| brightness | 84.9118 | 111.8542 | 89.5596 | 115.1155 | 4330 | 169 |
| contrast | 45.2256 | 54.1478 | 46.6235 | 54.9801 | 4330 | 169 |
| edge | 3.2854 | 5.0418 | 3.6010 | 5.3178 | 4330 | 169 |
| bits_per_pixel | 2.8723 | 3.6326 | 2.7001 | 3.6083 | 4330 | 169 |
| file_size | 74655.5000 | 90282.0000 | 157170.9046 | 130090.0118 | 4330 | 169 |

**These are associations, not causes.**

### Error concentration (Baseline B, top groups)

```json
{
  "video": {
    "255_214": 14,
    "386_154": 12,
    "842_714": 8,
    "847_906": 7,
    "661_670": 5,
    "955_078": 5,
    "521_517": 5,
    "731_741": 4,
    "003_000": 4,
    "ff:661": 3
  },
  "identity": {
    "ffc:255": 14,
    "ffc:386": 12,
    "ffc:842": 9,
    "ffc:661": 9,
    "ffc:847": 9,
    "ffc:955": 6,
    "ffc:731": 5,
    "ffc:521": 5,
    "ffc:003": 5,
    "ffc:851": 4
  },
  "domain": {
    "ffc": 148,
    "efs": 11,
    "oth": 7,
    "cdc": 2,
    "fe": 1
  },
  "method": {
    "mobileswap": 36,
    "real": 28,
    "faceswap": 12,
    "facedancer": 11,
    "e4s": 10,
    "SiT": 9,
    "simswap": 6,
    "DiT": 6,
    "fsgan": 5,
    "danet": 5
  }
}
```

Artifacts:
- `experiments/results/baseline/eval_b/error_quality_association.csv`
- `experiments/results/baseline/eval_b/error_concentration.json`
- `experiments/results/baseline/eval_b/error_quality_distributions.png`

---

## 8. Important Caveats

1. **One epoch only.** Both baselines trained for a single epoch on MPS (~22 min each). These are genuine baselines, not converged models. Longer training would likely change all numbers.
2. **Best checkpoint selected by validation accuracy.** Under a 24:1 imbalance, validation accuracy is a poor selection criterion and likely favours the fake-predicting solution. Balanced accuracy or MCC would be a better criterion — this is a concrete improvement for the next phase.
3. **Video/source overlap remains** in the primary protocol. Some of the apparent per-method performance may reflect source memorization rather than manipulation-artifact detection.
4. **Small per-method test counts.** Most methods have under 225 test images; per-method F1 = 1.0 does not establish robustness.
5. **No threshold tuning.** All metrics use the default 0.5 argmax threshold. `threshold_analysis.png` shows the trade-off curve.

---

## 9. Recommendation

**Baseline B (class-weighted) is the better reference baseline** for balanced accuracy (0.9018 vs 0.7326) and real recall (0.8363 vs 0.4678), with essentially identical ROC-AUC and MCC.

Neither run should be treated as a strong model. Both are 1-epoch references for Error Analysis.

---

## 10. Ranked Candidate Investigations for Phase 4 — Error Analysis

1. **Real-class failure.** Which real images are misclassified, and do they share source, quality, or domain properties? This is the single largest error source in Baseline A and remains substantial in B.
2. **Checkpoint-selection criterion.** Re-select the best checkpoint by balanced accuracy or MCC and quantify how much of the A/B difference is threshold effect versus learned representation.
3. **Threshold calibration.** Use `threshold_analysis.png` to find the operating point maximizing balanced accuracy for both baselines and compare like-for-like.
4. **Face-swap method weakness.** Investigate the lowest-F1 fake methods and whether their errors concentrate in specific source videos.
5. **Video-level error concentration.** Check whether errors cluster in specific source videos, which would suggest source memorization given the known video overlap.
6. **Quality-metric association.** Test whether errors correlate with sharpness, brightness, contrast, or compression beyond the medians reported above.
7. **Identity-level error concentration.** Determine whether errors concentrate in specific identities or are diffuse.
8. **Video-disjoint benchmark.** Evaluate the same checkpoint on `data/splits_video_clean/` to estimate how much performance depends on source overlap.
9. **Longer training.** Confirm which conclusions survive multi-epoch training before drawing conclusions about method difficulty.

---

## 11. Files

```
experiments/results/baseline/
├── baseline_comparison.csv
├── metrics_baseline_a.json
├── metrics_baseline_b.json
├── 20260830_155011_baseline_a_standard/     (checkpoints, logs, metrics)
├── 20260830_161435_baseline_b_weighted/     (checkpoints, logs, metrics)
├── eval_a/
│   ├── config.json, metrics.json, README.md
│   ├── test_predictions.csv, per_method_metrics.csv
│   └── confusion_matrix.png, confusion_matrix_normalized.png,
│       roc_curve.png, pr_curve.png, loss_curve.png,
│       test_score_distribution.png, threshold_analysis.png
└── eval_b/
    ├── (same as eval_a)
    ├── error_quality_association.csv
    ├── error_concentration.json
    └── error_quality_distributions.png
```

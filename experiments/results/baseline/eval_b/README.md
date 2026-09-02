# Baseline Evaluation

## Run
- Checkpoint: `experiments/results/baseline/20260830_161435_baseline_b_weighted/checkpoints/baseline_b_weighted_best.pt`
- Test CSV: `/Users/pickapu/Documents/PyCharmMiscProject/deepfake-ViT/data/protocol/test.csv`
- Evaluated: 2026-08-30 16:41:16

## Overall Test Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.9624 |
| Balanced Accuracy | 0.9018 |
| Precision | 0.9934 |
| Recall | 0.9674 |
| F1 | 0.9802 |
| MCC | 0.6319 |
| ROC-AUC | 0.9783 |
| PR-AUC | 0.9991 |
| Real Precision | 0.5035 |
| Real Recall | 0.8363 |
| Fake Precision | 0.9934 |
| Fake Recall | 0.9674 |
| TN/FP/FN/TP | 143/28/141/4187 |

## Files

- `test_predictions.csv` — per-image predictions and metadata
- `per_method_metrics.csv` — method-level metrics
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `roc_curve.png`
- `pr_curve.png`
- `loss_curve.png`
- `test_score_distribution.png`
- `threshold_analysis.png`
- `metrics.json`
- `config.json`

# Baseline Evaluation

## Run
- Checkpoint: `experiments/results/baseline/20260830_155011_baseline_a_standard/checkpoints/baseline_a_standard_best.pt`
- Test CSV: `/Users/pickapu/Documents/PyCharmMiscProject/deepfake-ViT/data/protocol/test.csv`
- Evaluated: 2026-08-30 16:27:23

## Overall Test Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.9773 |
| Balanced Accuracy | 0.7326 |
| Precision | 0.9794 |
| Recall | 0.9975 |
| F1 | 0.9883 |
| MCC | 0.6320 |
| ROC-AUC | 0.9795 |
| PR-AUC | 0.9992 |
| Real Precision | 0.8791 |
| Real Recall | 0.4678 |
| Fake Precision | 0.9794 |
| Fake Recall | 0.9975 |
| TN/FP/FN/TP | 80/91/11/4317 |

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

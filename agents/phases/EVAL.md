# EVAL.md — Evaluation

- **Title:** Evaluation & ViT-vs-CNN Comparison
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-18
- **Description:** Metrics, evaluation scripts, and the ViT-vs-CNN comparison
  with attention visualization.
- **Status:** In Progress

## Background

The rubric requires a measurable `>95%` test accuracy plus a matched-parameter
ViT-vs-CNN comparison and attention maps — all reported with 5W1H context.

## Goals / Purpose

- Report accuracy, precision, recall, F1, ROC-AUC, confusion matrix on the
  held-out test split.
- Compare ViT vs CNN; visualize attention (required deliverables).

## Input / Output

- **Input:** checkpoints + test data + cached features.
- **Output:** JSON/MD reports in `experiments/results/eval/`; figures in
  `experiments/plots/` and `experiments/results/report/figures/`.

## How to do it (general plan)

- `src/eval/evaluate.py`, `predict.py`, `eval_df40_vit_cnn.py`, `eval_*.py`.
- `src/experiments/compare_models.py`, `visualize_attention.py`, `assemble_attention_figure.py`.
- **Advanced Diagnostic Pipeline in Notebook 02:**
  - Optimal threshold cutoff ($\tau^*$) via Youden's J statistic ($J = \text{Sens} + \text{Spec} - 1$).
  - Test-Time Augmentation (TTA) with horizontal mirror prediction averaging.
  - ViT-S/16 + ConvNeXt-Tiny probability ensembling ($0.65 \cdot P_{ViT} + 0.35 \cdot P_{CNN}$).

## Pipeline

```
Checkpoints (ViT / CNN) → Validation Threshold Optimization (tau*) → Test-Time Augmentation (TTA) → ViT+CNN Ensemble → 6-in-1 Diagnostic Dashboard
```

## Detailed plan / gotchas

- Metrics per [rules/RESULTS_REPORTING.md](../rules/RESULTS_REPORTING.md) (5W1H).
- Eval loops guard with `torch.no_grad()`.
- Test set reports: `experiments/results/exp01_max_accuracy_report.json`.
- Attention output dir: `experiments/plots/attention/`.

## Links

- Progress: [../progress/EVAL_STATUS.md](../progress/EVAL_STATUS.md)
- Experiment plan: [../experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md](../experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md)
- Overview: [../OVERVIEW.md](../OVERVIEW.md)

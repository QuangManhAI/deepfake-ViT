# Baseline Experiments — Phase 3

**Protocol:** `identity_clean_v1` (identity-disjoint, seed 42)
**Backbone:** DINOv3 ViT-S/16 + `Linear(384, 2)` head
**Weights:** `experiments/checkpoints/weights/model.safetensors`
**Full report:** `../../../BASELINE_REPORT.md`

Root-level `metrics.json`, `config.json`, `test_predictions.csv`, `per_method_metrics.csv`,
and the `.png` figures are copies of the **recommended baseline (B, class-weighted)**.

## Runs

| Run | Loss | Directory |
|-----|------|-----------|
| Baseline A | standard `CrossEntropyLoss` | `20260830_155011_baseline_a_standard/` |
| Baseline B | class-weighted `CrossEntropyLoss` | `20260830_161435_baseline_b_weighted/` |

Each run directory contains `checkpoints/`, `logs/`, and `metrics/` (history JSONL + config JSON).

## Evaluations

| Directory | Baseline |
|-----------|----------|
| `eval_a/` | A (standard) |
| `eval_b/` | B (class-weighted) — recommended |

## Test Metrics

| Metric | A (standard) | B (class-weighted) |
|--------|-------------|--------------------|
| Accuracy | 0.9773 | 0.9624 |
| **Balanced Accuracy** | **0.7326** | **0.9018** |
| MCC | 0.6320 | 0.6319 |
| ROC-AUC | 0.9795 | 0.9783 |
| PR-AUC | 0.9992 | 0.9991 |
| Real Recall | 0.4678 | 0.8363 |
| Fake Recall | 0.9975 | 0.9674 |
| TN / FP / FN / TP | 80 / 91 / 11 / 4317 | 143 / 28 / 141 / 4187 |

FP = REAL predicted as FAKE. FN = FAKE predicted as REAL.

**Accuracy is misleading** under the 25:1 test imbalance. Use balanced accuracy and MCC.

## Reproduce

```bash
# Baseline A
.venv/bin/python src/training/train.py \
    --output-dir experiments/results/baseline \
    --run-name baseline_a_standard \
    --report experiments/results/baseline/metrics_baseline_a.json \
    --epochs 1

# Baseline B
.venv/bin/python src/training/train.py \
    --output-dir experiments/results/baseline \
    --run-name baseline_b_weighted \
    --report experiments/results/baseline/metrics_baseline_b.json \
    --class-weight --epochs 1

# Evaluate
.venv/bin/python src/training/evaluate_baseline.py \
    --run-dir experiments/results/baseline/<run_dir> \
    --output-dir experiments/results/baseline/eval_x --batch-size 64

# Consolidate + report
.venv/bin/python src/training/baseline_summary.py
```

Train/val/test CSVs are resolved automatically from `data/protocol/protocol_config.json`.

## Caveats

- **1 epoch only** (~22 min each on MPS). These are references, not converged models.
- **Best checkpoint selected by validation accuracy**, which is a poor criterion under imbalance.
- **Video/source overlap** remains in the primary protocol.
- **Small per-method test counts** (65–225 images); per-method F1 = 1.0 is not robust evidence.
- **No threshold tuning**; all metrics use the default 0.5 argmax threshold.

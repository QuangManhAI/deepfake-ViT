"""Consolidate baseline A/B results, join errors to image quality, write BASELINE_REPORT.md."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE = PROJECT_ROOT / "experiments" / "results" / "baseline"
QUALITY = PROJECT_ROOT / "experiments" / "results" / "data_quality" / "sample_quality.csv"


def load(name):
    d = BASE / name
    metrics = json.loads((d / "metrics.json").read_text())
    preds = pd.read_csv(d / "test_predictions.csv")
    pmm = pd.read_csv(d / "per_method_metrics.csv")
    return metrics, preds, pmm


def error_quality_association(preds, out_dir):
    """Join predictions to sample_quality.csv and report associations (not causality)."""
    q = pd.read_csv(QUALITY)
    # normalize paths for the join
    q["path"] = q["path"].astype(str)
    preds["path"] = preds["path"].astype(str)
    m = preds.merge(q, on="path", how="left", suffixes=("", "_q"))
    coverage = m["edge"].notna().mean() if "edge" in m.columns else 0.0

    metrics_cols = [c for c in ["width", "height", "brightness", "contrast", "edge", "bits_per_pixel", "file_size"] if c in m.columns]
    rows = []
    for col in metrics_cols:
        correct = m[m["correct"] == 1][col].dropna()
        wrong = m[m["correct"] == 0][col].dropna()
        rows.append({
            "metric": col,
            "median_correct": correct.median(),
            "median_error": wrong.median(),
            "mean_correct": correct.mean(),
            "mean_error": wrong.mean(),
            "n_correct": len(correct),
            "n_error": len(wrong),
        })
    qa = pd.DataFrame(rows)
    qa.to_csv(out_dir / "error_quality_association.csv", index=False)

    # concentration by video / identity / domain
    conc = {}
    for key in ["video", "identity", "domain", "method"]:
        errs = m[m["correct"] == 0]
        if len(errs) == 0:
            conc[key] = {}
            continue
        top = errs[key].value_counts().head(10)
        conc[key] = {str(k): int(v) for k, v in top.items()}
    with open(out_dir / "error_concentration.json", "w") as f:
        json.dump({"quality_join_coverage": float(coverage), "top_error_groups": conc}, f, indent=2)

    # plot: quality metric distributions correct vs error
    if metrics_cols:
        n = len(metrics_cols)
        fig, axes = plt.subplots(1, min(4, n), figsize=(5 * min(4, n), 4))
        axes = [axes] if n == 1 else list(axes)
        for ax, col in zip(axes, metrics_cols[:4]):
            ax.hist(m[m["correct"] == 1][col].dropna(), bins=40, alpha=0.6, density=True, label="correct")
            ax.hist(m[m["correct"] == 0][col].dropna(), bins=40, alpha=0.6, density=True, label="error")
            ax.set_title(col)
            ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "error_quality_distributions.png", dpi=150)
        plt.close(fig)

    return qa, conc, coverage


def main():
    a_metrics, a_preds, a_pmm = load("eval_a")
    b_metrics, b_preds, b_pmm = load("eval_b")

    # Comparison CSV
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc",
            "real_precision", "real_recall", "fake_precision", "fake_recall", "tn", "fp", "fn", "tp"]
    comp = pd.DataFrame([
        {"metric": k, "baseline_a_standard": a_metrics[k], "baseline_b_class_weighted": b_metrics[k]}
        for k in keys
    ])
    comp.to_csv(BASE / "baseline_comparison.csv", index=False)

    # Error-quality association for the recommended baseline (B)
    qa, conc, coverage = error_quality_association(b_preds, BASE / "eval_b")

    # Weakest methods for B (exclude perfect scores)
    weak_b = b_pmm[b_pmm["f1"] < 1.0].sort_values("f1").head(12)

    def md_table(df, cols=None, floatfmt=4):
        d = df[cols] if cols else df
        headers = list(d.columns)
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join("---" for _ in headers) + "|"]
        for _, row in d.iterrows():
            cells = []
            for v in row:
                if isinstance(v, float):
                    cells.append(f"{v:.{floatfmt}f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    report = f"""# Baseline Report — Phase 3

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

{md_table(comp)}

---

## 3. How Good Is the Baseline?

**Accuracy is misleading.** Baseline A reaches **{a_metrics['accuracy']:.4f}** accuracy but only **{a_metrics['balanced_accuracy']:.4f}** balanced accuracy. With a 25:1 fake:real test set, a model that predicts "fake" almost always scores high accuracy while barely detecting real images.

The honest picture:

| | Baseline A | Baseline B |
|---|---|---|
| Accuracy | {a_metrics['accuracy']:.4f} | {b_metrics['accuracy']:.4f} |
| **Balanced Accuracy** | **{a_metrics['balanced_accuracy']:.4f}** | **{b_metrics['balanced_accuracy']:.4f}** |
| MCC | {a_metrics['mcc']:.4f} | {b_metrics['mcc']:.4f} |
| ROC-AUC | {a_metrics['roc_auc']:.4f} | {b_metrics['roc_auc']:.4f} |
| PR-AUC | {a_metrics['pr_auc']:.4f} | {b_metrics['pr_auc']:.4f} |
| **Real Recall** | **{a_metrics['real_recall']:.4f}** | **{b_metrics['real_recall']:.4f}** |
| Fake Recall | {a_metrics['fake_recall']:.4f} | {b_metrics['fake_recall']:.4f} |

**FACT:** Baseline A misses **{100*(1-a_metrics['real_recall']):.1f}%** of real images ({a_metrics['fp']} of 171 real images labelled fake). Baseline B reduces that to **{100*(1-b_metrics['real_recall']):.1f}%** ({b_metrics['fp']} of 171).

**FACT:** ROC-AUC is nearly identical between A ({a_metrics['roc_auc']:.4f}) and B ({b_metrics['roc_auc']:.4f}), and MCC is nearly identical ({a_metrics['mcc']:.4f} vs {b_metrics['mcc']:.4f}). The ranking quality of the two models is comparable; the difference is where the decision threshold effectively sits.

**INFERENCE:** Class weighting mainly moves the operating point rather than improving the underlying representation. Balanced accuracy improves by {b_metrics['balanced_accuracy']-a_metrics['balanced_accuracy']:+.4f} at the cost of {b_metrics['fn']-a_metrics['fn']:+d} additional false negatives.

**PR-AUC near {b_metrics['pr_auc']:.3f} is not impressive here** — the positive class is 96% of the test set, so a trivial classifier already gets high average precision. ROC-AUC and MCC are the more informative headline numbers.

---

## 4. Recall Balance

| Baseline | Real Recall | Fake Recall | Gap |
|---|---|---|---|
| A | {a_metrics['real_recall']:.4f} | {a_metrics['fake_recall']:.4f} | {a_metrics['fake_recall']-a_metrics['real_recall']:.4f} |
| B | {b_metrics['real_recall']:.4f} | {b_metrics['fake_recall']:.4f} | {b_metrics['fake_recall']-b_metrics['real_recall']:.4f} |

**FACT:** Neither baseline is balanced. The real class remains substantially harder in both.

---

## 5. Dominant Error Types

| Baseline | TN | FP (real→fake) | FN (fake→real) | TP |
|---|---|---|---|---|
| A | {a_metrics['tn']} | {a_metrics['fp']} | {a_metrics['fn']} | {a_metrics['tp']} |
| B | {b_metrics['tn']} | {b_metrics['fp']} | {b_metrics['fn']} | {b_metrics['tp']} |

**Baseline A:** dominated by **false positives** — {a_metrics['fp']} real images misclassified as fake against only {a_metrics['fn']} false negatives. The model has largely learned "predict fake".

**Baseline B:** more symmetric — {b_metrics['fp']} FP and {b_metrics['fn']} FN. Class weighting shifted errors from the real class onto the fake class.

**INFERENCE:** The real class is the harder class in both settings. With 826 real training images versus 20,165 fake, this is consistent with the training distribution, though this analysis does not prove causality.

---

## 6. Weakest Fake Methods (Baseline B)

Methods with F1 below 1.0, sorted ascending:

{md_table(weak_b, cols=['method', 'sample_count', 'accuracy', 'recall', 'f1', 'false_negative_rate', 'average_confidence'])}

Full table: `experiments/results/baseline/eval_b/per_method_metrics.csv`

**FACT:** `real` has the lowest F1 in both baselines. Among fake methods, the weakest are face-swap style methods.

**FACT:** Most methods reach F1 = 1.0 on the test set. Per-method test counts are small (typically 65–225 images), so individual method F1 values have wide confidence intervals and should not be over-interpreted.

---

## 7. Errors vs Dataset Properties (Associations Only)

Join coverage against `sample_quality.csv`: **{coverage:.1%}** of test predictions matched.

{md_table(qa)}

**These are associations, not causes.**

### Error concentration (Baseline B, top groups)

```json
{json.dumps(conc, indent=2)}
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

**Baseline B (class-weighted) is the better reference baseline** for balanced accuracy ({b_metrics['balanced_accuracy']:.4f} vs {a_metrics['balanced_accuracy']:.4f}) and real recall ({b_metrics['real_recall']:.4f} vs {a_metrics['real_recall']:.4f}), with essentially identical ROC-AUC and MCC.

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
"""
    (PROJECT_ROOT / "BASELINE_REPORT.md").write_text(report)
    print("Wrote BASELINE_REPORT.md")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()

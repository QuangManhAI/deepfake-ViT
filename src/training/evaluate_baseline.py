"""Evaluate a fine-tuned baseline and produce the required reports + figures."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import DinoViTClassifier, EVAL_TF, IMG_SIZE, MEAN, STD, load_dinov3  # type: ignore


class MetadataImageDataset(Dataset):
    """Image dataset that also returns path and metadata."""

    def __init__(self, csv_path, detailed_csv, transform=None):
        self.transform = transform
        with open(csv_path, newline="", encoding="utf-8") as f:
            import csv

            self.rows = [(r["path"], int(r["label"])) for r in csv.DictReader(f)]
        with open(detailed_csv, newline="", encoding="utf-8") as f:
            detailed = {r["path"]: r for r in csv.DictReader(f)}
        self.detailed = [detailed[p] for p, _ in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        img = Image.open(path).convert("RGB") if Path(path).is_absolute() else Image.open(PROJECT_ROOT / path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path


def find_run_dir(output_dir, run_name):
    base = Path(output_dir)
    dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.endswith(run_name)], reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No run directory found for {run_name} in {output_dir}")
    return dirs[0]


def load_model(checkpoint_path, device, weights_path=None):
    weights = weights_path or str(PROJECT_ROOT / "experiments" / "checkpoints" / "weights" / "model.safetensors")
    backbone = load_dinov3(weights, img_size=IMG_SIZE)
    model = DinoViTClassifier(backbone).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[key])
    model.eval()
    return model


def per_method_metrics(df):
    methods = sorted(df["method"].unique())
    rows = []
    for method in methods:
        d = df[df["method"] == method]
        y = d["label"].values
        p = d["predicted_label"].values
        pos = 0 if method == "real" else 1
        cm = confusion_matrix(y, p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "method": method,
            "sample_count": len(d),
            "accuracy": accuracy_score(y, p),
            "precision": precision_score(y, p, pos_label=pos, zero_division=0),
            "recall": recall_score(y, p, pos_label=pos, zero_division=0),
            "f1": f1_score(y, p, pos_label=pos, zero_division=0),
            "false_negative_rate": fn / max(1, (fn + tp)) if pos == 1 else fp / max(1, (fp + tn)),
            "false_positive_rate": fp / max(1, (fp + tn)) if pos == 1 else fn / max(1, (fn + tp)),
            "average_confidence": d["probability_fake"].mean(),
        })
    out = pd.DataFrame(rows).sort_values("f1", ascending=True)
    return out


def plot_confusion_matrix(y, pred, out_path, normalize=False):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(y, prob, out_path):
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr(y, prob, out_path):
    precision, recall, _ = precision_recall_curve(y, prob)
    ap = average_precision_score(y, prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"PR-AUC = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(history_path, out_path):
    records = []
    with open(history_path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if "train_loss" in df.columns:
        axes[0, 0].plot(df["epoch"], df["train_loss"], label="train")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
    if "val_loss" in df.columns:
        axes[0, 1].plot(df["epoch"], df["val_loss"], label="val")
        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
    if "val_acc" in df.columns:
        axes[1, 0].plot(df["epoch"], df["val_acc"], label="val_acc")
        axes[1, 0].set_title("Validation Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()
    if "val_f1" in df.columns:
        axes[1, 1].plot(df["epoch"], df["val_f1"], label="val_f1")
        axes[1, 1].set_title("Validation F1")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("F1")
        axes[1, 1].legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_test_score_distribution(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [(0, "green"), (1, "red")]:
        d = df[df["label"] == label]["probability_fake"]
        ax.hist(d, bins=50, alpha=0.6, label=f"{'Real' if label == 0 else 'Fake'} (n={len(d)})", color=color)
    ax.set_xlabel("Probability Fake")
    ax.set_ylabel("Count")
    ax.set_title("Test Score Distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_analysis(y, prob, out_path):
    thresholds = np.linspace(0, 1, 101)
    f1s, bas = [], []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        f1s.append(f1_score(y, pred, zero_division=0))
        bas.append(balanced_accuracy_score(y, pred))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s, label="F1")
    ax.plot(thresholds, bas, label="Balanced Accuracy")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold vs Metric")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate(run_dir, output_dir, device="mps", batch_size=32, num_workers=0):
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load protocol config for test CSVs
    with open(PROJECT_ROOT / "data" / "protocol" / "protocol_config.json") as f:
        protocol = json.load(f)

    # Load training config/hyperparameters
    run_name = run_dir.name.split("_", 2)[-1] if run_dir.name.count("_") >= 2 else "baseline"
    config_path = run_dir / "metrics" / f"{run_dir.name.split('_', 2)[2] if run_dir.name.count('_') >= 2 else 'run'}_config.json"
    # Try generic config
    config_candidates = list(run_dir.glob("metrics/*_config.json"))
    train_config = json.loads(config_candidates[0].read_text()) if config_candidates else {}

    test_csv = protocol["test_csv"]
    test_detailed = str(Path(test_csv).with_stem(Path(test_csv).stem + "_detailed"))

    # Data
    test_ds = MetadataImageDataset(test_csv, test_detailed, transform=EVAL_TF)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    with open(test_detailed, newline="", encoding="utf-8") as f:
        import csv
        path_to_meta = {r["path"]: r for r in csv.DictReader(f)}

    # Model
    best_ckpt = run_dir / "checkpoints" / f"{train_config.get('run_name', 'dinov3_finetuned')}_best.pt"
    if not best_ckpt.exists():
        best_ckpt = next(run_dir.glob("checkpoints/*_best.pt"))
    device = torch.device(device if device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = load_model(str(best_ckpt), device)

    # Predict
    all_y, all_prob, all_pred = [], [], []
    all_paths = []
    with torch.no_grad():
        for x, y, paths in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(1)
            all_y.extend(y.tolist())
            all_prob.extend(probs[:, 1].tolist())
            all_pred.extend(pred.tolist())
            all_paths.extend(paths)

    # Build predictions DataFrame
    metas = [path_to_meta.get(p, {}) for p in all_paths]
    df = pd.DataFrame({
        "path": all_paths,
        "label": all_y,
        "predicted_label": all_pred,
        "probability_fake": all_prob,
        "confidence": [max(p, 1 - p) for p in all_prob],
        "method": [m.get("method", "") for m in metas],
        "identity": [m.get("identity", "") for m in metas],
        "video": [m.get("video", "") for m in metas],
        "domain": [m.get("domain", "") for m in metas],
    })
    df["correct"] = (df["label"] == df["predicted_label"]).astype(int)
    df["error_type"] = "CORRECT"
    df.loc[(df["label"] == 0) & (df["predicted_label"] == 1), "error_type"] = "FALSE_POSITIVE"
    df.loc[(df["label"] == 1) & (df["predicted_label"] == 0), "error_type"] = "FALSE_NEGATIVE"
    df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Overall metrics
    y = df["label"].values
    pred = df["predicted_label"].values
    prob = df["probability_fake"].values
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "real_precision": precision_score(y, pred, pos_label=0, zero_division=0),
        "real_recall": recall_score(y, pred, pos_label=0, zero_division=0),
        "fake_precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "fake_recall": recall_score(y, pred, pos_label=1, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # Per-method metrics
    pmm = per_method_metrics(df)
    pmm.to_csv(output_dir / "per_method_metrics.csv", index=False)

    # Plots
    plot_confusion_matrix(y, pred, output_dir / "confusion_matrix.png")
    plot_confusion_matrix(y, pred, output_dir / "confusion_matrix_normalized.png", normalize=True)
    plot_roc(y, prob, output_dir / "roc_curve.png")
    plot_pr(y, prob, output_dir / "pr_curve.png")
    plot_test_score_distribution(df, output_dir / "test_score_distribution.png")
    plot_threshold_analysis(y, prob, output_dir / "threshold_analysis.png")

    # Loss curves from history
    history_files = list(run_dir.glob("metrics/*_history.jsonl"))
    if history_files:
        plot_loss_curves(history_files[0], output_dir / "loss_curve.png")

    # Save metrics and config
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "protocol": protocol,
            "train_config": train_config,
            "best_checkpoint": str(best_ckpt),
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)

    # README
    readme = f"""# Baseline Evaluation

## Run
- Checkpoint: `{best_ckpt}`
- Test CSV: `{test_csv}`
- Evaluated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Overall Test Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1 | {metrics['f1']:.4f} |
| MCC | {metrics['mcc']:.4f} |
| ROC-AUC | {metrics['roc_auc']:.4f} |
| PR-AUC | {metrics['pr_auc']:.4f} |
| Real Precision | {metrics['real_precision']:.4f} |
| Real Recall | {metrics['real_recall']:.4f} |
| Fake Precision | {metrics['fake_precision']:.4f} |
| Fake Recall | {metrics['fake_recall']:.4f} |
| TN/FP/FN/TP | {metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']} |

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
"""
    (output_dir / "README.md").write_text(readme)
    print(f"Saved baseline outputs to: {output_dir}")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Training run directory")
    parser.add_argument("--output-dir", default="experiments/results/baseline/evaluation", help="Output directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args.run_dir, args.output_dir, device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
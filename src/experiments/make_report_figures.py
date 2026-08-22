"""Sinh các figure cho coursework report (PDF LaTeX) từ các file JSON kết quả.

Không load model / không cần GPU — chỉ đọc JSON và vẽ matplotlib (nhẹ, không lo tràn RAM).

Output: experiments/results/report/figures/*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_HERE, "..", "..")   # repo root
REPORT_DIR = os.path.join(ROOT, "experiments", "results", "report")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

VIT_COLOR = "#1f77b4"   # blue
CNN_COLOR = "#d62728"   # red
GRAY = "#7f7f7f"


def load(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: Fine-tune loss + val accuracy (ViT) — từ finetune_insight_report.json
# ---------------------------------------------------------------------------
def fig_finetune_loss():
    d = load(os.path.join(ROOT, "experiments", "results", "finetune_insight_report.json"))
    hist = d["history"]
    epochs = [h["epoch"] for h in hist]
    train_loss = [h["train_loss"] for h in hist]
    val_acc = [h["accuracy"] * 100 for h in hist]
    val_f1 = [h["f1"] * 100 for h in hist]

    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    ax1.plot(epochs, train_loss, "o-", color=VIT_COLOR, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color=VIT_COLOR)
    ax1.tick_params(axis="y", labelcolor=VIT_COLOR)
    ax1.set_xticks(epochs)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, "s--", color=CNN_COLOR, label="Val accuracy")
    ax2.plot(epochs, val_f1, "^--", color=GRAY, label="Val F1")
    ax2.set_ylabel("Validation (%)", color=CNN_COLOR)
    ax2.tick_params(axis="y", labelcolor=CNN_COLOR)
    ax2.set_ylim(85, 100)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", frameon=False)
    ax1.set_title("Fine-tuning DINOv3 ViT-S/16 (5 epochs)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "finetune_loss.png")
    fig.savefig(out)
    plt.close(fig)
    print("Wrote", out)


# ---------------------------------------------------------------------------
# Figure 2: Linear-probe ViT vs CNN (DeepFakeFace) — comparison_report_v2.json
# ---------------------------------------------------------------------------
def fig_linprobe():
    d = load(os.path.join(ROOT, "experiments", "results", "comparison_report_v2.json"))
    vit = d["models"]["vit"]["metrics"]
    cnn = d["models"]["cnn"]["metrics"]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    vit_vals = [vit[m] * 100 for m in metrics]
    cnn_vals = [cnn[m] * 100 for m in metrics]

    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    b1 = ax.bar(x - w / 2, vit_vals, w, label="ViT-S/16 (Plus)", color=VIT_COLOR)
    b2 = ax.bar(x + w / 2, cnn_vals, w, label="ConvNeXt-Tiny", color=CNN_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7.5)
    ax.set_title("Linear probe on DeepFakeFace (identity-disjoint test)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "linprobe_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print("Wrote", out)


# ---------------------------------------------------------------------------
# Figure 3: DeepfakeTIMIT detection rate — deepfaketimit_report.json
# ---------------------------------------------------------------------------
def fig_timit():
    d = load(os.path.join(ROOT, "experiments", "results", "benchmark", "deepfaketimit_report.json"))
    vit = d["vit"]
    cnn = d["cnn"]
    groups = ["Overall", "HQ (128px)", "LQ (64px)"]
    vit_vals = [vit["detection_rate"] * 100, vit["hq_detection_rate"] * 100, vit["lq_detection_rate"] * 100]
    cnn_vals = [cnn["detection_rate"] * 100, cnn["hq_detection_rate"] * 100, cnn["lq_detection_rate"] * 100]

    x = np.arange(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    b1 = ax.bar(x - w / 2, vit_vals, w, label="ViT-S/16 (Plus)", color=VIT_COLOR)
    b2 = ax.bar(x + w / 2, cnn_vals, w, label="ConvNeXt-Tiny", color=CNN_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Detection rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower left")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_title("DeepfakeTIMIT (all frames fake)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "timit_detection.png")
    fig.savefig(out)
    plt.close(fig)
    print("Wrote", out)


# ---------------------------------------------------------------------------
# Figure 4: Confidence histogram p(fake) — DeepfakeTIMIT
# ---------------------------------------------------------------------------
def fig_timit_hist():
    d = load(os.path.join(ROOT, "experiments", "results", "benchmark", "deepfaketimit_report.json"))
    vit = np.array(d["vit"]["prob_fake"])
    cnn = np.array(d["cnn"]["prob_fake"])
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    axes[0].hist(vit, bins=bins, color=VIT_COLOR, alpha=0.85)
    axes[0].set_title("ViT-S/16 (Plus)")
    axes[0].set_xlabel("Predicted p(fake)")
    axes[1].hist(cnn, bins=bins, color=CNN_COLOR, alpha=0.85)
    axes[1].set_title("ConvNeXt-Tiny")
    axes[1].set_xlabel("Predicted p(fake)")
    axes[0].set_ylabel("Number of frames")
    for ax in axes:
        ax.axvline(0.5, color="k", ls="--", lw=0.8)
    fig.suptitle("Confidence distribution on DeepfakeTIMIT (640 fake frames)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "timit_prob_hist.png")
    fig.savefig(out)
    plt.close(fig)
    print("Wrote", out)


# ---------------------------------------------------------------------------
# Figure 5: Confusion matrix — fine-tune test result
# ---------------------------------------------------------------------------
def fig_finetune_cm():
    d = load(os.path.join(ROOT, "experiments", "results", "finetune_insight_report.json"))
    cm = np.array(d["test"]["confusion_matrix"])  # [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Real", "Pred Fake"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Real", "Fake"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_title("Fine-tuned ViT test set (n=11920)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "finetune_cm.png")
    fig.savefig(out)
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    fig_finetune_loss()
    fig_linprobe()
    fig_timit()
    fig_timit_hist()
    fig_finetune_cm()
    print("Done.")

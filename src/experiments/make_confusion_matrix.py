"""Xuất ma trận nhầm lẫn từ feature cache của eval_df40_all_methods.

Đọc feature memmap + meta, tái tạo logistic-regression pipeline GIỐNG HỆT
eval_df40_all_methods (StandardScaler + LogisticRegression balanced,
max_iter=2000, seed=42) rồi predict trên test -> ma trận nhầm lẫn 2x2
(real/fake) cho từng model. Lưu figure + in số ra console.

Chạy:
  .venv/bin/python src/experiments/make_confusion_matrix.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = "data/df40_subset"
FEATS = "experiments/results/features"
RESULT = "experiments/results/df40_all_methods_report.json"
FIG = "experiments/results/df40_confusion_matrix.png"

MODELS = [
    {"key": "vit", "label": "DINOv3 ViT-S/16 Plus", "dim": 384},
    {"key": "cnn", "label": "DINOv3 ConvNeXt-Tiny", "dim": 768},
]


def load(key, dim, split):
    suffix = "" if split == "" else f"_{split}"
    mmap = np.memmap(os.path.join(FEATS, f"df40_{key}{suffix}.mmap"),
                     dtype=np.float32, mode="r")
    meta = np.load(os.path.join(FEATS, f"df40_{key}{suffix}.mmap.meta.npz"))
    n = meta["labels"].shape[0]
    # file được allocate theo n_full (có thể > n do ảnh lỗi không được ghi)
    n_full = mmap.size // dim
    X = np.asarray(mmap.reshape(n_full, dim)[:n])
    return X, meta["labels"], meta["methods"]


def main():
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    report = json.load(open(RESULT))
    print(f"Split: train={report['split']['train']} test={report['split']['test']}\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    summary = {}
    for ax, cfg in zip(axes, MODELS):
        X_tr, y_tr, _ = load(cfg["key"], cfg["dim"], split="")
        X_te, y_te, m_te = load(cfg["key"], cfg["dim"], split="test")
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(class_weight="balanced",
                                               max_iter=2000, random_state=42))
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        cm = confusion_matrix(y_te, y_pred)  # rows=real, cols=pred; labels=[0,1]
        tn, fp, fn, tp = cm.ravel()
        acc = (tn + tp) / cm.sum()
        summary[cfg["key"]] = {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "acc": float(acc),
        }

        # --- vẽ heatmap ---
        norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{cfg['label']}\nAcc={acc:.4f} (n={len(y_te)})", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"]); ax.set_yticklabels(["Real", "Fake"])
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.0%})",
                        ha="center", va="center", fontsize=10,
                        color="white" if norm[i, j] > 0.5 else "black")
        # nhãn ô góc
        ax.text(-0.35, 0.5, "TN/FN", ha="center", va="center", rotation=90,
                fontsize=8, color="gray", transform=ax.transAxes)
        print(f"--- {cfg['label']} ---")
        print(f"  TN(real->real)={tn}  FP(real->fake)={fp}")
        print(f"  FN(fake->real)={fn}  TP(fake->fake)={tp}")
        print(f"  Accuracy={acc:.4f}\n")

    fig.suptitle("Ma trận nhầm lẫn — DF40 subset cân bằng (8 method fake)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG, dpi=150)
    print(f"Đã lưu: {FIG}")

    # --- ma trận chi tiết theo từng method fake (fake -> real = miss) ---
    print("=== Detection rate theo method (fake bị nhầm thành real) ===")
    for cfg in MODELS:
        X_tr, y_tr, _ = load(cfg["key"], cfg["dim"], split="")
        X_te, y_te, m_te = load(cfg["key"], cfg["dim"], split="test")
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(class_weight="balanced",
                                               max_iter=2000, random_state=42))
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        print(f"\n{cfg['label']}:")
        for m in sorted(set(m_te)):
            mask = m_te == m
            n = int(mask.sum())
            miss = int(((y_pred[mask] == 0) & (y_te[mask] == 1)).sum())
            print(f"  {m:<16} n={n:>4}  nhầm thành real={miss:>4}  ({miss/max(1,n):.1%})")

    with open(RESULT.replace(".json", "_confusion.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nĐã lưu summary: {RESULT.replace('.json', '_confusion.json')}")


if __name__ == "__main__":
    main()

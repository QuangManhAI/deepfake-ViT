"""Test riêng trên FAKE rồi trên REAL (frozen backbone + LR probe), tái dùng cache.

Đọc feature memmap đã extract (experiments/results/features/{tag}_{model}.mmap) — không extract lại.
Fit LR probe trên train (như eval_df40_all_methods), rồi đo 2 phần test riêng biệt:
  - FAKE-only (label=1): detection accuracy, precision, recall, F1, per-method
  - REAL-only (label=0): real recognition (true-negative rate) + FPR

Chạy:
  .venv/bin/python src/eval/eval_split_real_fake.py --model vit --tag test_data
  .venv/bin/python src/eval/eval_split_real_fake.py --model cnn --tag test_data
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEAT_DIM = {"vit": 384, "cnn": 768}
CACHE_DIR = "experiments/results/features"


def load(tag, key, suffix):
    mm = os.path.join(CACHE_DIR, f"{tag}_{key}{suffix}.mmap")
    meta = np.load(mm + ".meta.npz")
    X = np.memmap(mm, dtype=np.float32, mode="r", shape=(meta["labels"].shape[0],
                                                          FEAT_DIM[key]))
    return X, meta["labels"], meta["methods"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vit", "cnn"])
    ap.add_argument("--tag", default="test_data")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    X_tr, y_tr, _ = load(args.tag, args.model, "")
    X_te, y_te, m_te = load(args.tag, args.model, "_test")
    print(f"Cache {args.model}: train {len(y_tr):,} ({y_tr.sum():,} fake) | "
          f"test {len(y_te):,} ({int((y_te==1).sum()):,} fake, {int((y_te==0).sum()):,} real)")

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=2000,
                                           random_state=42))
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    out = {"model": args.model, "tag": args.tag}

    # ---- FAKE only ----
    mf = y_te == 1
    yf, pf = y_te[mf], y_pred[mf]
    fake_metrics = {
        "n": int(mf.sum()),
        "accuracy": float(accuracy_score(yf, pf)),           # = detection rate
        "precision": float(precision_score(yf, pf, zero_division=0)),
        "recall": float(recall_score(yf, pf, zero_division=0)),
        "f1": float(f1_score(yf, pf, zero_division=0)),
    }
    # per-method detection (chỉ trên fake test)
    per_m = {}
    for m in sorted(set(m_te[mf])):
        mask = m_te[mf] == m
        per_m[m] = {"n": int(mask.sum()),
                    "detection_rate": float((pf[mask] == 1).mean())}
    fake_metrics["per_method"] = per_m
    out["fake"] = fake_metrics

    # ---- REAL only ----
    mr = y_te == 0
    yr, pr = y_te[mr], y_pred[mr]
    real_metrics = {
        "n": int(mr.sum()),
        "real_accuracy": float(accuracy_score(yr, pr)),       # = true-negative rate
        "fpr": float((pr == 1).mean()),                        # false positive rate
    }
    out["real"] = real_metrics
    out["overall_auc"] = float(roc_auc_score(y_te, y_prob))   # toàn bộ test (2 class)

    # ---- in ----
    print(f"\n=== {args.model.upper()} — FAKE-ONLY ({fake_metrics['n']:,} ảnh) ===")
    print(f"  Detection Acc={fake_metrics['accuracy']:.4f} Prec={fake_metrics['precision']:.4f} "
          f"Rec={fake_metrics['recall']:.4f} F1={fake_metrics['f1']:.4f}")
    print(f"  {'method':12s}{'n':>6s}{'det':>8s}")
    for m, v in sorted(per_m.items()):
        print(f"  {m:12s}{v['n']:6d}{v['detection_rate']:8.3f}")
    print(f"\n=== {args.model.upper()} — REAL-ONLY ({real_metrics['n']:,} ảnh) ===")
    print(f"  Real acc (đúng real)={real_metrics['real_accuracy']:.4f} "
          f"FPR={real_metrics['fpr']:.4f} | Overall AUC={out['overall_auc']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

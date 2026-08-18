"""Đánh giá hiệu năng model bằng linear probe.

Đặc trưng được trích riêng từng split (train/val/test) — đảm bảo không có
identity leakage. Quy trình:
  1. Load features train (đã trích từ backbone)
  2. Train LogisticRegression (linear probe, class_weight='balanced') trên train
  3. Đánh giá trên test (báo cáo cả val nếu có): acc, precision, recall, F1, ROC-AUC, confusion matrix
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = {0: "real", 1: "fake"}


def load_features(path: str):
    data = np.load(path, allow_pickle=True)
    X, y = data["features"], data["labels"]
    print(f"  {os.path.basename(path)}: {X.shape} | real={int((y == 0).sum())}, fake={int((y == 1).sum())}")
    return X, y


def eval_split(clf, X, y):
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }, y_pred, y_prob


def print_results(name: str, m: dict):
    cm = m["confusion_matrix"]
    print(f"\n--- {name.upper()} ---")
    print(f"  accuracy : {m['accuracy']:.4f}")
    print(f"  precision: {m['precision']:.4f}   (fake=positive)")
    print(f"  recall   : {m['recall']:.4f}")
    print(f"  f1       : {m['f1']:.4f}")
    print(f"  roc_auc  : {m['roc_auc']:.4f}")
    print(f"  CM [TN FP; FN TP]: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")


def main():
    parser = argparse.ArgumentParser(description="Linear probe evaluation (train / eval tách riêng)")
    parser.add_argument("--train-features", required=True, help="features.npz của tập train")
    parser.add_argument("--test-features", required=True, help="features.npz của tập test")
    parser.add_argument("--val-features", default=None, help="features.npz của tập val (tùy chọn)")
    parser.add_argument("--output", default="experiments/results/evaluation_report.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Load features:")
    X_train, y_train = load_features(args.train_features)
    X_test, y_test = load_features(args.test_features)
    X_val, y_val = (load_features(args.val_features) if args.val_features else (None, None))

    # ---------- Train linear probe ----------
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed),
    )
    clf.fit(X_train, y_train)
    print("\nĐã train xong linear probe.")

    # ---------- Đánh giá ----------
    report = {"seed": args.seed, "splits": {}}
    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        if X is None:
            continue
        m, _, _ = eval_split(clf, X, y)
        report["splits"][name] = {"metrics": m}
        print_results(name, m)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu báo cáo: {args.output}")


if __name__ == "__main__":
    main()

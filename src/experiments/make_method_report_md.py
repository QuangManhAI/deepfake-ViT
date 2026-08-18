"""Tạo report Markdown kết quả theo TỪNG METHOD (40 method) từ eval JSON v3.

Đọc 2 file JSON của eval_identity_disjoint (vit + cnn) trên test_data_v3,
gộp thành bảng markdown. Vì 1 method có thể trải nhiều domain (cdc/ffc/oth),
report dùng trường `per_method_domain` để tách từng (method, domain).

Chạy:
  .venv/bin/python src/experiments/make_method_report_md.py \
      --vit experiments/results/eval/identity_disjoint_v3_vit.json \
      --cnn experiments/results/eval/identity_disjoint_v3_cnn.json \
      --output experiments/results/eval/report_40_methods_v3.md
"""
import argparse
import json
import os


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit", required=True)
    ap.add_argument("--cnn", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    V = load(args.vit)
    C = load(args.cnn)

    # ---- 40 method, tách theo domain ----
    # keys "method/domain" (loại real/*)
    md_keys = sorted(k for k in V["per_method_domain"] if not k.startswith("real/"))
    rows = []
    for k in md_keys:
        m, d = k.split("/", 1)
        v = V["per_method_domain"][k]
        c = C["per_method_domain"].get(k, {})
        rows.append((m, d, v["n"], v["rate"], c.get("rate")))
    # gộp 1 method nhiều domain thành 1 khối liên tục, method sắp alphabet
    rows.sort(key=lambda r: (r[0].lower(), r[1]))

    # ---- real theo domain ----
    real_d = {}
    for d, v in V.get("per_domain", {}).items():
        real_d[d] = {"n": v.get("n"), "n_real": None, "vit": v.get("real_acc"),
                     "cnn": C.get("per_domain", {}).get(d, {}).get("real_acc")}
    # n_real từ per_method_domain real/{d}
    for k, v in V["per_method_domain"].items():
        if k.startswith("real/"):
            d = k.split("/", 1)[1]
            if d in real_d:
                real_d[d]["n_real"] = v["n"]

    L = []
    L.append("# Kết quả 40 method — DINOv3 ViT-S/16 vs ConvNeXt-Tiny")
    L.append("")
    L.append("- **Dataset:** `test_data_v3/` — cấu trúc `real/` + `<method>/fake/`")
    L.append("- **Protocol:** identity-disjoint (split seed 42, train_ratio 0.7, theo cột `identity`) — "
             "model test chỉ trên identity chưa từng thấy trong train")
    L.append("- **Probe:** frozen backbone (feature extractor) + LogisticRegression linear probe "
             "(class_weight=balanced)")
    L.append(f"- **Train/test:** {V.get('n_train'):,} / {V.get('n_test'):,} ảnh, "
             f"{V.get('n_identity_keys'):,} identity keys")
    L.append("")

    m = V["metrics"]; cm = C["metrics"]
    L.append("## Tổng quan")
    L.append("")
    L.append("| Metric | DINOv3 ViT-S/16 | ConvNeXt-Tiny |")
    L.append("|---|---|---|")
    for k, label in [("accuracy", "Accuracy"), ("precision", "Precision"),
                     ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "AUC")]:
        L.append(f"| {label} | {m[k]:.4f} | {cm[k]:.4f} |")
    L.append(f"| Nhận đúng REAL (acc) | {V['real_acc']:.4f} (FPR {1-V['real_acc']:.1%}) | "
             f"{C['real_acc']:.4f} (FPR {1-C['real_acc']:.1%}) |")
    L.append(f"| Bắt FAKE (det) | {V['fake_detection']:.4f} | {C['fake_detection']:.4f} |")
    if V.get("paired_only"):
        p, cp = V["paired_only"], C.get("paired_only") or {}
        L.append(f"| Paired-only Acc | {p['accuracy']:.4f} | "
                 f"{cp.get('accuracy', float('nan')):.4f} |")
    L.append("")

    L.append("## Theo domain (test)")
    L.append("")
    L.append("| Domain | n | ViT acc | ViT real | ViT fake | CNN acc | CNN real | CNN fake |")
    L.append("|---|---|---|---|---|---|---|---|")
    for d in sorted(V.get("per_domain", {})):
        v = V["per_domain"][d]; c = C["per_domain"].get(d, {})
        L.append(f"| {d} | {v['n']} | {v.get('acc', float('nan')):.4f} | "
                 f"{v.get('real_acc', float('nan')):.4f} | {v.get('fake_det', float('nan')):.4f} | "
                 f"{c.get('acc', float('nan')):.4f} | {c.get('real_acc', float('nan')):.4f} | "
                 f"{c.get('fake_det', float('nan')):.4f} |")
    L.append("")

    L.append("## 40 method — detection rate (fake) theo (method, domain)")
    L.append("")
    L.append("Domain: **cdc**=Celeb-DF · **ffc**=FF++ · **efs**=tổng hợp · "
             "**oth**=không ghép · **fe**=expression. Một method có thể trải nhiều domain.")
    L.append("")
    L.append("| Method | domain | n | ViT det | CNN det |")
    L.append("|---|---|---:|---:|---:|")
    for m, d, n, vd, cd in rows:
        cd_s = f"{cd:.4f}" if cd is not None else "—"
        L.append(f"| {m} | {d} | {n:,} | {vd:.4f} | {cd_s} |")
    L.append("")

    L.append("## REAL theo domain")
    L.append("")
    L.append("| Domain | n real | ViT real acc | CNN real acc |")
    L.append("|---|---|---:|---:|")
    for d in sorted(real_d):
        r = real_d[d]
        if r["n_real"] is None or r["vit"] is None:
            continue  # domain không có ảnh real
        cnn_s = f"{r['cnn']:.4f}" if r["cnn"] is not None else "—"
        L.append(f"| {d} | {r['n_real']} | {r['vit']:.4f} | {cnn_s} |")
    L.append("")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Saved: {args.output} ({len(md_keys)} dòng method/domain)")


if __name__ == "__main__":
    main()

"""Phân tích ngưỡng quyết định tối ưu cho model LoRA finetuned.

Model có AUC 0.92 (xếp hạng tốt) nhưng real/ffc = 0.27 ở ngưỡng 0.5. Script này:
  - chạy lại forward trên eval-test (cùng protocol identity-disjoint seed 42)
  - dump P(fake) của từng ảnh
  - sweep ngưỡng → tìm điểm cân bằng real_acc vs fake_det (Youden J / balanced acc)

Chạy: .venv/bin/python scripts/analyze_threshold.py --ckpt outputs/finetune/vit_lora_finetuned.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_finetuned_identity_disjoint import (
    BackboneClassifier, EVAL_TF, identity_disjoint_split, read_manifest,
)
from src.models.dinov3_vit import DinoViT
from src.models.lora import apply_lora

IMG_SIZE = 256


@torch.no_grad()
def dump_probs(model, items, device, batch_size=64):
    model.eval()
    y_all, prob_all, m_all, d_all = [], [], [], []
    bad = 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        xs, ys, ms, ds = [], [], [], []
        for p, mth, v, lab, ident, dom in batch:
            try:
                xs.append(EVAL_TF(__import__("PIL").Image.open(p).convert("RGB")))
                ys.append(lab); ms.append(mth); ds.append(dom)
            except Exception:
                bad += 1
        if not xs:
            continue
        x = torch.stack(xs).to(device)
        probs = torch.softmax(model(x), dim=1)
        prob_all.extend(probs[:, 1].tolist())
        y_all.extend(ys); m_all.extend(ms); d_all.extend(ds)
    return np.array(y_all), np.array(prob_all), np.array(m_all), np.array(d_all), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="test_data_v3")
    ap.add_argument("--ckpt", default="outputs/finetune/vit_lora_finetuned.pt")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu")
    backbone = DinoViT(img_size=IMG_SIZE, gated_mlp=True)
    if "lora_config" in ck:
        lc = ck["lora_config"]
        apply_lora(backbone, r=lc.get("r", 16), alpha=lc.get("alpha", 32.0),
                   targets=lc.get("targets", ("q_proj", "v_proj")))
    model = BackboneClassifier(backbone)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(device).eval()

    items = read_manifest(args.root)
    _, test, _ = identity_disjoint_split(items, 0.7, seed=42)
    y, prob, ms, ds, bad = dump_probs(model, test, device, args.batch_size)
    print(f"test: {len(y):,} (bad={bad}) | real={int((y==0).sum())} fake={int((y==1).sum())}")
    np.savez("outputs/eval/lora_probs.npz", y=y, prob=prob, ms=ms, ds=ds)

    print(f"\n{'thr':>5s}{'real_acc':>9s}{'fake_det':>9s}{'bal_acc':>9s}{'F1':>7s}")
    best = None
    for t in np.arange(0.05, 0.96, 0.05):
        pred = (prob >= t).astype(int)
        ra = float((pred[y == 0] == 0).mean())
        fd = float((pred[y == 1] == 1).mean())
        ba = (ra + fd) / 2
        tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0
        print(f"{t:.2f}{ra:9.3f}{fd:9.3f}{ba:9.3f}{f1:7.3f}")
        if best is None or ba > best[1]:
            best = (t, ba, ra, fd)
    print(f"\nNgưỡng cân bằng tốt nhất: thr={best[0]:.2f} bal_acc={best[1]:.3f} "
          f"(real_acc={best[2]:.3f}, fake_det={best[3]:.3f})")

    # per-domain ở ngưỡng 0.5 và ngưỡng best
    WEAK = {"faceswap", "facedancer", "inswap", "fsgan", "simswap", "blendface",
            "pixart", "DiT", "uniface", "SiT", "lia", "mobileswap", "MRAA", "e4s"}
    for thr in (0.5, best[0]):
        pred = (prob >= thr).astype(int)
        print(f"\n--- thr={thr:.2f} ---")
        for dom in sorted(set(ds[ms == "real"])):
            m = (ms == "real") & (ds == dom)
            print(f"  real/{dom}: n={int(m.sum())} acc_real={(pred[m]==0).mean():.3f}")
        # 14 method yếu — det gộp + từng method
        wk = np.isin(ms, list(WEAK)) & (ms != "real")
        print(f"  [14 method yếu] n={int(wk.sum())} det={(pred[wk]==1).mean():.3f}")
        for mth in sorted(WEAK):
            m = ms == mth
            if m.sum():
                print(f"    {mth:12s} n={int(m.sum())} det={(pred[m]==1).mean():.3f}")


if __name__ == "__main__":
    main()

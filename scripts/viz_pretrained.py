#!/usr/bin/env python3
"""Figure pretrained vs finetuned trên test cân bằng 21,446.

Đọc preds npz trong experiments/results/coursework_vs/ (các tag có sẵn đều vẽ được):
  - Pretr_Plus_v3 / Pretr_ConvNeXt_v3 : linear probe trên backbone pretrained
  - Plus_viT_v3 (A0) / plus_v3_s1 (A1) / ConvNeXt_v3 (finetuned)
Vẽ: tổng thể; det-rate per-method (38 fake); det-rate theo 4 nhóm deepfake;
real_acc theo source. Ghi ra agents/figures/session2/fig09..fig12.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results", "coursework_vs")
FIG = os.path.join(ROOT, "agents", "figures", "session2")
os.makedirs(FIG, exist_ok=True)

TAGS = ["Pretr_ConvNeXt_v3", "Pretr_Plus_v3", "plus_v3_s1", "Plus_viT_v3", "ConvNeXt_v3"]
LABEL = {
    "Pretr_ConvNeXt_v3": "ConvNeXt pretrained (probe)",
    "Pretr_Plus_v3": "ViT-S/16+ pretrained (probe)",
    "plus_v3_s1": "ViT-Plus finetune A1",
    "Plus_viT_v3": "ViT-Plus finetune A0",
    "ConvNeXt_v3": "ConvNeXt finetuned",
}
COL = {
    "Pretr_ConvNeXt_v3": "#c47ba0",
    "Pretr_Plus_v3": "#b0884f",
    "plus_v3_s1": "#3f6fb5",
    "Plus_viT_v3": "#9db6d8",
    "ConvNeXt_v3": "#26a269",
}
ORDER = ["Pretr_ConvNeXt_v3", "Pretr_Plus_v3", "ConvNeXt_v3", "Plus_viT_v3", "plus_v3_s1"]

# nhóm phương thức (theo 4 kiểu deepfake session-1, áp cho 38 method có trong test)
FAM = {
    "Face Swap": ["faceswap", "deepfake_faceswap", "simswap", "inswap", "mobileswap",
                  "facedancer", "blendface", "uniface", "deepfacelab"],
    "Reenactment": ["sadtalker", "wav2lip", "fomm", "MRAA", "lia", "mcnet", "tpsm",
                    "facevid2vid", "hyperreenact", "pirender", "one_shot_free",
                    "danet", "fsgan", "heygen"],
    "Face Synthesis": ["DiT", "SiT", "StyleGAN2", "StyleGAN3", "StyleGANXL", "sd2.1",
                       "MidJourney", "pixart", "RDDM", "ddim", "VQGAN"],
    "Face Editing": ["stargan", "styleclip", "e4e", "e4s"],
}
WEAK8 = {"faceswap", "deepfake_faceswap", "wav2lip", "sadtalker", "fsgan",
         "facedancer", "inswap", "mobileswap"}


def load(tag):
    p = os.path.join(RES, f"{tag}_preds.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return dict(preds=z["preds"], probs=z["probs"], labels=z["labels"],
                methods=z["methods"], sources=z["sources"])


def metrics(d):
    y, pr = d["labels"], d["preds"]
    acc = float((pr == y).mean())
    real_acc = float((pr[y == 0] == 0).mean())
    fake_rec = float((pr[y == 1] == 1).mean())
    auc = None
    if "probs" in d and d["probs"].size:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, d["probs"]))
    return dict(acc=acc, real_acc=real_acc, fake_rec=fake_rec, auc=auc)


def det_rate(d, mask):
    return float((d["preds"][mask] == 1).mean())


if "--tags" in sys.argv:
    sel = sys.argv[sys.argv.index("--tags") + 1:]
    TAGS = sel
DATA = {t: load(t) for t in TAGS}
DATA = {t: d for t, d in DATA.items() if d is not None}
ORDER = [t for t in ORDER if t in DATA]
print("Loaded:", list(DATA.keys()))
print("Plot order:", ORDER)

# ---------- fig09: tổng thể 5 model ----------
fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(len(ORDER)); w = 0.22
vals = {k: [metrics(DATA[t])[k] for t in ORDER] for k in ["acc", "real_acc", "fake_rec"]}
for j, (k, lab, c) in enumerate([("acc", "accuracy", "#444444"),
                                 ("real_acc", "real acc", "#1f6f3b"),
                                 ("fake_rec", "fake recall", "#8a2f2f")]):
    ax.bar(x + (j - 1) * w, vals[k], width=w, label=lab, color=c)
ax.set_xticks(x)
ax.set_xticklabels([f"{t}\n{LABEL[t]}" for t in ORDER], fontsize=8)
ax.set_ylim(0, 1.02)
ax.legend(ncol=3, fontsize=9, loc="lower right")
ax.set_title("Test cân bằng 21,446 — pretrained (frozen+probe) vs finetune")
ax.grid(axis="y", alpha=0.3)
for t in ORDER:
    m = metrics(DATA[t])
    ax.text(x[ORDER.index(t)] - w, 0.02, f"{m['acc']*100:.1f}%", fontsize=8,
            va="bottom", ha="center", color="#111")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig09_models_overall.png"), dpi=120); plt.close(fig)

# ---------- fig10: per-method det (2 pretrained) ----------
if {"Pretr_ConvNeXt_v3", "Pretr_Plus_v3"} <= set(DATA):
    pr_methods = np.array(DATA["Pretr_ConvNeXt_v3"]["methods"])
    y_all = DATA["Pretr_ConvNeXt_v3"]["labels"]
    fake_methods = sorted({m for m in pr_methods if m != "real"})
    rows = []
    for m in fake_methods:
        mask = (pr_methods == m) & (y_all == 1)
        rows.append((m, det_rate(DATA["Pretr_Plus_v3"], mask), det_rate(DATA["Pretr_ConvNeXt_v3"], mask)))
    rows.sort(key=lambda r: -(r[1] + r[2]) / 2)  # giảm dần theo điểm TB 2 pretrained
    names = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 12))
    yy = np.arange(len(names)); hh = 0.38
    v1 = [r[1] for r in rows]; v2 = [r[2] for r in rows]
    ax.barh(yy + hh / 2, v1, height=hh, color=COL["Pretr_Plus_v3"], label="ViT pretrained (probe)")
    ax.barh(yy - hh / 2, v2, height=hh, color=COL["Pretr_ConvNeXt_v3"], label="ConvNeXt pretrained (probe)")
    if "ConvNeXt_v3" in DATA:  # marker finetuned để thấy gap
        fn = [det_rate(DATA["ConvNeXt_v3"], (pr_methods == m) & (y_all == 1)) for m in names]
        ax.plot(fn, yy, "o", ms=4, color="#26a269", label="ConvNeXt finetuned", alpha=0.9)
    for yi, m in enumerate(names):
        if m in WEAK8:
            ax.text(1.01, yi + hh / 2, "W", va="center", fontsize=8,
                    color="#b8591a", fontweight="bold")
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 1.05); ax.set_xlabel("detection rate trên 300 ảnh/method (fake)")
    ax.set_title("Det-rate theo từng phương thức — pretrained vs finetuned (W = 8 method yếu A1)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_pretr_permethod.png"), dpi=120); plt.close(fig)
else:
    print("fig10 skip (cần 2 tag pretrained)")

# ---------- fig11: theo 4 nhóm deepfake ----------
fam_order = ["Face Swap", "Reenactment", "Face Synthesis", "Face Editing"]
print("\nDet-rate theo nhóm (ảnh-weighted):")
hdr = f"{'nhóm':<14}" + "".join(f"{t:>20}" for t in ORDER)
print(hdr)
fdata = {}
for fam in fam_order:
    mem = set(FAM[fam])
    mask = None
    for tag in DATA:
        mm = np.array(DATA[tag]["methods"]); y = DATA[tag]["labels"]
        m_ = (np.isin(mm, list(mem))) & (y == 1)
        if mask is None:
            mask = m_
    fdata[fam] = [det_rate(DATA[t], (np.isin(np.array(DATA[t]['methods']), list(mem))) & (DATA[t]['labels'] == 1))
                  for t in ORDER]
    print(f"{fam:<14}" + "".join(f"{v:20.3f}" for v in fdata[fam]))
# weak8 vs other
w8 = [det_rate(DATA[t], (np.isin(np.array(DATA[t]['methods']), list(WEAK8))) & (DATA[t]['labels'] == 1)) for t in ORDER]
other = [det_rate(DATA[t], (~np.isin(np.array(DATA[t]['methods']), list(WEAK8))) & (np.array(DATA[t]['methods']) != 'real') & (DATA[t]['labels'] == 1)) for t in ORDER]
print(f"{'8-method yếu':<14}" + "".join(f"{v:20.3f}" for v in w8))
print(f"{'34 method còn lại':<14}" + "".join(f"{v:20.3f}" for v in other))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
x = np.arange(len(ORDER)); w = 0.16
ax = axes[0]
for i, fam in enumerate(fam_order):
    ax.barh(x + (i - 1.5) * w, fdata[fam], height=w, label=fam)
ax.set_yticks(x); ax.set_yticklabels([t.split("_")[0] for t in ORDER], fontsize=8)
ax.set_xlim(0, 1.05); ax.set_xlabel("det-rate (fake, ảnh-weighted)")
ax.set_title("A. Theo 4 nhóm kiểu deepfake"); ax.legend(fontsize=8, ncol=2)
ax.grid(axis="x", alpha=0.3)
ax2 = axes[1]
for i, (vals, nm) in enumerate([(w8, "8 method yếu (A1 target)"), (other, "34 method còn lại")]):
    ax2.bar(x + (i - 0.5) * w, vals, width=w, label=nm)
ax2.set_xticks(x); ax2.set_xticklabels([t.split("_")[0] for t in ORDER], rotation=15, fontsize=8)
ax2.set_ylim(0, 1.05); ax2.set_ylabel("det-rate"); ax2.legend(fontsize=8)
ax2.set_title("B. Nhóm yếu vs còn lại"); ax2.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig11_pretr_family.png"), dpi=120); plt.close(fig)

# ---------- fig12: real theo source ----------
_ref = ORDER[0]
r_real = np.array(DATA[_ref]["sources"])[np.array(DATA[_ref]["labels"]) == 0]
rsrcs = sorted(set(r_real))
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(rsrcs)); w = 0.15
for j, t in enumerate(ORDER):
    mm = np.array(DATA[t]["methods"]); y = DATA[t]["labels"]
    srcs = np.array(DATA[t]["sources"])
    rr = [(srcs == s) & (y == 0) for s in rsrcs]
    ax.bar(x + (j - (len(ORDER) - 1) / 2) * w, [det_rate(DATA[t], rr_) for rr_ in rr],
           width=w, label=t.split("_")[0], color=COL[t])
ax.set_xticks(x); ax.set_xticklabels(rsrcs, fontsize=9)
ax.set_ylim(0, 1.02); ax.set_ylabel("real_acc theo source"); ax.legend(fontsize=7, ncol=5)
ax.set_title("Real theo source (giữ được real) — 1,728 ff++_real / 8,787 ffhq / 208 test_data_v3")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig12_pretr_realsource.png"), dpi=120); plt.close(fig)

print("\nFigures →", FIG)

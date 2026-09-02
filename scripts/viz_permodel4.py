#!/usr/bin/env python3
"""fig13 — det-rate per-method cho 4 model (2 pretrained probe + 2 finetune tốt nhất)
+ 1 panel so sánh cả 4, trên test cân bằng 21,446.

Thay lối nhìn của fig10 (2 pretrained + marker finetuned) bằng:
  hàng 1: ViT pretrained (probe) · ConvNeXt pretrained (probe)
  hàng 2: ViT-Plus finetune A1   · ConvNeXt finetuned
  hàng dưới (full-width): so sánh dot-plot cả 4, cùng thứ tự method.
Cùng thứ tự method ở mọi panel (mean det-rate 4 model, mạnh nhất trên cùng).
Ghi agents/figures/session2/fig13_permodel_detrate.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results", "coursework_vs")
FIG = os.path.join(ROOT, "agents", "figures", "session2")
os.makedirs(FIG, exist_ok=True)

# 2 pretrained (probe) + 2 finetune tốt nhất (A1 = plus_v3_s1, ConvNeXt_v3)
TAGS = ["Pretr_Plus_v3", "Pretr_ConvNeXt_v3", "plus_v3_s1", "ConvNeXt_v3"]
GRID = [(0, 0, "Pretr_Plus_v3"), (0, 1, "Pretr_ConvNeXt_v3"),
        (1, 0, "plus_v3_s1"), (1, 1, "ConvNeXt_v3")]
LABEL = {
    "Pretr_Plus_v3": "ViT-S/16+ pretrained (probe)",
    "Pretr_ConvNeXt_v3": "ConvNeXt pretrained (probe)",
    "plus_v3_s1": "ViT-Plus finetune A1",
    "ConvNeXt_v3": "ConvNeXt finetuned",
}
COL = {
    "Pretr_Plus_v3": "#b0884f",
    "Pretr_ConvNeXt_v3": "#c47ba0",
    "plus_v3_s1": "#3f6fb5",
    "ConvNeXt_v3": "#26a269",
}
WEAK8 = {"faceswap", "deepfake_faceswap", "wav2lip", "sadtalker", "fsgan",
         "facedancer", "inswap", "mobileswap"}


def load(tag):
    p = os.path.join(RES, f"{tag}_preds.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return dict(preds=z["preds"], labels=z["labels"], methods=z["methods"])


def det_rate(d, mask):
    return float((d["preds"][mask] == 1).mean())


def metrics(d):
    y, pr = d["labels"], d["preds"]
    return dict(acc=(pr == y).mean(), real=(pr[y == 0] == 0).mean(),
                fake=(pr[y == 1] == 1).mean())


DATA = {t: load(t) for t in TAGS}
DATA = {t: d for t, d in DATA.items() if d is not None}
assert len(DATA) == len(TAGS), f"thiếu preds: {[t for t in TAGS if t not in DATA]}"

ref = DATA[TAGS[0]]
methods = np.array(ref["methods"])
y_all = ref["labels"]
fake_methods = sorted({m for m in methods if m != "real"})

# per-method det-rate cho từng model; thứ tự giảm dần theo điểm TB 4 model
rows = []
for m in fake_methods:
    mask = (methods == m) & (y_all == 1)
    rows.append((m, {t: det_rate(DATA[t], mask) for t in TAGS}))
rows.sort(key=lambda r: -np.mean(list(r[1].values())))
names = [r[0] for r in rows]
means = {t: np.mean([r[1][t] for r in rows]) for t in TAGS}

fig = plt.figure(figsize=(17, 24))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.05, 0.9],
              hspace=0.14, wspace=0.10)

for r_, c_, tag in GRID:
    ax = fig.add_subplot(gs[r_, c_])
    vals = [rows[i][1][tag] for i in range(len(names))]
    yy = np.arange(len(names))
    ax.barh(yy, vals, height=0.72, color=COL[tag], alpha=0.92)
    # tô đậm + ghi * cho 8 method yếu (mục tiêu A1)
    tick_labels = [f"{m}*" if m in WEAK8 else m for m in names]
    ax.set_yticks(yy)
    tls = ax.set_yticklabels(tick_labels, fontsize=7.5)
    for tl, m in zip(tls, names):
        if m in WEAK8:
            tl.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("detection rate (fake, 300 ảnh/method)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    m_ = metrics(DATA[tag])
    ax.set_title(f"{LABEL[tag]}   ·   acc {m_['acc']*100:.1f}%   ·   mean det {means[tag]*100:.1f}%",
                 fontsize=10.5, loc="left")
fig.suptitle("* = 8 method yếu (mục tiêu sampler A1) — cùng thứ tự method ở mọi ô (mạnh nhất trên cùng)",
             fontsize=10, y=0.997)

# ---- panel so sánh cả 4 (dot plot, full-width) ----
ax = fig.add_subplot(gs[2, :])
for j, t in enumerate(TAGS):
    off = (j - 1.5) * 0.16
    yy = np.arange(len(names)) + off
    xs = [rows[i][1][t] for i in range(len(names))]
    ax.plot(xs, yy, "o", ms=5.5, color=COL[t], label=LABEL[t])
    # đường đứt theo mean det chung của model (toàn fake)
    ax.axvline(means[t], color=COL[t], ls="--", lw=0.8, alpha=0.55)
    ax.text(means[t], len(names) + 0.7, f"{means[t]*100:.0f}%", color=COL[t],
            ha="center", fontsize=8)
ax.set_yticks(np.arange(len(names)))
ax.set_yticklabels([f"{m}*" if m in WEAK8 else m for m in names], fontsize=7.5)
for tl, m in zip(ax.get_yticklabels(), names):
    if m in WEAK8:
        tl.set_fontweight("bold")
ax.invert_yaxis()
ax.set_xlim(0, 1.03)
ax.set_xlabel("detection rate (fake)")
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)
ax.set_title("So sánh cả 4 — điểm mỗi method; nét đứt = mean det của từng model", fontsize=10.5, loc="left")
ax.legend(ncol=4, fontsize=9, loc="lower right")

fig.subplots_adjust(top=0.985, bottom=0.008, left=0.10, right=0.995,
                    hspace=0.18, wspace=0.10)
out = os.path.join(FIG, "fig13_permodel_detrate.png")
fig.savefig(out, dpi=120)
plt.close(fig)
print("saved", out)

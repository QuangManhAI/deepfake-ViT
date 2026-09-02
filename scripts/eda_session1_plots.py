#!/usr/bin/env python3
"""Session 1 — DF40 data EDA figures.

Reads agents/figures/session1/summary.json + hist_data.npz (produced by
eda_session1_data.py) and renders all figures for the session-1 report.

Palette: validated dataviz categorical slots (blue/orange/aqua/yellow/magenta/green).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "agents", "figures", "session1")
os.makedirs(FIG, exist_ok=True)

# ---- validated palette (light-surface slots, dataviz) ----
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
RED = "#e34948"
INK, MUTED, GRIDL = "#0b0b0b", "#52514e", "#e1e0d9"
FAMILY_COLOR = {"Face Swap": BLUE, "Reenactment": ORANGE, "Face Synthesis": AQUA, "Face Editing": YELLOW}
REAL_C, FAKE_C = GREEN, RED

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRIDL, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 10, "axes.grid": True, "grid.color": GRIDL, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 150, "savefig.bbox": "tight",
})

S = json.load(open(os.path.join(FIG, "summary.json")))
H = np.load(os.path.join(FIG, "hist_data.npz"))
methods_summary = json.load(open(os.path.join(ROOT, "data/splits/methods_summary.json")))
eval_json = json.load(open(os.path.join(ROOT, "outputs/eval/identity_disjoint_v3_vit.json")))
FAM_ORDER = ["Face Swap", "Reenactment", "Face Synthesis", "Face Editing"]
fam_color = lambda m: FAMILY_COLOR[S["method_to_family"][m]]

def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print(f"  wrote {name}")

# =====================================================================
# Fig 01 — Real vs Fake, toàn bộ + từng split
# =====================================================================
splits = S["split_info_identity_disjoint"]
groups = [("Toàn bộ", S["test_data_v3_full"]["real"], S["test_data_v3_full"]["fake"])]
for k in ("train", "val", "test"):
    v = splits[k]
    groups.append((k.capitalize(), v["real"], v["fake"]))
fig, ax = plt.subplots(figsize=(7.6, 4.2))
y = np.arange(len(groups))[::-1]
h = 0.34
ax.barh(y + h / 2, [g[1] for g in groups], height=h, color=REAL_C, label="Real")
ax.barh(y - h / 2, [g[2] for g in groups], height=h, color=FAKE_C, label="Fake")
for yy, (name, nr, nf) in zip(y, groups):
    ax.text(nr, yy + h / 2, f"{nr:,}", va="center", ha="right", fontsize=9, color=INK)
    ax.text(nf, yy - h / 2, f"{nf:,}", va="center", ha="left", fontsize=9, color=INK)
ax.set_yticks(y); ax.set_yticklabels([g[0] for g in groups])
ax.set_xlim(0, 33000)
ax.set_xlabel("Số ảnh")
ax.set_title("test_data_v3 — real vs fake (toàn bộ và từng split)")
ax.legend(loc="lower right", frameon=False)
save(fig, "fig01_real_fake.png")

# =====================================================================
# Fig 02 — Nguồn ảnh real (FF++ vs Celeb-DF v2)
# =====================================================================
# Memmap lưu chung nhãn 'real' (1.177 ảnh). Nguồn chính xác nhất hiện có là
# eval identity-disjoint, tách real theo domain: real/ffc (FF++) vs real/cdc
# (Celeb-DF v2) trên bộ 9.232 ảnh giữ lại.
src = {"FF++ (ffc)": eval_json["per_method_domain"].get("real/ffc", {}).get("n", 0),
       "Celeb-DF v2 (cdc)": eval_json["per_method_domain"].get("real/cdc", {}).get("n", 0)}
src = dict(sorted(src.items(), key=lambda kv: -kv[1]))
fig, ax = plt.subplots(figsize=(6.2, 3.6))
names, vals = list(src.keys()), list(src.values())
tot = sum(vals)
bars = ax.bar(names, vals, color=[GREEN, GREEN], width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,} ({v/tot*100:.0f}%)",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Số ảnh real (bộ eval 9.232)")
ax.set_ylim(0, max(vals) * 1.18)
ax.set_title("Nguồn ảnh real trong bộ eval identity-disjoint")
save(fig, "fig02_real_sources.png")

# =====================================================================
# Fig 03 — Nhóm method: số lượng method + số ảnh fake theo family
# =====================================================================
n_methods = [S["family_method_count"][f] for f in FAM_ORDER]
n_fake = [S["family_fake_full"][f] for f in FAM_ORDER]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
ax = axes[0]
bars = ax.bar(FAM_ORDER, n_methods, color=[FAMILY_COLOR[f] for f in FAM_ORDER], width=0.55)
for b, v in zip(bars, n_methods):
    ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, 16); ax.set_ylabel("Số method"); ax.set_title("(a) Số method mỗi nhóm")
for lbl in ax.get_xticklabels():
    lbl.set_rotation(18); lbl.set_ha("right")
ax = axes[1]
bars = ax.bar(FAM_ORDER, n_fake, color=[FAMILY_COLOR[f] for f in FAM_ORDER], width=0.55)
for b, v in zip(bars, n_fake):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, 12000); ax.set_ylabel("Số ảnh fake (test_data_v3)")
ax.set_title("(b) Số ảnh fake mỗi nhóm")
for lbl in ax.get_xticklabels():
    lbl.set_rotation(18); lbl.set_ha("right")
fig.suptitle("Nhóm 40 method DF40 theo kiểu deepfake", y=1.02, fontsize=12)
save(fig, "fig03_families.png")

# =====================================================================
# Fig 04 — 40 method × số ảnh fake (test_data_v3), tô màu theo family
# =====================================================================
pmf = S["per_method_full"]
items = sorted(pmf.items(), key=lambda kv: -kv[1])
fig, ax = plt.subplots(figsize=(8.6, 9.2))
names = [m for m, _ in items]; vals = [v for _, v in items]
colors = [fam_color(m) for m in names]
y = np.arange(len(names))[::-1]
ax.barh(y, vals, color=colors, height=0.72)
for yy, v, m in zip(y, vals, names):
    ax.text(v, yy, f"{v:,}", va="center", ha="left", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlim(0, max(vals) * 1.15)
ax.set_xlabel("Số ảnh fake (test_data_v3, tổng 29.514)")
ax.set_title("40 method — số ảnh fake, tô màu theo 4 nhóm")
handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in FAM_ORDER]
ax.legend(handles, FAM_ORDER, loc="lower right", frameon=False)
save(fig, "fig04_methods_fake.png")

# =====================================================================
# Fig 05 — Kích thước ảnh
# =====================================================================
size_counts = S["size_counts"]
labels, vals = zip(*sorted(size_counts.items(), key=lambda kv: int(kv[0].split("_")[0])))
fig, ax = plt.subplots(figsize=(6.2, 3.6))
bars = ax.bar([f"{a}×{b}" for a, b in [l.split("_") for l in labels]], vals,
              color=[AQUA, BLUE, ORANGE], width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Số ảnh (mẫu khảo sát 2.400 ảnh)")
ax.set_ylim(0, max(vals) * 1.15)
ax.set_title("Phân bố kích thước ảnh")
save(fig, "fig05_size.png")

# =====================================================================
# Fig 06 & 07 — Hist brightness / sharpness theo nhóm pixel
# =====================================================================
pix_groups = ["real/CollabDiff", "real/deepfacelab", "fake/CollabDiff", "fake/deepfacelab",
              "fake/facedancer", "fake/faceswap", "fake/DiT", "fake/ddim"]
short = {"real/CollabDiff": "Real — CollabDiff", "real/deepfacelab": "Real — DeepFaceLab",
         "fake/CollabDiff": "Fake — CollabDiff", "fake/deepfacelab": "Fake — DeepFaceLab",
         "fake/facedancer": "Fake — FaceDancer", "fake/faceswap": "Fake — FaceSwap",
         "fake/DiT": "Fake — DiT", "fake/ddim": "Fake — DDIM"}
for arrkey, title, xlab, col, fname in [
    ("", "Độ sáng trung bình (0–255)", "độ sáng", AQUA, "fig06_brightness_hist.png"),
    ("__sharp", "Độ sắc nét (Laplacian variance)", "độ sắc nét", BLUE, "fig07_sharpness_hist.png"),
]:
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    for ax, g in zip(axes.flat, pix_groups):
        data = H[g + arrkey]
        ax.hist(data, bins=24, color=col, alpha=0.85, edgecolor="white", linewidth=0.3)
        if arrkey == "__sharp":
            # cap heavy tail for legibility
            data = np.clip(data, 0, np.percentile(data, 97))
        ax.set_title(short[g], fontsize=9)
        ax.set_xlabel(xlab, fontsize=8); ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout()
    save(fig, fname)

# =====================================================================
# Fig 08 — Vấn đề trùng identity (frames vs identities riêng biệt)
# =====================================================================
ids = S["identity_stats_local"]
methods = list(ids.keys())
frames = [ids[m]["n_frames"] for m in methods]
ident = [ids[m]["n_identity_dirs"] for m in methods]
fpi = [ids[m]["frames_per_identity_mean"] for m in methods]
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(methods)); w = 0.34
ax.bar(x - w / 2, frames, w, color=BLUE, label="Tổng frame")
ax.bar(x + w / 2, ident, w, color=ORANGE, label="Identity riêng biệt (folder)")
for i, v in enumerate(frames):
    ax.text(x[i] - w / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5)
for i, v in enumerate(ident):
    ax.text(x[i] + w / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5)
for i, fp in enumerate(fpi):
    ax.annotate(f"~{fp} frame/identity", (x[i], ident[i] * 0.35),
                ha="center", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRIDL, lw=0.8))
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel("Số lượng (log)")
ax.set_title("Dữ liệu dạng frames-by-frames: nhiều frame nhưng identity lặp (~31 frame/người)")
ax.legend(frameon=False)
save(fig, "fig08_identity.png")

# =====================================================================
# Fig 09 — Gallery fake (6 method × 3 ảnh)
# =====================================================================
gal = S["gallery"]["fake"]
order = ["facedancer", "faceswap", "DiT", "ddim", "CollabDiff", "deepfacelab"]
from PIL import Image
fig, axes = plt.subplots(3, 6, figsize=(13, 6.6))
for j, m in enumerate(order):
    for i, p in enumerate(gal[m][:3]):
        ax = axes[i][j]
        im = np.asarray(Image.open(p).convert("RGB").resize((128, 128)))
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(GRIDL)
        if i == 0:
            ax.set_title(m, fontsize=9)
fig.suptitle("Gallery ảnh fake — 6 method đại diện (test_data_v3)", y=1.0, fontsize=12)
fig.tight_layout()
save(fig, "fig09_gallery_fake.png")

# =====================================================================
# Fig 10 — Gallery real vs fake (cặp CollabDiff + DeepFaceLab)
# =====================================================================
rows = [
    ("CollabDiff", S["gallery"]["real_CollabDiff"], S["gallery"]["fake_CollabDiff_pairs"]),
    ("DeepFaceLab", S["gallery"]["real_deepfacelab"], S["gallery"]["fake_deepfacelab_pairs"]),
]
fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
for r, (mname, reals, fakes) in enumerate(rows):
    for i in range(4):
        ax = axes[r][i]
        if i % 2 == 0:
            im = np.asarray(Image.open(reals[i // 2]).convert("RGB").resize((128, 128)))
            color, label = GREEN, "real"
        else:
            im = np.asarray(Image.open(fakes[i // 2]).convert("RGB").resize((128, 128)))
            color, label = RED, "fake"
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2.2)
        if i < 2:
            ax.set_title(label + ("  (" + mname + ")" if i == 0 else ""), fontsize=9, color=color)
fig.suptitle("Cặp real vs fake — cùng method (viền xanh = real, đỏ = fake)", y=1.0, fontsize=12)
fig.tight_layout()
save(fig, "fig10_gallery_pairs.png")

# =====================================================================
# Fig 11 — Chia train/val/test: real/fake + số identity
# =====================================================================
sp = S["split_info_identity_disjoint"]
names = ["train", "val", "test"]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(3); w = 0.34
ax.bar(x - w / 2, [sp[k]["real"] for k in names], w, color=REAL_C, label="Real")
ax.bar(x + w / 2, [sp[k]["fake"] for k in names], w, color=FAKE_C, label="Fake")
for i, k in enumerate(names):
    ax.text(x[i] - w / 2, sp[k]["real"], f"{sp[k]['real']:,}", ha="center", va="bottom", fontsize=9)
    ax.text(x[i] + w / 2, sp[k]["fake"], f"{sp[k]['fake']:,}", ha="center", va="bottom", fontsize=9)
    ax.text(x[i], sp[k]["fake"] * 0.5, f"{sp[k]['identities']:,}\nidentity", ha="center", va="center",
            fontsize=8.5, color="white", fontweight="bold")
    ax.text(x[i], sp[k]["fake"] * 1.08, f"tỷ lệ real:fake ≈ {sp[k]['ratio']}",
            ha="center", va="bottom", fontsize=8.5, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels([k.upper() for k in names])
ax.set_ylim(0, 24500); ax.set_ylabel("Số ảnh")
ax.set_title("Chia identity-disjoint: train / val / test")
ax.legend(loc="upper right", frameon=False)
save(fig, "fig11_split.png")

# =====================================================================
# Fig 12 — Kết quả eval theo domain (identity-disjoint split)
# =====================================================================
dom = eval_json["per_domain"]
dnames = ["cdc", "efs", "fe", "ffc", "oth"]
dlabel = {"cdc": "cdc (Celeb-DF)", "efs": "efs (tổng hợp)", "fe": "fe (chỉnh sửa)",
          "ffc": "ffc (FF++)", "oth": "oth (khác)"}
n = [dom[d]["n"] for d in dnames]; acc = [dom[d]["acc"] for d in dnames]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
bars = ax.bar([dlabel[d] for d in dnames], n, color=[BLUE, AQUA, MAGENTA, ORANGE, YELLOW], width=0.55)
for b, v in zip(bars, n):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Số ảnh (eval)"); ax.set_title("(a) Số mẫu mỗi domain")
for lbl in ax.get_xticklabels():
    lbl.set_rotation(16); lbl.set_ha("right")
ax = axes[1]
bars = ax.bar([dlabel[d] for d in dnames], acc, color=[BLUE, AQUA, MAGENTA, ORANGE, YELLOW], width=0.55)
ax.axhline(0.9519, color=INK, lw=1.2, ls="--")
ax.text(3.6, 0.955, "acc trung bình 95.2%", fontsize=8, color=INK)
for b, v in zip(bars, acc):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0.5, 1.04); ax.set_ylabel("Accuracy"); ax.set_title("(b) Accuracy mỗi domain")
for lbl in ax.get_xticklabels():
    lbl.set_rotation(16); lbl.set_ha("right")
fig.suptitle("Eval identity-disjoint (ViT-S/16+ pretrain) — theo domain", y=1.02, fontsize=12)
save(fig, "fig12_domain_eval.png")

# =====================================================================
# Fig 13 — Detection rate theo method (test_data_v3), highlight method yếu
# =====================================================================
per_meth = eval_json["per_method"]
# method yếu: detection thấp trên ffc (FF++ source)
weak = sorted([(k, v["detection_rate"]) for k, v in per_meth.items() if k != "real/ffc"],
              key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(8.6, 9.6))
names = [k for k, _ in weak]; rates = [v for _, v in weak]
y = np.arange(len(names))[::-1]
colors = [fam_color(m) for m in names]
ax.barh(y, rates, color=colors, height=0.72)
for yy, v in zip(y, rates):
    ax.text(v, yy, f"{v:.1%}", va="center", ha="left", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlim(0, 1.08); ax.set_xlabel("Detection rate (mẫu test_data_v3)")
ax.set_title("Detection rate theo method — 40 method, tô màu theo nhóm")
handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in FAM_ORDER]
ax.legend(handles, FAM_ORDER, loc="lower right", frameon=False)
save(fig, "fig13_method_detection.png")

print("\nDone. Figures in", FIG)

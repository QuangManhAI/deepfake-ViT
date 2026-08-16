"""Ghép các attention-map PNG đã có thành một figure 2x4 cho report (không load model).

Dùng ảnh có sẵn trong outputs/attention/ (real_0 / fake_0, các layer 5/8/11).
Output: report/figures/attention_grid.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ATTN = os.path.join(ROOT, "outputs", "attention")
FIG_DIR = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LAYERS = [5, 8, 11]
COLS = ["Original"] + [f"Layer {L}" for L in LAYERS]
ROWS = ["Real", "Fake"]

fig, axes = plt.subplots(2, 4, figsize=(9.5, 4.6))
for r, prefix in enumerate(["real", "fake"]):
    name = "0"
    files = [f"{prefix}_{name}_original.png"] + [f"{prefix}_{name}_layer{L}.png" for L in LAYERS]
    for c, fname in enumerate(files):
        p = os.path.join(ATTN, fname)
        img = mpimg.imread(p)
        ax = axes[r, c]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(COLS[c], fontsize=11)
        if c == 0:
            ax.set_ylabel(ROWS[r], fontsize=11, rotation=0, labelpad=28, va="center")

fig.suptitle("DINOv3 ViT attention maps (CLS token, averaged over heads)", fontsize=12, y=1.02)
fig.tight_layout()
out = os.path.join(FIG_DIR, "attention_grid.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("Wrote", out)

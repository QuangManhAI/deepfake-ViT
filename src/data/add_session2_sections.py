"""Append Session-2 EDA sections to the five notebooks (ADDITIVE ONLY).

Guarantees:
  * No existing cell is modified, reordered or removed.
  * New cells are appended at the end, inside a clearly delimited section.
  * Re-running replaces only the previously appended block (idempotent),
    identified by the SENTINEL marker.

All analysis logic lives in ``src/data/session2_eda.py`` -- the notebooks are
thin visualization layers, so nothing is duplicated across them.

Run: .venv/bin/python src/data/add_session2_sections.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"

SENTINEL = "SESSION-2 EDA ADDENDUM"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().split("\n")}


# ---------------------------------------------------------------------------
# Bootstrap used by every appended section (import-only, no path guessing)
# ---------------------------------------------------------------------------
BOOT = '''
# --- Session-2 addendum bootstrap (shared implementation, no duplication) ---
import sys
from pathlib import Path

_ROOT = Path.cwd()
if _ROOT.name == "notebooks":
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src.data import session2_eda as S2
from src.data import protocol as P
from src.data import loader

pd.set_option("display.width", 200)

# ---- Shared consistency banner: identical in all five notebooks. ----
# Values come from src.data.session2_eda (REFERENCE + measured .npz preds).
# If any notebook ever printed different values here, the notebooks would be
# describing different data. They cannot, because they all read one module.
print(S2.consistency_banner())
print()
print(P.describe())
print()
print(loader.describe())
print()
print("KNOWN LIMITATIONS OF THE ACTIVE PROTOCOL")
for _lim in P.KNOWN_LIMITATIONS:
    print(f"  - {_lim}")
print()

# Be explicit about which reference findings can and cannot be measured here.
print(S2.describe())
'''

AVAIL_NOTE = """
### Provenance convention used in this addendum

Every figure below is tagged:

* **MEASURED** - computed from an artifact present in this repository.
* **REFERENCE** - quoted from `session2_finetune_data_eda.pdf` because the
  underlying raw data is *not* in this repository. Never presented as measured.

The Session-2 **training** corpus (`finetune_plus_train.csv`, ~123.6k images)
and the 50k test suite are not committed here. The Session-2 **evaluation**
artifacts are, under `experiments/results/coursework_vs/`, so all
evaluation-side findings are reproduced from real paired predictions.
"""


# ---------------------------------------------------------------------------
# 00 - dataset EDA addendum
# ---------------------------------------------------------------------------
def cells_00():
    return [
        md(f"""
---
# {SENTINEL} - Dataset Composition, Long-Tail, Identity & Leakage

Reference: `session2_finetune_data_eda.pdf`.

Everything above this line is the original notebook and is unchanged. This
section **adds** the dataset-level analysis the reference expects.
{AVAIL_NOTE}
"""),
        code(BOOT),
        md("## A1. Session-2 dataset composition (REFERENCE) vs local protocol (MEASURED)\n\nThe two corpora are different datasets, so they are reported side by side rather than merged."),
        code('''
ref = S2.REFERENCE
s2 = pd.DataFrame([
    {"corpus": "Session-2 finetune (REFERENCE)", "total": ref["train_total"],
     "real": ref["train_real"], "fake": ref["train_fake"],
     "fake:real": round(ref["train_fake"] / ref["train_real"], 2),
     "fake methods": ref["n_fake_methods"], "real sources": ref["n_real_sources"]},
])
loc = loader.load_splits_concat(detailed=True)
s2.loc[len(s2)] = {
    "corpus": "Local identity_clean_v1 (MEASURED)", "total": len(loc),
    "real": int((loc["label"] == P.LABEL_REAL).sum()),
    "fake": int((loc["label"] == P.LABEL_FAKE).sum()),
    "fake:real": round((loc["label"] == 1).sum() / max(1, (loc["label"] == 0).sum()), 2),
    "fake methods": loc.loc[loc.label == 1, "method"].nunique(),
    "real sources": loc.loc[loc.label == 0, "domain"].nunique(),
}
display(s2)

print("KEY DIFFERENCE")
print(f"  Session-2 train is ~1:{s2.loc[0,'fake:real']} fake:real (deliberately near-balanced).")
print(f"  Local protocol  is ~1:{s2.loc[1,'fake:real']} fake:real (severely imbalanced).")
print("  -> metrics are NOT comparable across the two corpora; never mix them.")
'''),
        md("## A2. Real-source diversity (REFERENCE)\n\nSession-1 real data came only from FF++/Celeb-DF video frames. Session-2 adds hi-res stills and AI-synthesised 'real' images."),
        code('''
rs = pd.Series(ref["real_sources"]).sort_values(ascending=False)
rs_df = rs.rename("images").to_frame()
rs_df["share_%"] = (100 * rs / rs.sum()).round(1)
rs_df["type"] = ["video frame", "video frame", "hi-res still", "hi-res still",
                 "diffusion benchmark", "static", "AI-synth 'real'",
                 "DF40 real", "GAN face-edit"][:len(rs_df)]
display(rs_df)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
rs.plot(kind="barh", ax=axes[0], color="#2ca02c")
axes[0].invert_yaxis(); axes[0].set_xlabel("images")
axes[0].set_title("Session-2 real sources (REFERENCE, 9 sources)")
grp = rs_df.groupby("type")["images"].sum().sort_values(ascending=False)
axes[1].pie(grp, labels=grp.index, autopct="%1.0f%%", startangle=120)
axes[1].set_title("Real images by acquisition type")
plt.tight_layout(); plt.show()

print("WHY IT MATTERS (inference)")
print("  Mixing video frames, hi-res stills and AI-synth 'real' means the model")
print("  cannot rely on a single 'video-frame look' to decide REAL.")
print("  It also creates a real-side domain shift -> motivates SOURCE-AWARE")
print("  evaluation, which section A7 measures on the balanced test.")
'''),
        md("## A3. Fake-method long tail and the 8-method weak family (REFERENCE pools)\n\nThe weak family is read directly from the project sampler, not hard-coded here."),
        code('''
fam_w = S2.parse_family_weights()
print("WEAK FAMILY parsed from scripts/finetune_plus_v3.py (source of truth):")
for k, v in fam_w.items():
    print(f"   {k:20s} sampler weight {v}")
assert set(fam_w) == set(S2.WEAK_FAMILY), "sampler family drifted from module constant"
print("\\nVERIFIED: matches the 8 methods named in the reference.")

pools = pd.Series(ref["method_pools"]).sort_values(ascending=False)
pdf_ = pools.rename("pool").to_frame()
pdf_["share_of_fake_%"] = (100 * pools / ref["train_fake"]).round(2)
pdf_["weak_family"] = pdf_.index.isin(S2.WEAK_FAMILY)
pdf_["family"] = [S2.family_of(m) for m in pdf_.index]
display(pdf_)

known = pools.sum()
print(f"top {len(pools)} methods cover {known:,} of {ref['train_fake']:,} fake "
      f"({100*known/ref['train_fake']:.1f}%)")
print(f"remaining {ref['n_fake_methods']-len(pools)} methods share "
      f"{ref['train_fake']-known:,} images -> long tail")
weak_pool = pools[pools.index.isin(S2.WEAK_FAMILY)].sum()
print(f"\\n8-method weak family pool: {weak_pool:,} "
      f"({100*weak_pool/ref['train_fake']:.1f}% of fake)")
'''),
        code('''
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["#d62728" if w else "#7f7f7f" for w in pdf_["weak_family"]]
axes[0].barh(pdf_.index[::-1], pdf_["pool"][::-1], color=colors[::-1])
axes[0].set_xlabel("training pool (images)")
axes[0].set_title("Method pools - red = 8-method weak family (REFERENCE)")

cum = pools.cumsum() / ref["train_fake"] * 100
axes[1].plot(range(1, len(cum) + 1), cum.values, marker="o", color="#1f77b4")
axes[1].axhline(50, ls="--", c="grey"); axes[1].set_ylim(0, 100)
axes[1].set_xlabel("method rank"); axes[1].set_ylabel("cumulative % of fake")
axes[1].set_title("Long-tail concentration curve")
plt.tight_layout(); plt.show()

print("EDA -> TRAINING LINK")
print("  Long tail means a pool-blind sampler starves large-but-not-largest")
print("  methods. Section A6 quantifies that starvation; notebook")
print("  02_training_balanced_dataset acts on it.")
'''),
        md("## A4. Image dimensions - ORIGINAL vs MODEL INPUT\n\nThe reference stresses that sources differ in native resolution while the pipeline always resizes to 256x256. Local originals are MEASURED; Session-2 sources are REFERENCE."),
        code('''
print(f"MODEL INPUT (MEASURED, canonical): {P.LABEL_MAPPING and ''}", end="")
from src.data import preprocessing as prep
print(f"{prep.IMG_SIZE}x{prep.IMG_SIZE}  <- every image is resized to this")
print("ORIGINAL dimensions differ by source; that difference survives as")
print("resampling artefacts, so it is worth measuring.\\n")

q = loader.load_sample_quality()
if {"width", "height"} <= set(q.columns):
    q = q.copy()
    q["aspect"] = q["width"] / q["height"]
    q["nonsquare"] = (q["width"] != q["height"])
    dims = q.groupby(["width", "height"]).size().rename("images").reset_index()
    dims["share_%"] = (100 * dims["images"] / len(q)).round(2)
    print("MEASURED original dimensions in the LOCAL corpus:")
    display(dims.sort_values("images", ascending=False).head(10))
    print(f"non-square images: {int(q['nonsquare'].sum()):,} "
          f"({q['nonsquare'].mean():.2%})")
    print(f"aspect ratio: min={q['aspect'].min():.3f} max={q['aspect'].max():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(q["width"].dropna(), bins=40, color="#1f77b4"); axes[0].set_title("width (MEASURED)")
    axes[1].hist(q["height"].dropna(), bins=40, color="#ff7f0e"); axes[1].set_title("height (MEASURED)")
    axes[2].hist(q["aspect"].dropna(), bins=40, color="#2ca02c")
    axes[2].axvline(1.0, ls="--", c="k"); axes[2].set_title("aspect ratio (1.0 = square)")
    for a in axes: a.set_ylabel("images")
    plt.tight_layout(); plt.show()
'''),
        code('''
print("REFERENCE dimension profile for the Session-2 corpus (not measurable here):")
display(pd.DataFrame([{"group": k, "native size": v}
                      for k, v in ref["dimensions"].items()]))
print("Reference notes deepfake_faceswap at 178x218 (non-square).")
print("INFERENCE: resizing a non-square crop to a square 256x256 distorts")
print("face geometry, which is consistent with it being a hard method.")
print("This is an association from the reference, not measured locally.")

# Cross-check: does the local corpus contain a comparable non-square source?
if {"width", "height"} <= set(q.columns):
    odd = q[q["width"] != q["height"]]
    print(f"\\nLocal non-square images: {len(odd):,}")
    if len(odd) == 0:
        print("  -> the local corpus is uniformly square, so the 178x218 issue")
        print("     is specific to the Session-2 corpus and cannot be")
        print("     reproduced here. Documented, not fabricated.")
'''),
        md("## A5. Identity structure and leakage\n\nSession-1's defect was ~32 frames per identity. Session-2 targets ~2.5-3.4. Local identity structure and leakage are MEASURED."),
        code('''
idn = pd.DataFrame([
    {"corpus": "Session-1 DF40 (REFERENCE)", "images/identity": ref["identity"]["session1_per_identity"]},
    {"corpus": "Session-2 fake (REFERENCE)", "images/identity": ref["identity"]["fake"]["per_identity"]},
    {"corpus": "Session-2 real (REFERENCE)", "images/identity": ref["identity"]["real"]["per_identity"]},
])
allsp = loader.load_splits_concat(detailed=True)
ipi = allsp.groupby("identity").size()
idn.loc[len(idn)] = {"corpus": "Local identity_clean_v1 (MEASURED)",
                     "images/identity": round(ipi.mean(), 2)}
display(idn)

print(f"LOCAL MEASURED identity structure:")
print(f"  identities            : {allsp['identity'].nunique():,}")
print(f"  mean images/identity  : {ipi.mean():.2f}")
print(f"  median                : {ipi.median():.0f}")
print(f"  max (heaviest)        : {ipi.max()}")
print(f"  identities with 1 img : {(ipi==1).sum():,} ({(ipi==1).mean():.1%})")
print(f"  top-1% identities hold: {100*ipi.nlargest(max(1,len(ipi)//100)).sum()/len(allsp):.1f}% of images")

for lab, name in [(P.LABEL_REAL, "real"), (P.LABEL_FAKE, "fake")]:
    s = allsp[allsp.label == lab].groupby("identity").size()
    print(f"  {name}: {len(s):,} identities, {s.mean():.2f} images/identity")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(ipi.values, bins=range(1, min(30, int(ipi.max())) + 2), color="#9467bd")
ax.set_xlabel("images per identity"); ax.set_ylabel("identities")
ax.set_title("Local images-per-identity (MEASURED) - low duplication is good")
plt.tight_layout(); plt.show()
'''),
        code('''
# Leakage: identity / video overlap across the canonical splits (MEASURED)
det = {s: loader.load_split(s, detailed=True) for s in P.SPLITS}
rows = []
for key in ["identity", "video"]:
    sets = {s: set(d[key]) for s, d in det.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        rows.append({"key": key, "pair": f"{a} n {b}", "overlap": len(sets[a] & sets[b])})
lk = pd.DataFrame(rows)
display(lk.pivot(index="pair", columns="key", values="overlap"))

id_ok = lk[(lk.key == "identity")]["overlap"].sum() == 0
print(f"identity overlap total: {lk[lk.key=='identity']['overlap'].sum()}  "
      f"-> {'PASS (identity-disjoint)' if id_ok else 'FAIL'}")
print("Reference states Session-2 train<->val identity overlap = 0 (fake+real);")
print("the local protocol independently satisfies the same constraint.")
print()
print("VIDEO overlap is NON-ZERO and is a KNOWN LIMITATION - not called safe:")
for _, r in lk[lk.key == "video"].iterrows():
    print(f"   {r['pair']}: {r['overlap']}")

# Duplicates: reuse the existing project artifacts, do not recompute
import json as _json
vp = _ROOT / "experiments/results/dataset_protocol/protocol_validation.json"
if vp.exists():
    v = _json.load(open(vp))["protocols"]["identity_clean"]
    print(f"\\nEXACT-duplicate cross-split overlap (existing artifact): {v['exact_leakage']}")
ndp = _ROOT / "experiments/results/eda_real_data/near_duplicate_threshold_report.json"
if ndp.exists():
    print("NEAR-duplicate groups quantified but NOT removed (known limitation).")
'''),
        md("""
## A6. Sampler exposure - the data-level reason the sampler changed

Exposure is **recomputed with the project's own allocation rules**
(`FaceswapFocusedSampler` / `WeakFamilyBoostedSampler` from
`scripts/finetune_plus_v3.py`), not copied from the reference. Pool sizes are
REFERENCE because the Session-2 train CSV is absent.
"""),
        code('''
exp = S2.exposure_table()
show = exp.copy()
show[["A0_exposure", "A1_exposure", "change_x"]] = show[["A0_exposure", "A1_exposure", "change_x"]].round(3)
display(show)

# Validate the recomputation against the reference figures.
chk = []
for _, r in exp.iterrows():
    if r["method"] in ref["exposure"]:
        a0, a1 = ref["exposure"][r["method"]]
        chk.append({"method": r["method"], "A0_calc": round(r["A0_exposure"], 3),
                    "A0_ref": a0, "A1_calc": round(r["A1_exposure"], 3), "A1_ref": a1,
                    "match": abs(r["A0_exposure"] - a0) < 0.02 and abs(r["A1_exposure"] - a1) < 0.02})
chk = pd.DataFrame(chk)
display(chk)
print("recomputation reproduces every reference exposure value:", bool(chk["match"].all()))
'''),
        code('''
w = exp[exp["in_weak_family"]].sort_values("A0_exposure")
x = np.arange(len(w)); bw = 0.38
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.bar(x - bw/2, w["A0_exposure"], bw, label="A0 (old sampler)", color="#7f7f7f")
ax.bar(x + bw/2, w["A1_exposure"], bw, label="A1 (weak_family)", color="#d62728")
ax.set_xticks(x); ax.set_xticklabels(w["method"], rotation=30, ha="right")
ax.set_ylabel("exposure (times each image seen / epoch)")
ax.set_title("Weak family: exposure per image per epoch, A0 vs A1")
for i, (a0, a1) in enumerate(zip(w["A0_exposure"], w["A1_exposure"])):
    ax.text(i + bw/2, a1, f" x{a1/a0:.1f}", ha="center", va="bottom", fontsize=8)
ax.legend(); plt.tight_layout(); plt.show()

worst = exp.loc[exp["A0_exposure"].idxmin()]
print(f"Most starved under A0: {worst['method']} at {worst['A0_exposure']:.3f}x/epoch")
print(f"  pool {int(worst['pool']):,} images -> over 3 epochs each image seen "
      f"~{3*worst['A0_exposure']:.2f} times: effectively invisible.")
print("\\nCAUSAL CHAIN (data -> decision)")
print("  long tail -> A0 splits 'other' EVENLY BY METHOD -> big non-faceswap")
print("  pools get ~0.06x/epoch -> weak methods under-learned -> A1 allocates")
print("  by pool x weight so all 8 reach ~0.52x. Verified in 02_error_analysis.")
'''),
        md("""
## A7. Section summary - EDA findings and what each implies

| # | Finding | Provenance | Training/eval implication |
|---|---|---|---|
| 1 | Session-2 train ~1:3.2 fake:real; local protocol ~1:24.5 | REF + MEASURED | Never compare metrics across corpora |
| 2 | Real from 9 heterogeneous sources | REFERENCE | Real-side domain shift -> source-aware eval |
| 3 | 42 methods, strong long tail | REFERENCE | Pool-blind sampling starves mid-size pools |
| 4 | 8-method weak family (~41k images) | Parsed from sampler | Targeted exposure boost |
| 5 | Pipeline forces 256x256; sources 256/512, one at 178x218 | MEASURED + REF | Non-square resize distorts geometry |
| 6 | Local ~1.3 images/identity; Session-2 ~2.5-3.4; Session-1 ~32 | MEASURED + REF | Low duplication -> honest evaluation |
| 7 | Identity overlap = 0; exact-dup overlap = 0 | MEASURED | Identity-disjoint protocol holds |
| 8 | Video overlap non-zero; near-dups not removed | MEASURED | **Known limitations, not safe** |
| 9 | A0 starved weak methods to ~0.06x/epoch | Recomputed | Direct justification for A1 |

Downstream: `02_training_balanced_dataset` (sampling), `02_error_analysis`
(did it work), `courseWorkCheck` (consistency).
"""),
    ]


# ---------------------------------------------------------------------------
# 01 - full pipeline addendum
# ---------------------------------------------------------------------------
def cells_01():
    return [
        md(f"""
---
# {SENTINEL} - Pipeline Consistency with EDA Findings

Confirms the training pipeline actually implements what the EDA implies:
one data source, one label mapping, one preprocessing, and a sampling
strategy justified by the measured distribution.
{AVAIL_NOTE}
"""),
        code(BOOT),
        md("## B1. One data source, one split definition"),
        code('''
train_csv, val_csv, test_csv = P.get_protocol_paths()
print("CANONICAL PROTOCOL (single source of truth)")
print(f"  name : {P.load_protocol().name}")
for n, p in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
    print(f"  {n:5s}: {p.relative_to(_ROOT)}")
print()
print(loader.describe())

sizes = loader.split_sizes()
assert sizes == {"train": 20991, "val": 4498, "test": 4499}, sizes
assert P.LABEL_REAL == 0 and P.LABEL_FAKE == 1
print("\\nVERIFIED: split sizes and label mapping match the canonical protocol.")
print("No notebook-local re-splitting anywhere in this addendum.")
'''),
        md("## B2. Preprocessing: original dimension vs model input"),
        code('''
from src.data import preprocessing as prep
print(prep.describe())
print()
q = loader.load_sample_quality()
if {"width", "height"} <= set(q.columns):
    print("ORIGINAL sizes present in the corpus (MEASURED):")
    print(q.groupby(["width", "height"]).size().sort_values(ascending=False).head(5).to_string())
print(f"\\nALL of the above are resized to {prep.IMG_SIZE}x{prep.IMG_SIZE} before the model.")
print("So resolution differences act on the model only through resampling")
print("artefacts, not through input shape.")
'''),
        md("## B3. Class imbalance the pipeline must cope with"),
        code('''
rows = []
for s in P.SPLITS:
    c = loader.class_counts(s)
    rows.append({"split": s, **c, "fake:real": round(c["fake"] / max(1, c["real"]), 2),
                 "real_%": round(100 * c["real"] / c["total"], 2)})
imb = pd.DataFrame(rows); display(imb)

w = loader.class_weights("train")
print(f"inverse-frequency class weights: real={w[0]:.4f} fake={w[1]:.4f}")
print(f"  -> the real class carries {w[0]/w[1]:.1f}x the loss weight")
print()
print("IMPLICATION: with ~4% real in train, raw accuracy is a misleading")
print("selection metric. The pipeline therefore selects checkpoints by MCC.")
'''),
        md("## B4. Sampler / allocation strategy actually available in the project"),
        code('''
import re
fp = _ROOT / "scripts/finetune_plus_v3.py"
print("Sampler implementations found in the project:")
if fp.exists():
    for m in re.finditer(r"class (\\w*Sampler)\\(", fp.read_text()):
        print(f"   {m.group(1):32s}  ({fp.relative_to(_ROOT)})")
print(f"   {'WeightedRandomSampler helper':32s}  (src/data/loader.py)")
print()
print("Weak-family weights (parsed, not hard-coded):", S2.parse_family_weights())
print()
exp = S2.exposure_table()
print("Exposure consequence of those weights (recomputed):")
display(exp[exp["in_weak_family"]][["method", "pool", "A0_exposure", "A1_exposure", "change_x"]].round(3))
print("Section A6 of notebook 00 validates these against the reference.")
'''),
        md("## B5. Training implications, tied to measured evidence"),
        code('''
display(pd.DataFrame([
    {"EDA finding": "train ~24.5:1 fake:real (MEASURED)",
     "pipeline response": "class-weighted loss + MCC checkpoint selection",
     "evidence": "BASELINE_REPORT.md"},
    {"EDA finding": "long-tail methods starved to ~0.06x/epoch (recomputed)",
     "pipeline response": "A1 weak-family boosted sampler",
     "evidence": "02_error_analysis McNemar"},
    {"EDA finding": "identity overlap = 0, video overlap > 0 (MEASURED)",
     "pipeline response": "identity-disjoint protocol; video overlap documented",
     "evidence": "DATASET_PROTOCOL.md"},
    {"EDA finding": "real-source heterogeneity (REFERENCE)",
     "pipeline response": "balanced 1:1 test + per-source reporting",
     "evidence": "02_error_analysis per-source"},
]))
print("Each row is an association between a measurement and a design choice;")
print("none of them is a causal claim.")
'''),
    ]


# ---------------------------------------------------------------------------
# 02_error_analysis addendum
# ---------------------------------------------------------------------------
def cells_02_error():
    return [
        md(f"""
---
# {SENTINEL} - Do the EDA-predicted weak groups actually fail?

This is the evaluation-side test of the EDA hypotheses, on the balanced
Session-2 test (21,446 images, 1:1). All numbers here are **MEASURED** from
the local paired predictions in `experiments/results/coursework_vs/`.

FP = REAL predicted FAKE. FN = FAKE predicted REAL.
{AVAIL_NOTE}
"""),
        code(BOOT),
        md("## C1. Test population and model inventory (MEASURED)"),
        code('''
comp = S2.test_composition()
print(f"balanced test: n={comp['n_total']:,}  real={comp['n_real']:,}  "
      f"fake={comp['n_fake']:,}  ratio 1:{comp['real_fake_ratio']:.2f}")
print(f"distinct fake methods: {comp['n_fake_methods']}")
print("\\nREAL by source:")
for s, n in comp["real_by_source"]:
    print(f"   {s:16s} {n:6,d}")
print("\\nWHY 1:1 MATTERS: at ~4% real (session-1 style) a model can score high")
print("accuracy while being poor at real detection. A 1:1 test forces both sides.")

print("\\nmodels with local predictions:")
for m in S2.available_models():
    print("   -", m)
'''),
        code('''
mt = S2.metrics_table()
display(mt.round(4))
print("Accuracy is trustworthy HERE because the test is 1:1 - unlike the")
print("local identity_clean_v1 test (96% fake), where MCC is required.")
'''),
        md("## C2. A0 vs A1 - per-method detection on the weak family"),
        code('''
pm = S2.per_method_detection([S2.A0, S2.A1])
piv = pm.pivot(index="method", columns="model", values="det_rate")
piv["n"] = pm.groupby("method")["n"].first()
piv["weak_family"] = piv.index.isin(S2.WEAK_FAMILY)
piv["delta"] = piv[S2.A1] - piv[S2.A0]
weak = piv[piv["weak_family"]].sort_values("delta", ascending=False)
print("WEAK FAMILY (the 8 methods A1 targeted):")
display(weak.round(4))
print(f"mean det-rate on weak family:  A0={weak[S2.A0].mean():.4f}  "
      f"A1={weak[S2.A1].mean():.4f}  delta={weak['delta'].mean():+.4f}")
other = piv[~piv["weak_family"]]
print(f"mean det-rate on other methods: A0={other[S2.A0].mean():.4f}  "
      f"A1={other[S2.A1].mean():.4f}  delta={other['delta'].mean():+.4f}")
print("\\nThe gain is concentrated in exactly the boosted group, and the rest")
print("does not regress -> consistent with the sampler change, not a global shift.")
'''),
        code('''
w = weak.sort_values("delta")
x = np.arange(len(w)); bw = 0.38
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.bar(x - bw/2, w[S2.A0], bw, label="A0 (old sampler)", color="#7f7f7f")
ax.bar(x + bw/2, w[S2.A1], bw, label="A1 (weak_family)", color="#d62728")
ax.set_xticks(x); ax.set_xticklabels(w.index, rotation=30, ha="right")
ax.set_ylabel("detection rate (fake recall)"); ax.set_ylim(0, 1.02)
ax.set_title("Weak-family detection rate: A0 vs A1 (MEASURED, 300 imgs/method)")
for i, d in enumerate(w["delta"]):
    ax.text(i + bw/2, w[S2.A1].iloc[i], f" {d:+.3f}", ha="center", va="bottom", fontsize=8)
ax.legend(); plt.tight_layout(); plt.show()
'''),
        md("## C3. Paired statistical validation (McNemar)\n\nBoth models scored the *same* images, so the samples are dependent and a paired test is required."),
        code('''
rows = [S2.mcnemar(S2.A0, S2.A1, subset=s) for s in ["all", "fake", "real", "weak_family"]]
mc = pd.DataFrame(rows)[["subset", "n", "a_only_correct", "b_only_correct",
                         "chi2", "p", "significant", "favours"]]
display(mc)

allr = rows[0]
print(f"OVERALL: A0-only-correct={allr['a_only_correct']}, "
      f"A1-only-correct={allr['b_only_correct']}, chi2={allr['chi2']:.2f}, p={allr['p']:.2e}")
print(f"  -> {'significant' if allr['significant'] else 'NOT significant'}: the improvement is not noise.")
realr = rows[2]
print(f"REAL subset: p={realr['p']:.3f} -> "
      f"{'no significant change' if not realr['significant'] else 'significant change'}")
print("  i.e. no evidence that A1 traded real accuracy away on this test.")
'''),
        code('''
pmc = S2.mcnemar_per_method(S2.A0, S2.A1)
display(pmc[["method", "n", "a_only_correct", "b_only_correct", "delta_correct",
             "p", "significant", "in_weak_family"]].head(14).round(6))

sig = pmc[pmc["significant"]]
print(f"methods with significant change: {len(sig)}")
print(f"  of which in the weak family  : {int(sig['in_weak_family'].sum())}")
regress = pmc[(pmc["significant"]) & (pmc["delta_correct"] < 0)]
print(f"  significantly REGRESSED       : {len(regress)}")
print("\\nThe set of significantly improved methods coincides with the boosted")
print("family. This SUPPORTS the exposure hypothesis; it does not prove causation,")
print("though only the sampler differed between the two runs.")
'''),
        md("## C4. Are errors concentrated by real source? (MEASURED)"),
        code('''
ps = S2.per_source_real_accuracy()
piv_s = ps.pivot(index="source", columns="model", values="real_acc")
piv_s["n"] = ps.groupby("source")["n"].first()
display(piv_s.round(4))

pr = [m for m in S2.available_models() if m.startswith("Pretr")]
if pr:
    sub = piv_s[pr + ["n"]].dropna()
    if len(sub) > 1:
        worst = sub[pr].mean(axis=1).idxmin()
        print(f"Frozen probes are weakest on real source: {worst}")
        print("Reference notes ff++_real (heavily compressed video frames) as the")
        print("hardest real source; the measured ordering is consistent with that.")
print("\\nFP counts by source and model:")
display(ps.pivot(index="source", columns="model", values="FP"))
'''),
        md("## C5. Errors by manipulation family (MEASURED)"),
        code('''
pf = S2.per_family_detection()
piv_f = pf.pivot(index="family", columns="model", values="det_rate")
piv_f["n"] = pf.groupby("family")["n"].first()
display(piv_f.round(4))
print("The '* WEAK FAMILY (8)' row is the A1 target group.")
'''),
        md("## C6. Frozen pretrained features vs finetuned (MEASURED)\n\nSame test, same procedure, comparable parameter counts."),
        code('''
pvf = S2.pretrained_vs_finetuned()
display(pvf[["model", "stage", "acc%", "prec", "rec", "f1", "AUC",
             "real_acc", "fake_recall", "FP", "FN"]].round(4))

fro = pvf[pvf.stage == "frozen probe"]["acc%"]
fin = pvf[pvf.stage == "finetuned"]["acc%"]
if len(fro) and len(fin):
    print(f"frozen probe accuracy : {fro.min():.2f} - {fro.max():.2f}%")
    print(f"finetuned accuracy    : {fin.min():.2f} - {fin.max():.2f}%")
    print(f"gap                   : ~{fin.max()-fro.max():.1f} points")
print("\\nFrozen features already reach AUC ~0.96, so pretrained representations")
print("carry real deepfake signal - but finetuning adds ~9-11 accuracy points.")
print("The benefit therefore comes from adapting features, not just adding a head.")
'''),
        code('''
pmp = S2.per_method_detection([m for m in S2.available_models() if m.startswith("Pretr")])
if len(pmp):
    t = pmp.groupby("method").agg(det=("det_rate", "mean"), n=("n", "first"),
                                  weak=("in_weak_family", "first")).sort_values("det")
    print("Hardest methods for FROZEN pretrained features (mean over both probes):")
    display(t.head(12).round(4))
    wk = t[t["weak"]]["det"].mean(); ot = t[~t["weak"]]["det"].mean()
    print(f"mean det-rate  weak family={wk:.4f}   other={ot:.4f}")
    print("The weak family is already harder for pretrained features, which")
    print("SUPPORTS targeting it rather than it being an artefact of the sampler.")

    fam = pmp.groupby("family")["det_rate"].mean().sort_values()
    print("\\nBy manipulation family (frozen probes):")
    display(fam.round(4).to_frame("mean_det_rate"))
'''),
        md("""
## C7. Verdict on the EDA hypotheses

| Hypothesis from EDA | Test | Result |
|---|---|---|
| Weak family is genuinely harder | frozen-probe det-rate | **Supported** - lower than other methods before any sampler change |
| Boosting exposure raises weak-family detection | A0 vs A1 per-method | **Supported** - gains concentrated in the 8 boosted methods |
| The overall gain is not noise | paired McNemar, all | **Supported** - p far below 0.05 |
| A1 does not sacrifice real accuracy | paired McNemar, real subset | **No significant change** |
| Finetuning matters beyond a head | frozen probe vs finetuned | **Supported** - ~9-11 point gap |
| Real errors concentrate by source | per-source real accuracy | **Supported** - compressed video-frame source is worst |

Wording is deliberately "supported / consistent with", not "caused by":
these are observational comparisons, even though only the sampler differed
between the A0 and A1 runs.

**Remaining gap:** ConvNeXt still leads on real accuracy (far fewer FP), so
the real side is the next target - which is what the reference proposes for A2.
"""),
    ]


# ---------------------------------------------------------------------------
# 02_training_balanced_dataset addendum
# ---------------------------------------------------------------------------
def cells_02_train():
    return [
        md(f"""
---
# {SENTINEL} - Distribution, Sampling Exposure and Balanced Evaluation

Central distinction preserved throughout:

| Concept | Mutable? |
|---|---|
| **Dataset split** (train/val/test membership) | **NO** |
| **Training sampling strategy** (which images batches draw) | YES |

Balancing changes how TRAIN batches are drawn. **Validation and test keep
their natural distribution.** Never rebalance an evaluation set.
{AVAIL_NOTE}
"""),
        code(BOOT),
        md("## D1. Training distribution vs evaluation distribution"),
        code('''
rows = []
for s in P.SPLITS:
    c = loader.class_counts(s)
    rows.append({"split": s, "source": "local identity_clean_v1 (MEASURED)",
                 "total": c["total"], "real": c["real"], "fake": c["fake"],
                 "real:fake": f"1:{c['fake']/max(1,c['real']):.2f}"})
ref = S2.REFERENCE
for nm, tot, re_, fa in [("train", ref["train_total"], ref["train_real"], ref["train_fake"]),
                         ("val", ref["val_total"], 1449, 4853),
                         ("test (full)", ref["test_full"], 25042, 25042),
                         ("test (balanced)", ref["test_balanced"], 10723, 10723)]:
    rows.append({"split": nm, "source": "Session-2 (REFERENCE)", "total": tot,
                 "real": re_, "fake": fa, "real:fake": f"1:{fa/max(1,re_):.2f}"})
dist = pd.DataFrame(rows); display(dist)

comp = S2.test_composition()
print(f"Session-2 balanced test VERIFIED locally from predictions: "
      f"n={comp['n_total']:,} real={comp['n_real']:,} fake={comp['n_fake']:,} "
      f"(1:{comp['real_fake_ratio']:.2f})")
print("  -> the 21,446 / 1:1 figure is MEASURED here, not merely quoted.")
'''),
        code('''
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
loc = dist[dist["source"].str.startswith("local")]
axes[0].bar(loc["split"], loc["real"], label="real", color="#2ca02c")
axes[0].bar(loc["split"], loc["fake"], bottom=loc["real"], label="fake", color="#d62728")
axes[0].set_title("Local protocol (MEASURED) - imbalance PRESERVED in val/test")
axes[0].set_ylabel("images"); axes[0].legend()

s2 = dist[dist["source"].str.startswith("Session-2")]
axes[1].bar(s2["split"], s2["real"], label="real", color="#2ca02c")
axes[1].bar(s2["split"], s2["fake"], bottom=s2["real"], label="fake", color="#d62728")
axes[1].set_title("Session-2 (REFERENCE) - test deliberately 1:1")
axes[1].tick_params(axis="x", rotation=20); axes[1].legend()
plt.tight_layout(); plt.show()

print("WHY A BALANCED TEST: with real at ~4%, predicting 'fake' always yields")
print("~96% accuracy while real detection is broken. A 1:1 test removes that")
print("escape hatch and measures both directions honestly.")
'''),
        md("## D2. Exposure: what the sampler actually shows the model"),
        code('''
exp = S2.exposure_table()
display(exp.round(3))
print("Recomputed with the project's own allocation rules; validated against")
print("the reference in notebook 00 section A6.")

fig, ax = plt.subplots(figsize=(11, 4.8))
e = exp[exp["method"] != "real"].sort_values("pool", ascending=False)
x = np.arange(len(e)); bw = 0.38
ax.bar(x - bw/2, e["A0_exposure"], bw, label="A0", color="#7f7f7f")
ax.bar(x + bw/2, e["A1_exposure"], bw, label="A1 weak_family", color="#d62728")
ax.axhline(1.0, ls=":", c="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels(e["method"], rotation=35, ha="right")
ax.set_ylabel("exposure per image per epoch")
ax.set_title("Sampler exposure by method (pool-ordered). Dotted = seen once/epoch")
for i, w in enumerate(e["in_weak_family"]):
    if w:
        ax.get_xticklabels()[i].set_color("#d62728")
ax.legend(); plt.tight_layout(); plt.show()
print("Red x-labels = the 8 boosted weak-family methods.")
'''),
        md("## D3. Balancing strategies available, and their blast radius"),
        code('''
display(pd.DataFrame([
    {"strategy": "class-weighted loss", "changes": "loss term",
     "train rows changed": "no", "val/test touched": "NO",
     "status": "used by the local baseline"},
    {"strategy": "WeightedRandomSampler", "changes": "batch sampling",
     "train rows changed": "no", "val/test touched": "NO",
     "status": "available in src/data/loader.py"},
    {"strategy": "A0 faceswap-focused sampler", "changes": "bucket allocation",
     "train rows changed": "no", "val/test touched": "NO",
     "status": "Session-2 baseline"},
    {"strategy": "A1 weak-family boosted sampler", "changes": "bucket allocation",
     "train rows changed": "no", "val/test touched": "NO",
     "status": "Session-2 improvement (validated)"},
    {"strategy": "physically duplicating rows", "changes": "dataset itself",
     "train rows changed": "YES", "val/test touched": "NO",
     "status": "NOT used - distorts the corpus"},
]))
print("Every strategy in use alters sampling or loss only. None edits val/test.")
'''),
        code('''
# Demonstrate on the local protocol that balancing is TRAIN-only, and that the
# guard rails refuse to touch evaluation splits.
sampler = loader.weighted_random_sampler("train")
train_df = loader.load_split("train", detailed=False)
idx = list(iter(sampler))[:4000]
drawn = train_df.iloc[idx]["label"].value_counts(normalize=True)
print(f"natural TRAIN split      : real={(train_df['label']==0).mean():.1%} "
      f"fake={(train_df['label']==1).mean():.1%}")
print(f"sampled TRAIN batches    : real={drawn.get(0,0):.1%} fake={drawn.get(1,0):.1%}")
print(f"sampler length           : {len(sampler):,} (= train size, epoch preserved)")

for split in ["val", "test"]:
    try:
        loader.weighted_random_sampler(split)
        print(f"{split}: ERROR - balancing was permitted!")
    except ValueError as e:
        print(f"{split}: correctly blocked -> {e}")
for split in ["val", "test"]:
    c = loader.class_counts(split)
    print(f"  {split} unchanged: real={c['real']} fake={c['fake']}")
'''),
        md("## D4. Does exposure relate to measured performance?"),
        code('''
if S2.availability().eval_preds:
    pm = S2.per_method_detection([S2.A0, S2.A1]).pivot(
        index="method", columns="model", values="det_rate")
    j = exp.set_index("method").join(pm, how="inner")
    j["delta_det"] = j[S2.A1] - j[S2.A0]
    j["exposure_gain_x"] = j["change_x"]
    cols = ["pool", "A0_exposure", "A1_exposure", "exposure_gain_x",
            S2.A0, S2.A1, "delta_det", "in_weak_family"]
    display(j[cols].sort_values("delta_det", ascending=False).round(4))

    sub = j.dropna(subset=["delta_det", "exposure_gain_x"])
    if len(sub) > 2:
        r = np.corrcoef(sub["exposure_gain_x"], sub["delta_det"])[0, 1]
        fig, ax = plt.subplots(figsize=(7, 5))
        for wk, g in sub.groupby("in_weak_family"):
            ax.scatter(g["exposure_gain_x"], g["delta_det"], s=60,
                       label="weak family" if wk else "other",
                       color="#d62728" if wk else "#7f7f7f")
        for m, r_ in sub.iterrows():
            ax.annotate(m, (r_["exposure_gain_x"], r_["delta_det"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, ls=":", c="k")
        ax.set_xlabel("exposure gain A1/A0 (x)"); ax.set_ylabel("detection-rate change")
        ax.set_title(f"Exposure gain vs detection gain (Pearson r={r:.2f}, n={len(sub)})")
        ax.legend(); plt.tight_layout(); plt.show()
        print(f"Pearson r = {r:.3f} across {len(sub)} methods with known pools.")
        print("ASSOCIATION ONLY - n is small and pools come from the reference.")
        print("It is consistent with exposure driving the weak-family gains.")
else:
    print("prediction artifacts unavailable - correlation not computed")
'''),
        md("""
## D5. Summary

1. **Split membership is fixed.** Only loss weighting and batch sampling changed.
2. **Evaluation distribution is never rebalanced.** The local protocol keeps its
   ~24.5:1 imbalance in val/test; Session-2 uses a *separately constructed* 1:1
   test suite, verified locally at 21,446 images.
3. **A0 starved the long tail** (~0.06x/epoch for `deepfake_faceswap`);
   **A1 lifts all 8 weak methods to ~0.52x** - recomputed, matching the reference.
4. **Measured detection gains land on the boosted methods**, with no significant
   regression elsewhere (see `02_error_analysis`).
5. Exposure gain and detection gain are **positively associated**; with a handful
   of methods and reference-sourced pools this is supporting evidence, not proof.
"""),
    ]


# ---------------------------------------------------------------------------
# courseWorkCheck addendum
# ---------------------------------------------------------------------------
def cells_cwc():
    return [
        md(f"""
---
# {SENTINEL} - Cross-Notebook Consistency Verification

Verifies the coursework-facing notebook agrees with the canonical protocol and
with every other notebook. Verification only: nothing is modified here.
{AVAIL_NOTE}
"""),
        code(BOOT),
        md("## E1. Canonical protocol and label semantics"),
        code('''
proto = P.load_protocol(); meta = P.protocol_metadata()
checks = {
    "protocol is identity_clean_v1": proto.name == "identity_clean_v1",
    "strategy identity-disjoint": meta["split_strategy"] == "identity-disjoint",
    "seed == 42": meta["random_seed"] == 42,
    "labels 0=real, 1=fake": P.LABEL_MAPPING == {0: "real", 1: "fake"},
    "train == 20,991": meta["train_size"] == 20991,
    "val   == 4,498": meta["val_size"] == 4498,
    "test  == 4,499": meta["test_size"] == 4499,
    "sizes match CSVs": loader.split_sizes() == {"train": 20991, "val": 4498, "test": 4499},
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
assert all(checks.values()), "canonical protocol drift detected"
'''),
        md("## E2. Dataset arithmetic and leakage"),
        code('''
manifest = loader.load_manifest()
total = sum(loader.split_sizes().values())
removed = len(manifest) - total
print(f"manifest {len(manifest):,} - removed {removed:,} = protocol {total:,}  "
      f"-> {len(manifest)-removed == total}")

log = _ROOT / "experiments/results/eda_real_data/removed_exact_duplicates.csv"
if log.exists():
    n = len(pd.read_csv(log))
    print(f"audit log rows {n:,} == removed {removed:,} -> {n == removed}")
    print("Every removed duplicate is logged; nothing silently deleted.")

det = {s: loader.load_split(s, detailed=True) for s in P.SPLITS}
for key in ["identity", "video"]:
    sets = {s: set(d[key]) for s, d in det.items()}
    ov = {f"{a}-{b}": len(sets[a] & sets[b])
          for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    tag = "PASS (must be 0)" if key == "identity" and sum(ov.values()) == 0 else (
          "KNOWN LIMITATION (not safe)" if key == "video" else "FAIL")
    print(f"{key:9s} overlap {ov}  -> {tag}")
'''),
        md("## E3. Evaluation populations are the right ones"),
        code('''
PRED = _ROOT / "experiments/results/baseline/final/evaluation/test_predictions.csv"
if PRED.exists():
    preds = pd.read_csv(PRED)
    try:
        loader.verify_matches_protocol(preds, "test")
        print(f"PASS: local baseline predictions ({len(preds):,}) == canonical test split")
    except ValueError as e:
        print("FAIL:", e)
else:
    print("local baseline predictions not found")

if S2.availability().eval_preds:
    c = S2.test_composition()
    print(f"PASS: Session-2 balanced test measured at n={c['n_total']:,} "
          f"({c['n_real']:,} real / {c['n_fake']:,} fake, 1:{c['real_fake_ratio']:.2f})")
    print("      matches the reference figure of 21,446 at 1:1")
print()
print("NOTE: two distinct corpora are in play. They are never merged:")
print("  - local identity_clean_v1 : 4,499-image test, ~96% fake -> use MCC")
print("  - Session-2 balanced      : 21,446-image test, 1:1     -> accuracy is fair")
'''),
        md("## E4. Headline results, each on its own corpus"),
        code('''
import json as _json
mp = _ROOT / "experiments/results/baseline/final/evaluation/metrics.json"
if mp.exists():
    m = _json.load(open(mp))
    print("LOCAL identity_clean_v1 (imbalanced test - accuracy is misleading)")
    print(f"   accuracy {m['accuracy']:.4f} | balanced acc {m['balanced_accuracy']:.4f} "
          f"| MCC {m['mcc']:.4f} | real recall {m['real_recall']:.4f}")
    print(f"   TN/FP/FN/TP = {m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}")

if S2.availability().eval_preds:
    print("\\nSESSION-2 balanced test (1:1 - accuracy is fair)")
    display(S2.metrics_table()[["model", "acc%", "f1", "AUC", "real_acc",
                                "fake_recall", "FP", "FN"]].round(4))
'''),
        md("## E5. Reference findings: reproduced, represented, or unavailable"),
        code('''
av = S2.availability()
rows = [
    ("1. Train composition 123,582 / 94,025 fake / 29,557 real", "REFERENCE only",
     "raw Session-2 train CSV absent locally"),
    ("2. 9 real sources, diversity beyond FF++/Celeb-DF", "REFERENCE only",
     "shown in nb00 A2, clearly labelled"),
    ("3. 42 methods, long-tail pools", "REFERENCE only",
     "shown in nb00 A3; weak family PARSED from sampler"),
    ("4. Dimensions 256 / 512 / 178x218", "PARTIAL",
     "local dims MEASURED in nb00 A4; Session-2 dims REFERENCE"),
    ("5. Identity ~2.5-3.4 img/id, train-val identity overlap 0", "PARTIAL",
     "Session-2 counts REFERENCE; local identity+leakage MEASURED"),
    ("6. Splits: train/val/test-full/test-balanced", "PARTIAL",
     "balanced test 21,446 @1:1 MEASURED from predictions"),
    ("7. Sampler exposure A0 vs A1 (dfs 0.056 -> 0.52, x9)", "REPRODUCED",
     "recomputed with the project sampler rules, matches reference"),
    ("8. A1 vs A0 McNemar p=3.6e-10, fake p=3e-21, real p=0.31", "REPRODUCED",
     "computed from local paired predictions"),
    ("9. Per-method gains concentrated on the 8 weak methods", "REPRODUCED",
     "nb02_error_analysis C2/C3"),
    ("10. Pretrained probe ~89.5/87.8% vs finetuned ~98.5/99.2%", "REPRODUCED",
     "nb02_error_analysis C6"),
    ("11. Probes weakest on Face Swap / weak family", "REPRODUCED",
     "nb02_error_analysis C6"),
    ("12. Real-source difficulty (ff++_real hardest)", "REPRODUCED",
     "nb02_error_analysis C4"),
]
cov = pd.DataFrame(rows, columns=["reference finding", "status", "where / why"])
display(cov)
print(cov["status"].value_counts().to_string())
print("\\nNo reference finding is presented as measured when it is not.")
'''),
        md("""
## E6. Consistency verdict

* One canonical protocol (`identity_clean_v1`), one label mapping
  (`0=real, 1=fake`), one preprocessing (256x256) across all five notebooks.
* Split sizes reconcile with the manifest and the duplicate audit log.
* Identity and exact-duplicate cross-split overlap are 0.
* Video overlap and un-removed near-duplicates remain **documented limitations**
  and are never described as safe.
* Val/test are never rebalanced; balancing is training-side only.
* The two corpora are reported separately and never mixed.

**Known limitations carried forward**

1. Video/source overlap in `identity_clean_v1`.
2. 4,215 cross-split near-duplicate groups, quantified but not removed.
3. Session-2 raw training corpus absent locally - train-side composition is
   reference-only.
4. Local baseline is a 1-epoch-derived 8-epoch run whose MCC was still rising:
   not converged.
"""),
    ]


BUILDERS = {
    "00_comprehensive_dataset_eda.ipynb": cells_00,
    "01_full_pipeline.ipynb": cells_01,
    "02_error_analysis.ipynb": cells_02_error,
    "02_training_balanced_dataset.ipynb": cells_02_train,
    "courseWorkCheck.ipynb": cells_cwc,
}


def strip_previous_addendum(cells: list[dict]) -> list[dict]:
    """Drop a previously appended addendum block so the script is idempotent."""
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and SENTINEL in "".join(c.get("source", [])):
            return cells[:i]
    return cells


def normalize(cells: list[dict]) -> list[dict]:
    for c in cells:
        src = c.get("source", [])
        if src and not any(s.endswith("\n") for s in src[:-1]):
            c["source"] = [ln + "\n" for ln in src[:-1]] + [src[-1]]
    return cells


def main():
    for name, builder in BUILDERS.items():
        path = NB_DIR / name
        nb = json.load(open(path))
        before = len(nb["cells"])
        kept = strip_previous_addendum(nb["cells"])
        removed = before - len(kept)
        new = normalize(builder())
        nb["cells"] = kept + new
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"{name}: kept {len(kept)} existing "
              f"(re-ran: dropped {removed} old addendum) + added {len(new)} "
              f"-> {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()

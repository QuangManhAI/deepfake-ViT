"""Session 2 EDA support -- shared, single implementation.

Reference: ``session2_finetune_data_eda.pdf`` (Session 2 finetune dataset +
sampler A0/A1 validation + pretrained linear-probe comparison).

Design rules
------------
* ONE implementation, imported by all notebooks. No copy-pasted loaders.
* Every number is either MEASURED from a local artifact or explicitly marked
  as REFERENCE (from the PDF) when the underlying data is not present locally.
  Nothing is fabricated.
* The Session 2 *raw training dataset* (``finetune_plus_train.csv``,
  ~123,582 images) and the 50k test suite are NOT in this repository. The
  Session 2 *evaluation artifacts* ARE, under
  ``experiments/results/coursework_vs/``, so evaluation-side findings are
  fully reproducible.

Use :func:`availability` to see exactly what can be computed.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Teammate-produced Session 2 evaluation artifacts (present locally).
CW_DIR = PROJECT_ROOT / "experiments" / "results" / "coursework_vs"
SUMMARY_JSON = CW_DIR / "summary.json"
PRETRAINED_SUMMARY_JSON = CW_DIR / "pretrained_summary.json"

# Session 2 raw dataset locations named in the reference (may be absent).
S2_TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "finetune_plus_train.csv"
S2_TEST_SUITE = PROJECT_ROOT / "data" / "deepfake_test_suite_full_50k"

# Scripts that DEFINE the Session 2 pipeline (present locally -> specs verifiable).
SCRIPT_PREP = PROJECT_ROOT / "scripts" / "prep_finetune_plus.py"
SCRIPT_FINETUNE = PROJECT_ROOT / "scripts" / "finetune_plus_v3.py"
SCRIPT_PROBE = PROJECT_ROOT / "scripts" / "probe_pretrained_test.py"

# ---------------------------------------------------------------------------
# Model registry: maps friendly names to the local prediction artifacts.
# A0 = old faceswap-focused sampler, A1 = new weak-family-boosted sampler.
# ---------------------------------------------------------------------------
MODELS = {
    "ViT-Plus A0 (old sampler)": "Plus_viT_v3_preds.npz",
    "ViT-Plus A1 (weak_family)": "plus_v3_s1_preds.npz",
    "ConvNeXt (finetuned ref)": "ConvNeXt_v3_preds.npz",
    "ViT-S/16 (session-1 ref)": "ViTsmall_server_v3_preds.npz",
    "Pretr ViT-Plus (frozen probe)": "Pretr_Plus_v3_preds.npz",
    "Pretr ConvNeXt (frozen probe)": "Pretr_ConvNeXt_v3_preds.npz",
}

A0 = "ViT-Plus A0 (old sampler)"
A1 = "ViT-Plus A1 (weak_family)"

# The 8-method "weak family" targeted by sampler A1.
# Sourced from WeakFamilyBoostedSampler.FAMILY_WEIGHTS in finetune_plus_v3.py
# (verified by parse_family_weights()); the PDF lists the same 8 methods.
WEAK_FAMILY = [
    "faceswap", "deepfake_faceswap", "wav2lip", "sadtalker",
    "fsgan", "facedancer", "inswap", "mobileswap",
]

# Method -> family grouping for the manipulation-type view.
METHOD_FAMILY = {
    # face swap
    "faceswap": "Face Swap", "deepfake_faceswap": "Face Swap", "simswap": "Face Swap",
    "blendface": "Face Swap", "inswap": "Face Swap", "mobileswap": "Face Swap",
    "facedancer": "Face Swap", "fsgan": "Face Swap", "uniface": "Face Swap",
    "e4s": "Face Swap", "deepfacelab": "Face Swap", "faceshifter": "Face Swap",
    # talking head / reenactment
    "wav2lip": "Talking Head", "sadtalker": "Talking Head", "heygen": "Talking Head",
    "fomm": "Reenactment", "mraa": "Reenactment", "MRAA": "Reenactment",
    "facevid2vid": "Reenactment", "one_shot_free": "Reenactment", "pirender": "Reenactment",
    "lia": "Reenactment", "tpsm": "Reenactment", "hyperreenact": "Reenactment",
    "danet": "Reenactment", "mcnet": "Reenactment",
    # synthesis
    "pixart": "Face Synthesis", "DiT": "Face Synthesis", "SiT": "Face Synthesis",
    "sd2.1": "Face Synthesis", "StyleGANXL": "Face Synthesis", "stylegan2": "Face Synthesis",
    "stylegan3": "Face Synthesis", "stylegan_xl": "Face Synthesis", "VQGAN": "Face Synthesis",
    "CollabDiff": "Face Synthesis", "ddim": "Face Synthesis", "MidJourney": "Face Synthesis",
    "whichisreal": "Face Synthesis", "sfhq_studio": "Face Synthesis",
    # editing
    "e4e": "Face Editing", "styleclip": "Face Editing", "stargan": "Face Editing",
    "starganv2": "Face Editing", "starganv1": "Face Editing",
}


def family_of(method: str) -> str:
    return METHOD_FAMILY.get(method, "Other")


# ---------------------------------------------------------------------------
# Availability -- be explicit about what is and is not reproducible
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Availability:
    train_csv: bool
    test_suite: bool
    eval_preds: bool
    summary: bool
    pretrained_summary: bool
    scripts: bool

    def report(self) -> str:
        def mark(b):
            return "AVAILABLE  " if b else "NOT PRESENT"
        lines = [
            "SESSION 2 ARTIFACT AVAILABILITY",
            f"  [{mark(self.train_csv)}] Session-2 train CSV        {S2_TRAIN_CSV.name}",
            f"  [{mark(self.test_suite)}] Session-2 50k test suite   {S2_TEST_SUITE.name}",
            f"  [{mark(self.eval_preds)}] Paired predictions (.npz)  {CW_DIR.name}/",
            f"  [{mark(self.summary)}] Finetune summary.json",
            f"  [{mark(self.pretrained_summary)}] Pretrained probe summary",
            f"  [{mark(self.scripts)}] Pipeline scripts (specs)",
            "",
            "CONSEQUENCE FOR THIS EDA:",
        ]
        if not self.train_csv:
            lines += [
                "  - Session-2 TRAIN-side composition (123,582 images; 42 methods;",
                "    9 real sources; per-method pools; identity counts) CANNOT be",
                "    recomputed here: the raw CSV/images are not in this repository.",
                "    Those figures are shown as REFERENCE (from the PDF) and are",
                "    clearly labelled. They are NOT presented as measured.",
            ]
        if self.eval_preds:
            lines += [
                "  - Session-2 EVALUATION side IS fully reproducible from the local",
                "    paired predictions: balanced-test composition, per-method and",
                "    per-source detection rates, A0-vs-A1 McNemar, and the",
                "    pretrained-probe vs finetuned comparison are all MEASURED.",
            ]
        return "\n".join(lines)


def availability() -> Availability:
    return Availability(
        train_csv=S2_TRAIN_CSV.exists(),
        test_suite=S2_TEST_SUITE.exists(),
        eval_preds=CW_DIR.exists() and any(CW_DIR.glob("*.npz")),
        summary=SUMMARY_JSON.exists(),
        pretrained_summary=PRETRAINED_SUMMARY_JSON.exists(),
        scripts=SCRIPT_FINETUNE.exists() and SCRIPT_PREP.exists(),
    )


# ---------------------------------------------------------------------------
# Reference (PDF) values -- used ONLY where local data is absent
# ---------------------------------------------------------------------------
REFERENCE = {
    "train_total": 123_582,
    "train_fake": 94_025,
    "train_real": 29_557,
    "n_fake_methods": 42,
    "n_real_sources": 9,
    "val_total": 6_302,
    "test_full": 50_084,
    "test_balanced": 21_446,
    "identity": {
        "fake": {"images": 94_025, "identities": 38_331, "per_identity": 2.5},
        "real": {"images": 29_557, "identities": 8_702, "per_identity": 3.4},
        "session1_per_identity": 32,
    },
    "real_sources": {
        "FaceForensics++ Real": 12_515, "Celeb-DF Real": 5_739,
        "ffhq_real (kaggle)": 4_655, "celebvhq_real": 3_800,
        "CollabDiff Real": 690, "whichfaceisreal Real": 683,
        "MidJourney Real": 669, "DF40 Real": 457, "starganv2 Real": 349,
    },
    "method_pools": {
        "faceswap": 11_953, "deepfake_faceswap": 7_672, "facedancer": 4_386,
        "sadtalker": 4_383, "sfhq_studio": 4_156, "simswap": 3_407,
        "blendface": 3_402, "fsgan": 3_401, "inswap": 2_932,
        "wav2lip": 2_927, "lia": 2_465, "mobileswap": 1_980,
    },
    "dimensions": {
        "pipeline_target": "256x256",
        "real FF++ / Celeb-DF": "256x256",
        "real ffhq / celebvhq": "512x512",
        "fake (most methods)": "256x256",
        "fake deepfake_faceswap": "178x218 (non-square)",
        "fake sfhq_studio": "512x512",
    },
    # A0/A1 per-image exposure per epoch, as reported in the reference.
    "exposure": {
        "deepfake_faceswap": (0.056, 0.52), "wav2lip": (0.148, 0.52),
        "sadtalker": (0.099, 0.52), "facedancer": (0.099, 0.52),
        "fsgan": (0.127, 0.52), "inswap": (0.148, 0.52),
        "mobileswap": (0.219, 0.52), "faceswap": (1.73, 1.03),
        "real": (0.70, 0.70),
    },
}


# ---------------------------------------------------------------------------
# Predictions loading (MEASURED)
# ---------------------------------------------------------------------------
def load_preds(model: str) -> dict:
    """Load one model's paired predictions on the balanced Session-2 test."""
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; choose from {list(MODELS)}")
    path = CW_DIR / MODELS[model]
    if not path.exists():
        raise FileNotFoundError(f"predictions not found: {path}")
    d = np.load(path, allow_pickle=True)
    return {
        "preds": d["preds"].astype(int),
        "probs": d["probs"].astype(float),
        "labels": d["labels"].astype(int),
        "methods": d["methods"].astype(str),
        "sources": d["sources"].astype(str),
    }


def available_models() -> list[str]:
    return [m for m, f in MODELS.items() if (CW_DIR / f).exists()]


def load_preds_frame(model: str) -> pd.DataFrame:
    """Predictions as a tidy DataFrame with family and error-type columns."""
    d = load_preds(model)
    df = pd.DataFrame({
        "label": d["labels"], "pred": d["preds"], "prob": d["probs"],
        "method": d["methods"], "source": d["sources"],
    })
    df["model"] = model
    df["correct"] = (df["label"] == df["pred"]).astype(int)
    df["family"] = df["method"].map(family_of)
    df["in_weak_family"] = df["method"].isin(WEAK_FAMILY)
    # FP = REAL predicted FAKE; FN = FAKE predicted REAL
    df["error_type"] = "CORRECT"
    df.loc[(df["label"] == 0) & (df["pred"] == 1), "error_type"] = "FALSE_POSITIVE"
    df.loc[(df["label"] == 1) & (df["pred"] == 0), "error_type"] = "FALSE_NEGATIVE"
    return df


def test_composition() -> dict:
    """MEASURED composition of the balanced Session-2 test set."""
    df = load_preds_frame(available_models()[0])
    real, fake = df[df.label == 0], df[df.label == 1]
    return {
        "n_total": len(df),
        "n_real": len(real),
        "n_fake": len(fake),
        "real_fake_ratio": len(fake) / max(1, len(real)),
        "real_by_source": Counter(real["source"]).most_common(),
        "n_fake_methods": fake["method"].nunique(),
        "fake_by_method": Counter(fake["method"]).most_common(),
        "fake_by_family": Counter(fake["family"]).most_common(),
    }


# ---------------------------------------------------------------------------
# Metrics (MEASURED)
# ---------------------------------------------------------------------------
def metrics(model: str) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    df = load_preds_frame(model)
    y, p, pr = df["label"].values, df["pred"].values, df["prob"].values
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tp = int(((y == 1) & (p == 1)).sum())
    return {
        "model": model,
        "acc%": 100 * accuracy_score(y, p),
        "prec": precision_score(y, p, zero_division=0),
        "rec": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "AUC": roc_auc_score(y, pr),
        "real_acc": tn / max(1, tn + fp),
        "fake_recall": tp / max(1, tp + fn),
        "FP": fp, "FN": fn, "TN": tn, "TP": tp,
    }


def metrics_table(models: list[str] | None = None) -> pd.DataFrame:
    models = models or available_models()
    return pd.DataFrame([metrics(m) for m in models])


# ---------------------------------------------------------------------------
# Per-method / per-source detection (MEASURED)
# ---------------------------------------------------------------------------
def per_method_detection(models: list[str] | None = None) -> pd.DataFrame:
    """Per-method detection rate (fake recall) for each model."""
    models = models or available_models()
    out = []
    for m in models:
        df = load_preds_frame(m)
        fake = df[df.label == 1]
        for method, d in fake.groupby("method"):
            out.append({"model": m, "method": method, "family": family_of(method),
                        "n": len(d), "det_rate": d["correct"].mean(),
                        "in_weak_family": method in WEAK_FAMILY})
    return pd.DataFrame(out)


def per_source_real_accuracy(models: list[str] | None = None) -> pd.DataFrame:
    """Per-real-source accuracy (1 - FP rate) for each model."""
    models = models or available_models()
    out = []
    for m in models:
        df = load_preds_frame(m)
        real = df[df.label == 0]
        for src, d in real.groupby("source"):
            out.append({"model": m, "source": src, "n": len(d),
                        "real_acc": d["correct"].mean(),
                        "FP": int((d["pred"] == 1).sum())})
    return pd.DataFrame(out)


def per_family_detection(models: list[str] | None = None) -> pd.DataFrame:
    models = models or available_models()
    out = []
    for m in models:
        df = load_preds_frame(m)
        fake = df[df.label == 1]
        for fam, d in fake.groupby("family"):
            out.append({"model": m, "family": fam, "n": len(d),
                        "det_rate": d["correct"].mean()})
        w = fake[fake["in_weak_family"]]
        if len(w):
            out.append({"model": m, "family": "* WEAK FAMILY (8)", "n": len(w),
                        "det_rate": w["correct"].mean()})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# McNemar paired test (MEASURED)
# ---------------------------------------------------------------------------
def mcnemar(model_a: str = A0, model_b: str = A1, subset: str = "all") -> dict:
    """Paired McNemar test between two models on the same test images.

    subset: "all" | "fake" | "real" | a method name | "weak_family"
    """
    from scipy.stats import chi2

    da, db = load_preds_frame(model_a), load_preds_frame(model_b)
    if not (da[["label", "method", "source"]].equals(db[["label", "method", "source"]])):
        raise ValueError("prediction files are not aligned on the same test population")

    mask = np.ones(len(da), dtype=bool)
    if subset == "fake":
        mask = (da["label"] == 1).values
    elif subset == "real":
        mask = (da["label"] == 0).values
    elif subset == "weak_family":
        mask = ((da["label"] == 1) & da["in_weak_family"]).values
    elif subset != "all":
        mask = (da["method"] == subset).values

    ca, cb = da["correct"].values[mask], db["correct"].values[mask]
    b = int(((ca == 1) & (cb == 0)).sum())   # a right, b wrong
    c = int(((ca == 0) & (cb == 1)).sum())   # a wrong, b right
    n = b + c
    if n == 0:
        return {"subset": subset, "n": int(mask.sum()), "a_only_correct": b,
                "b_only_correct": c, "chi2": float("nan"), "p": float("nan"),
                "significant": False, "favours": "tie"}
    # Continuity-corrected McNemar
    stat = (abs(b - c) - 1) ** 2 / n
    p = float(chi2.sf(stat, 1))
    return {
        "subset": subset, "n": int(mask.sum()),
        "a_only_correct": b, "b_only_correct": c,
        "chi2": float(stat), "p": p, "significant": p < 0.05,
        "favours": model_b if c > b else (model_a if b > c else "tie"),
    }


def mcnemar_per_method(model_a: str = A0, model_b: str = A1,
                       min_n: int = 50) -> pd.DataFrame:
    """Per-method paired comparison, with a minimum-support guard."""
    da = load_preds_frame(model_a)
    rows = []
    for method, d in da[da.label == 1].groupby("method"):
        if len(d) < min_n:
            continue
        r = mcnemar(model_a, model_b, subset=method)
        r["method"] = method
        r["in_weak_family"] = method in WEAK_FAMILY
        r["delta_correct"] = r["b_only_correct"] - r["a_only_correct"]
        rows.append(r)
    df = pd.DataFrame(rows)
    return df.sort_values("delta_correct", ascending=False) if len(df) else df


# ---------------------------------------------------------------------------
# Sampler exposure (computed with the REAL sampler algorithm)
# ---------------------------------------------------------------------------
def parse_family_weights() -> dict:
    """Read FAMILY_WEIGHTS straight out of finetune_plus_v3.py (no duplication)."""
    if not SCRIPT_FINETUNE.exists():
        return {}
    txt = SCRIPT_FINETUNE.read_text()
    start = txt.find("FAMILY_WEIGHTS = {")
    if start == -1:
        return {}
    end = txt.find("}", start)
    body = txt[start + len("FAMILY_WEIGHTS = {"):end]
    out = {}
    for part in body.split(","):
        if ":" not in part:
            continue
        k, v = part.split(":")
        k = k.strip().strip('"').strip("'")
        try:
            out[k] = float(v.strip())
        except ValueError:
            continue
    return out


def _largest_remainder(slots: int, weights: list[float]) -> list[int]:
    """Identical allocation rule to the project sampler."""
    if not weights or sum(weights) == 0:
        return [0] * len(weights)
    total = sum(weights)
    raw = [slots * w / total for w in weights]
    out = [int(x) for x in raw]
    rem = slots - sum(out)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
    for i in order[:rem]:
        out[i] += 1
    return out


def exposure_table(method_pools: dict[str, int] | None = None,
                   n_real: int | None = None,
                   n_fake_methods: int | None = None) -> pd.DataFrame:
    """Per-image exposure per epoch under sampler A0 vs A1.

    Recomputed with the project's own allocation rules
    (FaceswapFocusedSampler / WeakFamilyBoostedSampler), so the numbers are
    derived rather than copied. Pool sizes default to the reference values
    because the Session-2 train CSV is not present locally.

    ``n_fake_methods`` matters for A0: that sampler splits its "other" bucket
    EVENLY ACROSS METHODS, so the denominator must be the true total method
    count (42), not just the subset of pools passed in.
    """
    pools = dict(method_pools or REFERENCE["method_pools"])
    n_real = n_real or REFERENCE["train_real"]
    n_methods = n_fake_methods or REFERENCE["n_fake_methods"]
    fam_w = parse_family_weights() or {m: (2.0 if m == "faceswap" else 1.0)
                                       for m in WEAK_FAMILY}

    total = 2 * n_real  # both samplers emit 2 x num_real slots per epoch

    # ---- A0: FaceswapFocusedSampler -> P(faceswap)=.35 P(real)=.35 other=.30
    n_fs = int(round(0.35 * total))
    n_re = int(round(0.35 * total))
    n_other0 = total - n_fs - n_re
    # Even split across every non-faceswap method in the FULL dataset.
    n_others_total = max(1, n_methods - 1)
    per_method_slots = n_other0 / n_others_total
    a0 = {}
    for m, p in pools.items():
        slots = n_fs if m == "faceswap" else per_method_slots
        a0[m] = slots / max(1, p)

    # ---- A1: WeakFamilyBoostedSampler -> real .35 / boost .45 / other .20
    n_real_s = int(round(0.35 * total))
    n_boost = int(round(0.45 * total))
    n_other1 = total - n_real_s - n_boost
    fam_methods = [m for m in pools if m in fam_w]
    fam_weights = [pools[m] * fam_w[m] for m in fam_methods]
    fam_counts = dict(zip(fam_methods, _largest_remainder(n_boost, fam_weights)))
    # A1's "other" bucket is POOL-PROPORTIONAL, so every non-family image gets
    # the same exposure = n_other / (total non-family pool). That denominator is
    # the full dataset's non-family pool, not just the pools passed in.
    fam_pool_total = sum(pools[m] for m in fam_methods)
    other_pool_total = max(1, REFERENCE["train_fake"] - fam_pool_total)
    other_exposure = n_other1 / other_pool_total
    a1 = {}
    for m, p in pools.items():
        if m in fam_counts:
            a1[m] = fam_counts[m] / max(1, p)
        else:
            a1[m] = other_exposure

    rows = []
    for m, p in sorted(pools.items(), key=lambda kv: -kv[1]):
        rows.append({"method": m, "pool": p, "A0_exposure": a0[m],
                     "A1_exposure": a1[m],
                     "change_x": a1[m] / a0[m] if a0[m] else float("nan"),
                     "in_weak_family": m in WEAK_FAMILY})
    rows.append({"method": "real", "pool": n_real,
                 "A0_exposure": n_re / n_real, "A1_exposure": n_real_s / n_real,
                 "change_x": 1.0, "in_weak_family": False})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Teammate summaries (MEASURED, read-only)
# ---------------------------------------------------------------------------
def finetune_summary() -> dict:
    return json.load(open(SUMMARY_JSON)) if SUMMARY_JSON.exists() else {}


def pretrained_summary() -> dict:
    return json.load(open(PRETRAINED_SUMMARY_JSON)) if PRETRAINED_SUMMARY_JSON.exists() else {}


def pretrained_vs_finetuned() -> pd.DataFrame:
    """Frozen linear probe vs finetuned, same test and same procedure."""
    order = ["Pretr ViT-Plus (frozen probe)", "Pretr ConvNeXt (frozen probe)",
             A0, A1, "ConvNeXt (finetuned ref)"]
    models = [m for m in order if m in available_models()]
    t = metrics_table(models)
    t["stage"] = np.where(t["model"].str.startswith("Pretr"), "frozen probe", "finetuned")
    return t


def describe() -> str:
    a = availability()
    return a.report()


def consistency_banner() -> str:
    """Identical Session-2 fact sheet for every notebook addendum.

    Values come from :data:`REFERENCE` (PDF) and, when present, from the
    local paired ``.npz`` prediction artifacts. Nothing is hard-coded in
    the notebooks themselves, so all five notebooks print the same facts.
    """
    ref = REFERENCE
    a = availability()
    art = CW_DIR.relative_to(PROJECT_ROOT)
    lines = [
        "=" * 72,
        "Session 2 Analysis Consistency",
        "=" * 72,
        "Session-2 dataset identity : finetune_plus "
        "(Session-2 finetune corpus / session2_finetune_data_eda.pdf)",
        f"  total images (train)   : {ref['train_total']:,}  "
        f"({ref['train_real']:,} real / {ref['train_fake']:,} fake)  [REFERENCE]",
        f"  fake methods           : {ref['n_fake_methods']}  "
        f"| real sources: {ref['n_real_sources']}  [REFERENCE]",
        "  canonical test composition [REFERENCE]:",
        f"    val={ref['val_total']:,}  "
        f"test_full={ref['test_full']:,}  "
        f"test_balanced={ref['test_balanced']:,} (1:1)",
    ]
    if a.eval_preds:
        c = test_composition()
        lines += [
            f"  balanced test MEASURED : n={c['n_total']:,}  "
            f"real={c['n_real']:,}  fake={c['n_fake']:,}  "
            f"(1:{c['real_fake_ratio']:.2f})",
            f"  analysis/prediction artifacts : {art}/*.npz  [AVAILABLE]",
        ]
    else:
        lines.append(
            f"  analysis/prediction artifacts : {art}/*.npz  [ABSENT]"
        )
    lines += [
        "",
        "Protocol distinction (never mix corpora):",
        "  - Session-2 analysis uses the Session-2 balanced test / .npz preds above.",
        "  - Local active protocol is identity_clean_v1 "
        "(separate corpus; ~4.5k imbalanced test).",
        "=" * 72,
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(describe())
    print()
    print(consistency_banner())
    print()
    if availability().eval_preds:
        c = test_composition()
        print(f"Balanced test: n={c['n_total']:,} real={c['n_real']:,} "
              f"fake={c['n_fake']:,} (1:{c['real_fake_ratio']:.2f})")
        print(f"real sources: {c['real_by_source']}")
        print(f"fake methods: {c['n_fake_methods']}")
        print()
        print(metrics_table().to_string(index=False))

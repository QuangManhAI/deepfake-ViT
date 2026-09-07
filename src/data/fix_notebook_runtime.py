"""Targeted runtime fixes for the project notebooks.

Each fix repairs a real dependency (missing checkpoint, stale variable name,
training-only artifact) rather than masking the symptom. No dummy dataframes,
no fabricated metrics, no re-splitting.

Run: .venv/bin/python src/data/fix_notebook_runtime.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"


def load(name: str) -> dict:
    return json.load(open(NB_DIR / name))


def save(name: str, nb: dict) -> None:
    with open(NB_DIR / name, "w") as f:
        json.dump(nb, f, indent=1)


def src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def find_cell(nb: dict, needle: str, start: int = 0) -> int | None:
    for i in range(start, len(nb["cells"])):
        if needle in src(nb["cells"][i]):
            return i
    return None


# ---------------------------------------------------------------------------
# 02_error_analysis: predictions must come from a real model on the canonical
# test protocol. The LoRA checkpoint is absent, so fall back to the canonical
# baseline evaluation (same 4,499 paths), reordered to the notebook's order.
# ---------------------------------------------------------------------------
CANONICAL_FALLBACK = '''
if not cache_is_compatible:
    if CHECKPOINT.exists():
        ck = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
        lc = ck["lora_config"]
        backbone = DinoViT(img_size=IMG_SIZE, embed_dim=384, depth=12, num_heads=6,
                           mlp_ratio=4.0, num_registers=4, gated_mlp=True)
        n_wrap = apply_lora(backbone, r=lc["r"], alpha=lc["alpha"], targets=lc["targets"])
        model = BackboneClassifier(backbone)
        model.load_state_dict(ck["state_dict"])
        model.eval().to(device)

        ds = CsvImageDataset(OUT / "test_balanced.csv", transform=EVAL_TF)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
        preds, probs, labels = [], [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Predicting"):
                p = torch.softmax(model(x.to(device)), dim=1)
                preds.extend(p.argmax(1).cpu().tolist())
                probs.extend(p[:, 1].cpu().tolist())
                labels.extend(y.tolist())
        preds = np.asarray(preds); probs = np.asarray(probs); labels = np.asarray(labels)
        np.savez_compressed(predictions_path, preds=preds, probs=probs, labels=labels)
        MODEL_LABEL = "LoRA fine-tuned DINOv3 ViT-S/16 Plus"
        print(f"Saved canonical predictions: {len(labels):,} rows; LoRA projections wrapped={n_wrap}.")
    else:
        # The LoRA checkpoint is not in this repository. Rather than fabricate
        # predictions, reuse the committed canonical baseline evaluation, which
        # was produced by a real model on EXACTLY this test protocol.
        import pandas as _pd

        baseline_csv = (PROJECT_ROOT / "experiments/results/baseline/final"
                        / "evaluation/test_predictions.csv")
        if not baseline_csv.exists():
            raise FileNotFoundError(
                "No prediction source available for the canonical test protocol.\\n"
                f"  LoRA checkpoint missing : {CHECKPOINT}\\n"
                f"  baseline predictions    : {baseline_csv}\\n"
                "Run src/training/evaluate_baseline.py to produce the latter."
            )
        _bp = _pd.read_csv(baseline_csv)
        _bp["path"] = _bp["path"].astype(str)

        # Align the baseline rows to this notebook's (shuffled) canonical order.
        _order = [p for p, *_ in test_rows]
        _missing = set(_order) - set(_bp["path"])
        if _missing:
            raise ValueError(
                f"baseline predictions do not cover the canonical protocol: "
                f"{len(_missing)} paths absent"
            )
        _bp = _bp.set_index("path").reindex(_order)

        labels = _bp["label"].to_numpy(dtype=int)
        preds = _bp["predicted_label"].to_numpy(dtype=int)
        probs = _bp["probability_fake"].to_numpy(dtype=float)
        MODEL_LABEL = "DINOv3 ViT-S/16 baseline (class-weighted, MCC-selected)"

        # Verify the reindex preserved label agreement with the protocol.
        _proto_labels = np.asarray([lab for _, lab, *_ in test_rows], dtype=int)
        if not (labels == _proto_labels).all():
            raise ValueError("label mismatch after aligning baseline predictions")

        np.savez_compressed(predictions_path, preds=preds, probs=probs, labels=labels)
        print("LoRA checkpoint absent -> using the canonical baseline evaluation.")
        print(f"  source: {baseline_csv.relative_to(PROJECT_ROOT)}")
        print(f"  model : {MODEL_LABEL}")
        print(f"  rows  : {len(labels):,} (aligned to the canonical test protocol)")
else:
    MODEL_LABEL = "cached predictions"

if not all(len(value) == expected_rows for value in (preds, probs, labels)):
    raise ValueError("Prediction arrays do not match the canonical metadata row count.")
print(f"Test samples: {len(labels):,} | real: {(labels == 0).sum():,} | fake: {(labels == 1).sum():,}")
'''.strip()


def fix_02_error_analysis() -> list[str]:
    name = "02_error_analysis.ipynb"
    nb = load(name)
    notes = []

    i = find_cell(nb, "predictions_path = OUT / \"predictions.npz\"")
    if i is None:
        return ["02_error_analysis: prediction cell not found"]

    text = src(nb["cells"][i])
    marker = "if not cache_is_compatible:"
    head = text[: text.index(marker)]
    set_src(nb["cells"][i], head + CANONICAL_FALLBACK + "\n")
    notes.append(f"cell {i}: added canonical-baseline prediction fallback "
                 "(LoRA checkpoint absent)")

    # Plot titles must name the model actually plotted.
    replaced = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        t = src(c)
        if "LoRA fine-tuned DINOv3 ViT-S/16 Plus (canonical test protocol)" in t:
            t = t.replace(
                '"Confusion Matrix - LoRA fine-tuned DINOv3 ViT-S/16 Plus (canonical test protocol)"',
                'f"Confusion Matrix - {MODEL_LABEL} (canonical test protocol)"')
            set_src(c, t)
            replaced += 1
    if replaced:
        notes.append(f"{replaced} plot title(s) now use MODEL_LABEL so labels match the data")

    save(name, nb)
    return notes


# ---------------------------------------------------------------------------
# 02_training_balanced_dataset: cells after the (skipped) training loop depend
# on training products. Guard them so they report the missing artifact instead
# of raising NameError, WITHOUT inventing any values.
# ---------------------------------------------------------------------------
GUARD_TEMPLATE = '''# --- Requires a completed training run -------------------------------------
# Training is not executed in this audit, so the artifacts below are absent.
# Nothing is fabricated: the section reports what is missing and stops.
_needed = {needed!r}
_missing = [n for n in _needed if n not in dir()]
if _missing:
    print("SECTION SKIPPED - requires a completed training run.")
    print(f"  missing in-memory artifacts: {{_missing}}")
    print("  Run src/training/train.py (or the training cell above), then re-run.")
else:
{body}
'''


def guard_cell(cell: dict, needed: list[str]) -> None:
    body = src(cell).rstrip("\n")
    indented = "\n".join("    " + l if l.strip() else l for l in body.split("\n"))
    set_src(cell, GUARD_TEMPLATE.format(needed=needed, body=indented))


def fix_02_training() -> list[str]:
    name = "02_training_balanced_dataset.ipynb"
    nb = load(name)
    notes = []

    # (needle identifying the cell, names it depends on)
    targets = [
        ("fig, axes = plt.subplots(2, 2, figsize=(14, 10))", ["history"]),
        ("# Load the BEST Checkpoint from the current training run", ["GLOBAL_BEST_CKPT"]),
        ("fig, axes = plt.subplots(2, 2, figsize=(14, 11))", ["y_true", "y_prob"]),
        ("# Consolidate experiment metadata and export to JSON", ["TIMESTAMP_STR"]),
    ]
    for needle, needed in targets:
        i = find_cell(nb, needle)
        if i is None or "Requires a completed training run" in src(nb["cells"][i]):
            continue
        guard_cell(nb["cells"][i], needed)
        notes.append(f"cell {i}: guarded on {needed} (training artifact)")

    # Cells reading a predictions CSV that only training/eval produces.
    for needle in ["# Per-Method Detection Accuracy on DF40 Benchmark",
                   "# Load predictions and inspect error cases"]:
        i = find_cell(nb, needle)
        if i is None:
            continue
        t = src(nb["cells"][i])
        if "PRED_CSV_AVAILABLE" in t:
            continue
        guard = (
            "# --- Requires the test-prediction CSV produced after training -----------\n"
            "_pred_csv = RESULTS_DIR / \"test_balanced_predictions.csv\"\n"
            "PRED_CSV_AVAILABLE = _pred_csv.exists()\n"
            "if not PRED_CSV_AVAILABLE:\n"
            "    print(\"SECTION SKIPPED - prediction CSV not present.\")\n"
            "    print(f\"  expected: {_pred_csv}\")\n"
            "    print(\"  It is produced by evaluating a trained checkpoint.\")\n"
            "else:\n"
        )
        indented = "\n".join("    " + l if l.strip() else l
                             for l in t.rstrip("\n").split("\n"))
        set_src(nb["cells"][i], guard + indented + "\n")
        notes.append(f"cell {i}: guarded on test_balanced_predictions.csv")

    save(name, nb)
    return notes


# ---------------------------------------------------------------------------
# courseWorkCheck: legacy cells reference an external RunPod benchmark. Fix the
# real problems: a non-ASCII arrow inside code, and prediction arrays that were
# never produced locally.
# ---------------------------------------------------------------------------
def fix_courseworkcheck() -> list[str]:
    name = "courseWorkCheck.ipynb"
    nb = load(name)
    notes = []

    # Real bug: `print(""""""` is six quotes, so the triple-quoted string opens
    # AND closes immediately, turning the following prose into bare code and
    # raising SyntaxError on the first non-ASCII character. It should be `print("""`.
    # The arrows themselves are fine -- they live inside the string literal.
    for idx, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        t = src(c)
        if 'print("""""")' in t or 'print(""""""\n' in t:
            fixed = t.replace('print(""""""', 'print("""')
            try:
                compile(fixed, f"cell{idx}", "exec")
            except SyntaxError:
                continue
            set_src(c, fixed)
            notes.append(f'cell {idx}: fixed `print("""""" ` -> `print("""` '
                         "(6 quotes closed the string early; real SyntaxError)")

    # Guard the legacy evaluation cells that need locally-absent predictions.
    legacy_needles = [
        "# ---- confusion matrix ----",
        "# ---- ROC + Precision-Recall ----",
        "# ---- per-method accuracy: v3 vs baseline",
        "# ---- chi tiết faceswap",
    ]
    for needle in legacy_needles:
        i = find_cell(nb, needle)
        if i is None:
            continue
        t = src(nb["cells"][i])
        if "LEGACY SECTION" in t:
            continue
        guard = (
            "# --- LEGACY SECTION (external RunPod benchmark) -------------------------\n"
            "# These cells were authored against /workspace/... predictions that are\n"
            "# not part of this repository. They are preserved verbatim below but are\n"
            "# skipped when their inputs are absent. Nothing is fabricated.\n"
            "_legacy_needed = [n for n in ('labels', 'probs', 'preds') if n not in dir()]\n"
            "if _legacy_needed:\n"
            "    print('LEGACY SECTION SKIPPED - external benchmark predictions absent:',\n"
            "          _legacy_needed)\n"
            "    print('  See the SESSION-2 addendum below for the canonical, reproducible\\n'\n"
            "          '  evaluation on the current protocol.')\n"
            "else:\n"
        )
        indented = "\n".join("    " + l if l.strip() else l
                             for l in t.rstrip("\n").split("\n"))
        set_src(nb["cells"][i], guard + indented + "\n")
        notes.append(f"cell {i}: guarded legacy external-benchmark section")

    save(name, nb)
    return notes


def main():
    for label, fn in [("02_error_analysis", fix_02_error_analysis),
                      ("02_training_balanced_dataset", fix_02_training),
                      ("courseWorkCheck", fix_courseworkcheck)]:
        print(f"--- {label} ---")
        notes = fn()
        for n in notes or ["(no change needed)"]:
            print(f"    {n}")


if __name__ == "__main__":
    main()

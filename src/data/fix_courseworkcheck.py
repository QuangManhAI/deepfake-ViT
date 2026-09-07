"""Make courseWorkCheck.ipynb executable from the repository root.

The original notebook was authored on another machine and hardcodes
``/workspace/quangmanh/...`` and ``/workspace/hoangtuan/...``. Locally that
causes ``OSError: Read-only file system: '/workspace'`` in the setup cell and
cascading NameErrors everywhere after it.

What this does:
  * Rewrites ONLY the path block of the setup cell so every path resolves from
    the repository root (no /workspace, no writes outside the repo).
  * Points the backbone weights at the real local file.
  * Sets a LEGACY_DATA_AVAILABLE flag by checking whether the legacy inputs
    actually exist -- it never creates them.
  * Wraps each legacy cell so it skips with an explicit message when its inputs
    are absent, instead of raising. Original code is preserved verbatim.

The reproducible verification for the current protocol lives in the appended
SESSION-2 addendum, which is untouched.

Run: .venv/bin/python src/data/fix_courseworkcheck.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NB = PROJECT_ROOT / "notebooks" / "courseWorkCheck.ipynb"
SENTINEL = "SESSION-2 EDA ADDENDUM"
GUARD_FLAG = "LEGACY_DATA_AVAILABLE"

NEW_PATH_BLOCK = '''# ---------- paths (resolved from the repository root) ----------
# The original notebook hardcoded /workspace/quangmanh and /workspace/hoangtuan
# from another machine. Those roots do not exist here and are read-only, so all
# paths below are resolved relative to this repository instead.
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = PROJECT_ROOT
HT = PROJECT_ROOT
TEST_CSV = PROJECT_ROOT / "data/splits/test_balanced_fixed_zero_leakage.csv"
TRAIN_CSV = HT / "data/splits/train_v5_weakfix_v3.csv"
VAL_CSV = HT / "data/splits/val_v5_combined_universal_kaggle_boost.csv"
V3_CKPT = HT / "experiments/checkpoints/exp05_v5_weakfix_v3/best_model.pt"
# Local DINOv3 backbone actually present in this repository:
PRETRAINED = HT / "experiments/checkpoints/weights/model.safetensors"
DS_V2 = HT / "experiments/results/v5_weakfix_dataset_summary.json"
DS_V3 = HT / "experiments/results/v5_weakfix_v3_dataset_summary.json"
REPORT = HT / "experiments/results/v5_weakfix_v3_training_report.json"
PM_BASE = HT / "experiments/results/v5_combined_per_method_accuracy.csv"
PM_V2 = HT / "experiments/results/v5_weakfix_per_method_accuracy.csv"
PM_V3 = HT / "experiments/results/v5_weakfix_v3_per_method_accuracy.csv"

OUT = ROOT / "experiments/results/courseWorkCheck"
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")
print("DEVICE:", DEVICE, "| torch", torch.__version__, "| numpy", np.__version__)


def load_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


# The v5_weakfix corpus and the zero-leakage benchmark are NOT part of this
# repository. Detect that rather than fabricating the files.
_legacy_inputs = {"TRAIN_CSV": TRAIN_CSV, "TEST_CSV": TEST_CSV}
_absent = {k: str(v) for k, v in _legacy_inputs.items() if not v.exists()}
LEGACY_DATA_AVAILABLE = not _absent

if LEGACY_DATA_AVAILABLE:
    train = load_csv(TRAIN_CSV)
    test = load_csv(TEST_CSV)
    print(f"TRAIN = {len(train):,} anh | TEST = {len(test):,} anh")
else:
    print("LEGACY DATA NOT AVAILABLE IN THIS REPOSITORY")
    for k, v in _absent.items():
        print(f"   missing {k}: {v}")
    print("   -> the legacy v5_weakfix sections below will be skipped.")
    print("   -> nothing is fabricated; see the SESSION-2 addendum at the end")
    print("      for the reproducible evaluation on the current canonical protocol.")
print("Output dir:", OUT)'''


def src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main():
    nb = json.load(open(NB))
    cells = nb["cells"]
    notes = []

    addendum_at = next((i for i, c in enumerate(cells)
                        if c["cell_type"] == "markdown" and SENTINEL in src(c)),
                       len(cells))

    # ---- 1. setup cell: replace only the path block -----------------------
    setup_i = next(i for i in range(addendum_at)
                   if cells[i]["cell_type"] == "code" and "ROOT = Path(" in src(cells[i]))
    text = src(cells[setup_i])
    head = text[: text.index("# ---------- paths ----------")]
    # keep a non-interactive backend so plt.show() cannot fail headless
    head = head.replace('matplotlib.use("module://matplotlib_inline.backend_inline")',
                        'matplotlib.use("Agg")')
    set_src(cells[setup_i], head + NEW_PATH_BLOCK + "\n")
    notes.append(f"cell {setup_i}: paths resolved from repo root; /workspace removed; "
                 "Agg backend; LEGACY_DATA_AVAILABLE flag added")

    # ---- 2. guard every later legacy code cell ---------------------------
    guarded = 0
    for i in range(setup_i + 1, addendum_at):
        c = cells[i]
        if c["cell_type"] != "code":
            continue
        t = src(c)
        if not t.strip() or GUARD_FLAG in t:
            continue
        indented = "\n".join("    " + l if l.strip() else l
                             for l in t.rstrip("\n").split("\n"))
        guard = (
            "# --- LEGACY SECTION (v5_weakfix corpus / external benchmark) -----------\n"
            "# Preserved verbatim. Runs only when the legacy inputs are present.\n"
            f"if not {GUARD_FLAG}:\n"
            "    print('LEGACY SECTION SKIPPED - required legacy inputs are not in this repo.')\n"
            "else:\n"
        )
        set_src(c, guard + indented + "\n")
        guarded += 1

    notes.append(f"guarded {guarded} legacy cells on {GUARD_FLAG}")

    with open(NB, "w") as f:
        json.dump(nb, f, indent=1)

    for n in notes:
        print(f"    {n}")


if __name__ == "__main__":
    main()

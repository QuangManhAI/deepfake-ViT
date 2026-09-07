"""Make notebook plotting work AND render inline under a headless kernel.

Problem found during the clean-kernel audit: several notebooks did

    matplotlib.use("Agg")
    plt.show = lambda *a, **k: None

That stops plt.show() from raising in a non-interactive environment, but it
also means a full run produces ZERO embedded figures -- the notebooks looked
"clean" while silently generating no visible plots.

Fix: prefer the ipython inline backend (which works under nbclient and embeds
PNGs) and fall back to Agg only when IPython is absent. plt.show() is left as
the real function, so figures are both displayed and still savable.

Run: .venv/bin/python src/data/fix_plotting_backend.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"

NOTEBOOKS = [
    "00_comprehensive_dataset_eda.ipynb",
    "01_full_pipeline.ipynb",
    "02_error_analysis.ipynb",
    "02_training_balanced_dataset.ipynb",
    "courseWorkCheck.ipynb",
]

SHIM = '''import matplotlib
# Prefer the inline backend so figures render in the notebook; fall back to Agg
# when running outside IPython. Either way plt.show() is safe to call.
try:
    import matplotlib_inline  # noqa: F401
    get_ipython()             # raises NameError outside IPython
    matplotlib.use("module://matplotlib_inline.backend_inline")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt'''


def src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def patch_text(t: str) -> tuple[str, list[str]]:
    notes = []

    # 1. Drop the plt.show monkeypatch that suppressed every figure.
    if re.search(r"^\s*plt\.show\s*=\s*lambda", t, flags=re.M):
        t = re.sub(r"^\s*plt\.show\s*=\s*lambda.*\n", "", t, flags=re.M)
        notes.append("removed plt.show monkeypatch")

    # 2. Replace a hard Agg/inline selection with the portable shim.
    pattern = re.compile(
        r"import matplotlib\n"
        r'(?:matplotlib\.use\("(?:Agg|module://matplotlib_inline\.backend_inline)"\)\n)'
        r"import matplotlib\.pyplot as plt"
    )
    if pattern.search(t):
        t = pattern.sub(SHIM, t)
        notes.append("backend selection -> inline-preferring shim")
    return t, notes


def main():
    for name in NOTEBOOKS:
        path = NB_DIR / name
        nb = json.load(open(path))
        all_notes = []
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            t = src(c)
            new, notes = patch_text(t)
            if notes:
                set_src(c, new)
                all_notes.append(f"cell {i}: " + "; ".join(notes))
        if all_notes:
            with open(path, "w") as f:
                json.dump(nb, f, indent=1)
        print(f"{name}:")
        for n in all_notes or ["  (nothing to patch)"]:
            print(f"    {n}")


if __name__ == "__main__":
    main()

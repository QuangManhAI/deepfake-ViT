"""Execute the appended Session-2 addendum cells and report failures.

The ORIGINAL cells in these notebooks were authored against a RunPod
environment (`/workspace/...`) that does not exist here, so they cannot run
locally and are intentionally left untouched. This validator therefore
executes only the appended addendum cells, in order, per notebook, sharing one
namespace exactly as a notebook would.

Run: .venv/bin/python src/data/validate_addendum.py
"""

from __future__ import annotations

import io
import json
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"
SENTINEL = "SESSION-2 EDA ADDENDUM"

NOTEBOOKS = [
    "00_comprehensive_dataset_eda.ipynb",
    "01_full_pipeline.ipynb",
    "02_error_analysis.ipynb",
    "02_training_balanced_dataset.ipynb",
    "courseWorkCheck.ipynb",
]


def addendum_code_cells(nb_path: Path) -> list[str]:
    nb = json.load(open(nb_path))
    cells = nb["cells"]
    start = None
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and SENTINEL in "".join(c.get("source", [])):
            start = i
            break
    if start is None:
        return []
    return ["".join(c["source"]) for c in cells[start:] if c["cell_type"] == "code"]


def run_notebook(nb_path: Path) -> tuple[int, int, list[str]]:
    cells = addendum_code_cells(nb_path)
    # Emulate notebook globals: display() and a cwd of notebooks/
    ns: dict = {"display": lambda *a, **k: None, "__name__": "__main__"}
    failures = []
    ok = 0
    for i, src in enumerate(cells, 1):
        buf = io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                exec(compile(src, f"{nb_path.name}#addendum_cell_{i}", "exec"), ns)
            plt.close("all")
            ok += 1
        except Exception:
            failures.append(f"--- {nb_path.name} addendum cell {i} FAILED ---\n"
                            f"{traceback.format_exc()}\n"
                            f"stdout/stderr:\n{buf.getvalue()[-1500:]}")
    return ok, len(cells), failures


# Canonical facts that every notebook's Session-2 addendum must visibly print.
REQUIRED_BANNER_FACTS = [
    "Session 2 Analysis Consistency",
    "123,582",
    "42",
    "21,446",
    "50,084",
    "coursework_vs",
    "identity_clean_v1",
]


def collect_addendum_text(nb_path: Path) -> str:
    """Concatenate addendum cell sources + stored outputs (for fact checks)."""
    nb = json.load(open(nb_path))
    start = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown" and SENTINEL in "".join(c.get("source", [])):
            start = i
            break
    if start is None:
        return ""
    parts: list[str] = []
    for c in nb["cells"][start:]:
        parts.append("".join(c.get("source", [])))
        for o in c.get("outputs", []):
            if "text" in o:
                parts.append("".join(o["text"]))
            data = o.get("data", {})
            if "text/plain" in data:
                parts.append("".join(data["text/plain"]))
    return "\n".join(parts)


def check_banner_consistency() -> list[str]:
    """Ensure all five notebooks visibly print the same Session-2 facts."""
    fails: list[str] = []
    banners: dict[str, str] = {}
    for name in NOTEBOOKS:
        text = collect_addendum_text(NB_DIR / name)
        missing = [f for f in REQUIRED_BANNER_FACTS if f not in text]
        if missing:
            fails.append(f"{name}: missing consistency facts: {missing}")
        # Capture the banner block itself for cross-notebook equality.
        marker = "Session 2 Analysis Consistency"
        if marker in text:
            start = text.index(marker)
            end = text.find("====", start + len(marker))
            # Include trailing separator line if present.
            if end != -1:
                end2 = text.find("\n", end)
                end = end2 if end2 != -1 else end + 4
            banners[name] = text[start:end].strip()
    if len(banners) == len(NOTEBOOKS):
        first_name, first = next(iter(banners.items()))
        for name, banner in banners.items():
            if banner != first:
                fails.append(
                    f"{name}: Session 2 Analysis Consistency banner differs "
                    f"from {first_name}"
                )
    return fails


def main():
    # Run from notebooks/ so the bootstrap's Path.cwd() logic is exercised.
    import os
    os.chdir(NB_DIR)

    total_ok = total = 0
    all_fail: list[str] = []
    print("Validating appended Session-2 addendum cells\n")
    for name in NOTEBOOKS:
        ok, n, fails = run_notebook(NB_DIR / name)
        total_ok += ok
        total += n
        all_fail.extend(fails)
        status = "PASS" if ok == n else f"{n - ok} FAILED"
        print(f"  {name:44s} {ok}/{n} cells  [{status}]")

    print(f"\nTOTAL: {total_ok}/{total} addendum cells executed successfully")

    print("\nChecking Session 2 Analysis Consistency banner across notebooks...")
    banner_fails = check_banner_consistency()
    if banner_fails:
        all_fail.extend(banner_fails)
        for f in banner_fails:
            print(f"  FAIL: {f}")
    else:
        print("  PASS: all five notebooks print the same canonical Session-2 facts")

    if all_fail:
        print(f"\n{len(all_fail)} failure(s):\n")
        for f in all_fail:
            print(f)
        sys.exit(1)
    print("\nALL ADDENDUM CELLS PASS")


if __name__ == "__main__":
    main()

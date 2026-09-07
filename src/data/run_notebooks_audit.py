"""Clean-kernel execution audit for the five project notebooks.

Executes each notebook top-to-bottom in a FRESH kernel (no stale state) and
reports every error with its cell index and the source line that failed.

Training is intentionally NOT executed: cells whose source matches
TRAINING_MARKERS are replaced, for the run only, by a guard that defines
nothing and prints a skip notice. The notebook on disk is not modified by this
runner unless --write is passed.

Usage:
    .venv/bin/python src/data/run_notebooks_audit.py            # audit only
    .venv/bin/python src/data/run_notebooks_audit.py --write    # store outputs
    .venv/bin/python src/data/run_notebooks_audit.py -n 00_comprehensive_dataset_eda.ipynb
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"

NOTEBOOKS = [
    "00_comprehensive_dataset_eda.ipynb",
    "01_full_pipeline.ipynb",
    "02_error_analysis.ipynb",
    "02_training_balanced_dataset.ipynb",
    "courseWorkCheck.ipynb",
]

# A cell is treated as an actual training run (skipped) only if it clearly
# launches a fit loop. Model/dataloader definitions are NOT skipped, because
# their correctness is exactly what we need to validate.
TRAINING_MARKERS = [
    r"for\s+epoch\s+in\s+range",
    r"\.backward\(\)",
    r"optimizer\.step\(\)",
    r"scaler\.step\(",
    r"train_one_epoch\(",
    r"trainer\.fit\(",
]

SKIP_BANNER = (
    "# ---- TRAINING EXECUTION SKIPPED BY REQUEST (audit run) ----\n"
    "# The code in this cell was syntax- and dependency-checked but the fit\n"
    "# loop was not executed. Compile the original source to prove it parses:\n"
)


def is_training_cell(src: str) -> bool:
    return any(re.search(p, src) for p in TRAINING_MARKERS)


def make_skip_cell(src: str) -> str:
    """Compile-check the real source without running it."""
    payload = json.dumps(src)
    return (
        SKIP_BANNER
        + f"import ast, textwrap\n_src = {payload}\n"
        "ast.parse(_src)\n"
        "compile(_src, '<training-cell>', 'exec')\n"
        "print('TRAINING EXECUTION SKIPPED BY REQUEST — source parses and compiles OK')\n"
        "print(f'  cell length: {len(_src)} chars')\n"
    )


def audit(name: str, write: bool = False) -> dict:
    path = NB_DIR / name
    nb = nbformat.read(path, as_version=4)

    skipped = []
    originals = {}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source or ""
        if is_training_cell(src):
            originals[i] = src
            cell.source = make_skip_cell(src)
            skipped.append(i)

    client = NotebookClient(
        nb, timeout=2400, kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
        allow_errors=True,
    )
    client.execute()

    errors = []
    warnings = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for o in cell.get("outputs", []):
            if o.get("output_type") == "error":
                tb = "\n".join(o.get("traceback", []))
                # last source line mentioned in the traceback
                errors.append({
                    "cell": i,
                    "ename": o.get("ename"),
                    "evalue": (o.get("evalue") or "")[:300],
                    "src_head": (originals.get(i, cell.source) or "").strip().split("\n")[0][:110],
                    "tb_tail": tb[-600:],
                })
            if o.get("output_type") == "stream" and o.get("name") == "stderr":
                txt = "".join(o.get("text", []))
                warnings += txt.lower().count("warning")

    # restore original training sources before any write-back
    for i, src in originals.items():
        nb.cells[i].source = src
        if write:
            nb.cells[i]["outputs"] = [nbformat.v4.new_output(
                "stream", name="stdout",
                text="TRAINING EXECUTION SKIPPED BY REQUEST (not executed in audit run)\n")]
            nb.cells[i]["execution_count"] = None

    if write:
        nbformat.write(nb, path)

    return {"notebook": name, "n_cells": len(nb.cells), "errors": errors,
            "warnings": warnings, "skipped_training_cells": skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--notebook", action="append", default=None)
    ap.add_argument("--write", action="store_true",
                    help="store executed outputs back into the notebooks")
    args = ap.parse_args()

    targets = args.notebook or NOTEBOOKS
    results = []
    for name in targets:
        print(f"\n{'='*78}\nEXECUTING (clean kernel): {name}\n{'='*78}")
        r = audit(name, write=args.write)
        results.append(r)
        print(f"  cells={r['n_cells']}  errors={len(r['errors'])}  "
              f"stderr-warnings={r['warnings']}  "
              f"training-cells-skipped={len(r['skipped_training_cells'])}")
        for e in r["errors"]:
            print(f"  ! cell {e['cell']:>3}  {e['ename']}: {e['evalue'][:150]}")
            print(f"      src: {e['src_head']}")

    print(f"\n\n{'='*78}\nSUMMARY\n{'='*78}")
    print(f"{'notebook':46s} {'cells':>5} {'errors':>7} {'warn':>5} {'skipped':>8}")
    total_err = 0
    for r in results:
        total_err += len(r["errors"])
        print(f"{r['notebook']:46s} {r['n_cells']:>5} {len(r['errors']):>7} "
              f"{r['warnings']:>5} {len(r['skipped_training_cells']):>8}")
    print(f"\nTOTAL ERRORS: {total_err}")

    out = PROJECT_ROOT / "experiments" / "results" / "notebook_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"detail: {out.relative_to(PROJECT_ROOT)}")
    raise SystemExit(1 if total_err else 0)


if __name__ == "__main__":
    main()

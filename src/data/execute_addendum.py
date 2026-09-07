"""Execute ONLY the appended addendum cells and store their real outputs.

The original cells target a RunPod environment that does not exist here, so
executing the whole notebook would fail and would overwrite preserved original
outputs with errors. Instead this builds a temporary notebook containing just
the addendum cells, executes it with a real kernel, and copies the resulting
outputs back onto the matching cells.

Original cells are never executed and never modified.

Run: .venv/bin/python src/data/execute_addendum.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

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


def addendum_start(cells: list[dict]) -> int | None:
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and SENTINEL in "".join(c.get("source", [])):
            return i
    return None


def run(name: str) -> tuple[int, int]:
    path = NB_DIR / name
    raw = json.load(open(path))
    start = addendum_start(raw["cells"])
    if start is None:
        print(f"  {name}: no addendum found, skipped")
        return 0, 0

    addendum = raw["cells"][start:]
    code_idx = [i for i, c in enumerate(addendum) if c["cell_type"] == "code"]

    # Build a temp notebook of just the addendum code cells.
    tmp = nbformat.v4.new_notebook()
    tmp.cells = [nbformat.v4.new_code_cell("".join(addendum[i]["source"]))
                 for i in code_idx]
    tmp.metadata = {"kernelspec": {"display_name": "Python 3",
                                   "language": "python", "name": "python3"}}

    client = NotebookClient(tmp, timeout=1200, kernel_name="python3",
                            resources={"metadata": {"path": str(NB_DIR)}},
                            allow_errors=True)
    client.execute()

    ok = 0
    for slot, cell in zip(code_idx, tmp.cells):
        outs = [json.loads(nbformat.writes(nbformat.from_dict(o))) if False else dict(o)
                for o in cell.get("outputs", [])]
        errored = any(o.get("output_type") == "error" for o in outs)
        addendum[slot]["outputs"] = outs
        addendum[slot]["execution_count"] = cell.get("execution_count")
        if not errored:
            ok += 1
        else:
            err = next(o for o in outs if o.get("output_type") == "error")
            print(f"    ! cell {slot} error: {err.get('ename')}: {err.get('evalue')}")

    raw["cells"] = raw["cells"][:start] + addendum
    with open(path, "w") as f:
        json.dump(raw, f, indent=1)
    return ok, len(code_idx)


def main():
    print("Executing addendum cells with a real kernel (originals untouched)\n")
    t_ok = t_all = 0
    for name in NOTEBOOKS:
        ok, n = run(name)
        t_ok += ok
        t_all += n
        print(f"  {name:44s} {ok}/{n} cells produced clean output")
    print(f"\nTOTAL: {t_ok}/{t_all}")
    if t_ok != t_all:
        raise SystemExit(1)
    print("ALL ADDENDUM CELLS EXECUTED WITH OUTPUTS STORED")


if __name__ == "__main__":
    main()

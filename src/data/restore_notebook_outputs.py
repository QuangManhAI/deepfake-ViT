"""Restore notebook outputs that were stripped by an earlier commit.

Matches cells by exact source text against a reference git revision and copies
back ``outputs`` / ``execution_count``. Cells added after the reference commit
are left untouched.

Run: .venv/bin/python src/data/restore_notebook_outputs.py [--rev fa2f9ac^]
"""

from __future__ import annotations

import argparse
import json
import subprocess
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


def git_show(rev: str, path: str) -> dict | None:
    r = subprocess.run(["git", "show", f"{rev}:{path}"],
                       capture_output=True, text=True, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)


def src_key(cell: dict) -> str:
    return "".join(cell.get("source", []))


def restore(name: str, rev: str) -> tuple[int, int]:
    path = NB_DIR / name
    cur = json.load(open(path))
    ref = git_show(rev, f"notebooks/{name}")
    if ref is None:
        print(f"  {name}: reference revision unavailable, skipped")
        return 0, 0

    # Map source -> (outputs, execution_count) from the reference notebook.
    ref_map: dict[str, tuple] = {}
    for c in ref["cells"]:
        if c["cell_type"] != "code":
            continue
        outs = c.get("outputs", [])
        if outs:
            ref_map[src_key(c)] = (outs, c.get("execution_count"))

    restored = 0
    imgs = 0
    for c in cur["cells"]:
        if c["cell_type"] != "code":
            continue
        k = src_key(c)
        if k not in ref_map:
            continue
        outs, ec = ref_map[k]

        def n_img(os_):
            return sum(1 for o in os_ if "image/png" in str(o.get("data", {})))

        cur_outs = c.get("outputs", [])
        # Restore when the cell has no outputs at all, or when the reference
        # holds richer output (e.g. images) than the current stream-only stub.
        if not cur_outs or n_img(outs) > n_img(cur_outs) or len(outs) > len(cur_outs):
            c["outputs"] = outs
            c["execution_count"] = ec
            restored += 1
            imgs += n_img(outs)

    if restored:
        with open(path, "w") as f:
            json.dump(cur, f, indent=1)
    return restored, imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="fa2f9ac^",
                    help="git revision holding the original outputs")
    args = ap.parse_args()

    print(f"Restoring stripped outputs from {args.rev}")
    total_c = total_i = 0
    for nb in NOTEBOOKS:
        c, i = restore(nb, args.rev)
        total_c += c
        total_i += i
        print(f"  {nb}: restored {c} cell outputs ({i} images)")
    print(f"\nTotal: {total_c} cell outputs, {total_i} images restored")


if __name__ == "__main__":
    main()

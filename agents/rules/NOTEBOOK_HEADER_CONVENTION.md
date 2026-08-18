# NOTEBOOK_HEADER_CONVENTION.md
Rules for the first cell of every notebook. Condensed from
notebook_header_guide.md — see that file for the full worked example.
Cross-reference link requirements come from
[MD_CONVENTION.md](MD_CONVENTION.md#mandatory-cross-reference-links).

Every notebook's first cell = a single markdown cell with 4 sections, in order:

## 1. Title (H1)
`# <Scope> <N>: <Short Description>`

| Scope prefix | Use for |
|---|---|
| `Practice N:` | Standalone practice/learning notebooks |
| `Phase N:` | Foundational / diagnostic experiments |
| `Experiment N:` | Single-variable-change experiments |
| `Appendix N:` | Supplementary / follow-up analysis |

## 2. Subtitle + Purpose (H2 + short paragraph)
One-line H2 subtitle + 1–2 sentences on what the notebook does and why.

For single-variable experiments, use this structured form instead:
```
## Rationale
<why this experiment>

**Single variable changed**: <the one thing>
**Held constant**: <everything else — training config, loss, data pipeline, etc.>
```

## 3. Roadmap Table
Exactly these 4 columns, one row per notebook step:
```
| Step | Description | What it does | Import path |
```
- **Step**: sequential number
- **Description**: short action phrase
- **What it does**: 5–15 words
- **Import path**: `src/...` module or `—` if none
- Table must end with `---` immediately after

## 4. References (mandatory)
A `## References` block with **working cross-reference links** to everything the
notebook consumes or documents. Links are relative to the notebook's own
location (`notebooks/`), so from a notebook the paths start with `../`:

| Reference | Link (relative to `notebooks/`) |
|---|---|
| Rules | `[LOGGING_CHECKPOINT_RULES.md](../agents/rules/LOGGING_CHECKPOINT_RULES.md)`, `[RESULTS_REPORTING.md](../agents/rules/RESULTS_REPORTING.md)` |
| Training / experiment scripts | `src/training/<script>.py`, `src/experiments/<script>.py` (whichever produced the artifacts) |
| Artifact locations | `experiments/runs/`, `experiments/results/<experiment>/`, `experiments/plots/` |
| Related notebooks / docs | phase docs, experiment reports |

Each entry is a relative markdown link written from the notebook's location,
e.g. ``[<feature>_train.py](../src/training/<feature>_train.py)`` —
never bare paths. See [MD_CONVENTION.md](MD_CONVENTION.md#mandatory-cross-reference-links).

## Hard rule
This entire header is ONE markdown cell — the first cell in the notebook.
Section headings in the notebook body come after this block, not inside it.

## Output persistence & cell independence (hard rules)

### 1. Persist every output to its designated folder
Notebooks are **analysis-only**: they load artifacts produced by scripts and
never re-create run state. Any data a notebook itself produces must be written
to its designated folder — never left only in-memory or inside cell outputs:

| Output | Designated folder | Who writes it |
|---|---|---|
| Checkpoints, per-epoch history, config, logs, TensorBoard | `experiments/runs/<ts>_<run>/` | **the training script** (notebook only reads) |
| Consolidated metrics / results (e.g. eval JSON, NPZ features) | `experiments/results/<experiment>/` | scripts; notebooks may write *analysis* outputs (e.g. eval JSON) into `experiments/results/` |
| Plots / figures | `experiments/plots/` | notebook / plot scripts |

Use `PROJECT_ROOT`-relative `Path` objects and explicit `write_text` /
`torch.save` / `SummaryWriter` calls so results survive kernel restarts and
are reusable by later cells, other notebooks, and scripts. Training history,
checkpoints, and logs are **never re-saved from a notebook**.

### 2. Each cell must be runnable independently
Structure the notebook so any specific cell can be executed on its own, without
requiring all preceding cells — especially resource- or time-intensive ones:
- **Never contain a training loop.** Notebooks only load persisted artifacts
  (checkpoints, saved metrics) produced by scripts — see
  [LOGGING_CHECKPOINT_RULES.md](LOGGING_CHECKPOINT_RULES.md#1-script-only-runs--automatic-persistence).
- Recompute cheap prerequisites inline (data loaders, model builders, device),
  OR load persisted artifacts instead of depending on an earlier cell having run.
- Cells should be idempotent and safe to re-run in isolation once their cheap
  prerequisites are satisfied.
- If an artifact is missing, print the exact script command that produces it
  instead of silently training.

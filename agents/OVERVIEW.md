# OVERVIEW.md — Project Overview & Roadmap

- **Motivation/Background**: A single living overview keeps the project's
  plan, phases, and status visible without re-reading every phase doc.
- **Purpose**: Summarize the project: dataset, models, approach, phases,
  known constraints, and pointers to status docs.
- **Overview Pipeline**: Derived from `PURPOSE.md` and the roadmap template;
  updated at the start of every session/phase.
- **Detailed Plan**: §1 project; §2 plan; §3 phases; §4 known constraints;
  §5 progress pointers.
- **References**: `PURPOSE.md`, `templates/PROJECT_ROADMAP_TEMPLATE.md`,
  `phases/<PHASE>.md`.

---

> ## ✏️ HOW TO FILL THIS FILE (delete this block after filling)
>
> Replace the `[Fill in]` placeholders. Phase names should match the phase
> docs you create in `agents/phases/` (from
> [phases/PHASE_TEMPLATE.md](phases/PHASE_TEMPLATE.md)).

---

## Project

[Fill in] — one-line description of the project.

## Purpose

See [PURPOSE.md](PURPOSE.md) for the original brief.

## Plan

- **Dataset:** [Fill in — name, classes, sizes, source]
- **Models:** [Fill in — architectures, pretrained or from scratch]
- **Approach:** [Fill in — baseline → improvements → evaluation strategy]
- **Monitoring:** [Fill in — TensorBoard, logging, run tracking]
- **Success criteria:** [Fill in — measurable targets from PURPOSE.md]

## Phases

1. [DATA_PREP.md](phases/DATA_PREP.md) — data loading, transforms, split
2. [MODEL.md](phases/MODEL.md) — model building, adaptation, freeze/unfreeze
3. [TRAINING_INFO.md](phases/TRAINING_INFO.md) — training loop, hyperparameters
4. [EVAL.md](phases/EVAL.md) — evaluation, metrics, comparison
5. [Fill in more phases as planned]

## Known constraints

[Fill in — e.g. dataset size limits, input-size mismatches, compute budget,
Python-version gotchas. Flag these early; they are the most common source of
silent garbage results.]

## Progress

- [progress/DATA_PREP_STATUS.md](progress/DATA_PREP_STATUS.md)
- [progress/MODEL_STATUS.md](progress/MODEL_STATUS.md)
- [progress/TRAINING_STATUS.md](progress/TRAINING_STATUS.md)
- [progress/EVAL_STATUS.md](progress/EVAL_STATUS.md)

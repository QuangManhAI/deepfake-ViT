# <TITLE> — Template (agents/experiments)

Copy this file into `agents/experiments/` as `<EXP_OR_NAME>.md` for each
experiment. Register it in `agents/experiments/README.md`.

---

## Header

- **Title:** <Short experiment name>
- **Date created:** YYYY-MM-DD
- **Last updated:** YYYY-MM-DD
- **Description:** <One sentence: what this experiment tests.>
- **Status:** [To Do | In Progress | Done | On Hold | Canceled]
- **Experiment ID:** <e.g. EXP-07, or a short slug>

## Objective

<What hypothesis is being tested; what success looks like (with a number).>

## Single variable changed / held constant

- **Changed:** <the one thing under test>
- **Held constant:** <everything else — data, loss, scheduler, seed...>

## Setup

- **Data / split:** <source, train/val/test sizes>
- **Models / base:** <architectures, checkpoints>
- **Hyperparameters:** <lr, epochs, batch, loss, scheduler, seed>

## Results

| Metric | Before | After |
|---|---|---|
| Test accuracy | <x>% | <y>% |
| <other metric> | ... | ... |

## Analysis / interpretation

<What the numbers mean; whether the hypothesis holds; caveats.>

## Reproduce

<Commands or notebook path to rerun. Note if runtime > 5 min.>

## Links

- Notebook: <relative path>
- Status: [../progress/<EXP>_STATUS.md](../progress/<EXP>_STATUS.md)

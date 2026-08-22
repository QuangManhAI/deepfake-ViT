# Codebase Audit Report — deepfake-ViT (main @ 08d20d7)

- **Motivation/Background**: Fresh audit picking up from commit `881f8cf`
  (audit action tracker). Since then the repo gained a full DF40 data-prep
  pipeline (`prepare_df40_splits.py`, Celeb-DF extractors), 2 notebooks,
  8 data scripts, eval reports (40-method + leakage analysis), and new
  tests. Goal: verify the previous P0/P1/P2 findings actually landed, catch
  drift introduced by the new work, and re-score overall health before the
  `>95%` (target `>97.5%`) accuracy deadline.
- **Purpose**: Re-baseline code quality, security, dependencies, architecture,
  tests, performance, and rulebase compliance for `main`.
- **Overview Pipeline**: Audit process = git-diff inspection
  (`881f8cf..HEAD`) + static grep of `src/`, `tests/`, `notebooks/`,
  `agents/` + run of `tests/test_data_prep.py` (7/7) + compliance check
  against `agents/rules/*` + artifact/cross-reference checks of docs.
- **Detailed Plan**: Executive Summary → Findings Summary → per-area findings
  (code quality, security, dependencies, architecture, tests, performance,
  documentation) → Compliance → Risk → Overall Health → Prioritized Action Plan.
- **References**: `git`, `grep`, `python -m unittest`, project rulebase
  (`agents/rules/*`), `requirements.txt`, `requirements.lock.txt`,
  previous audit [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md) (main @ f3d521f),
  action tracker [CODEBASE_AUDIT_STATUS.md](progress/CODEBASE_AUDIT_STATUS.md).

---

> **AI-era audit perspective:** Strict lint/style/naming conformance is a
> **low-priority** signal. Findings focus on correctness, reproducibility,
> security, and runtime behavior; lint-only items are Info at most.

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Findings Summary](#2-findings-summary)
- [3. Code Quality](#3-code-quality)
- [4. Security Vulnerabilities](#4-security-vulnerabilities)
- [5. Dependency Health](#5-dependency-health)
- [6. Architecture Consistency](#6-architecture-consistency)
- [7. Test Coverage](#7-test-coverage)
- [8. Performance Bottlenecks](#8-performance-bottlenecks)
- [9. Documentation & Cross-Reference Drift](#9-documentation--cross-reference-drift)
- [10. Compliance with Policies and Procedures](#10-compliance-with-policies-and-procedures)
- [11. Detailed Risk Analysis](#11-detailed-risk-analysis)
- [12. Overall Project Health](#12-overall-project-health)
- [13. Prioritized Action Plan](#13-prioritized-action-plan)

---

## 1. Executive Summary

> **Scope:** branch `main` @ `08d20d7` (2026-08-22); delta reviewed from
> `881f8cf` (2026-08-18). Method = git diff + static grep + test run +
> doc-link verification.

**Verdict:** All 11 closable findings from the previous audit are genuinely
resolved (verified, not just ticked): `weights_only=True` everywhere,
`allow_pickle` gone, tests exist and pass (7/7), hardcoded `/Volumes/` paths
gone, full RNG seeding, pinned lockfile, Python 3.11, filled `OVERVIEW.md`,
`num_workers`/`pin_memory`/`--amp`, `.feynman/` untracked. The remaining open
item ARCH-1 (full-state checkpointing) is still on hold and now has a **second
instance**: `notebooks/01_full_pipeline.ipynb` re-introduces a training loop
+ `torch.load(weights_only=False)`.

New drift introduced by the DF40 data-prep work:

- **Untracked generated data breaks fresh-clone tests**: `data/splits/*.csv`
  and the split JSONs are gitignored (`*.csv`, `*.json`), but
  `tests/test_data_prep.py` requires them — a fresh clone fails `pytest`.
- **Global `*.json` gitignore silently drops result JSONs** from version
  control (e.g. `exp01_max_accuracy_report.json`,
  `final_test_report.json`) — results are invisible to the repo.
- **Hardcoded absolute paths returned** (`/workspace/hoangtuan/...`,
  `/workspace/data/...`) in the new extract scripts (CQ-1 regression on a
  different host).
- **Checkpoint path contract is broken**: `MODELS.md` §8, eval-script
  defaults (`experiments/checkpoints/finetune/...`), training defaults
  (`experiments/results/checkpoints`), and real artifacts (`outputs/...`)
  all disagree.
- **`outputs/` and `models/` are undocumented top-level trees** (drift vs
  [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md)); `outputs/` duplicates
  `experiments/results/` (research docs, lora_probs.npz, 40-method report).
- **Stale docs**: `OVERVIEW.md` links to two nonexistent notebooks
  (`full_pipeline.ipynb`, `02_advanced_accuracy_finetuning.ipynb`);
  `EVAL_STATUS.md` not updated for the 2026-08-22 eval work; lockfile missing
  `pandas` added to `requirements.txt`.

Overall health: **Good** (up from Fair). Top actions: [§13](#13-prioritized-action-plan).

---

## 2. Findings Summary

Legend: `✅ resolved` = verified fixed since previous audit; `⚠️ new` = found
in this audit; `(P0/P1/P2)` = priority tier, see [§13](#13-prioritized-action-plan).

| ID | Area | Severity | Title | Section |
|---|---|---|---|---|
| SEC-1 | Security | **High (was)** | ✅ resolved in `src/` — `weights_only=True` on all 6 loads; notebook instance tracked as [SEC-4](#4-security-vulnerabilities) | [4. Security](#4-security-vulnerabilities) |
| ⚠️ SEC-4 | Security | **Med** | `notebooks/01_full_pipeline.ipynb` uses `torch.load(weights_only=False)` + runs a training loop | [4. Security](#4-security-vulnerabilities), [6. Architecture](#6-architecture-consistency) |
| ARCH-1 | Architecture | **High** | Logging/checkpoint rules reference nonexistent modules; training still saves best-only state | [6. Architecture](#6-architecture-consistency) |
| TST-1 | Tests | **High** | ✅ resolved — `conftest.py`, `test_smoke.py`, `test_data_prep.py`, `pytest.ini`; 7/7 pass | [7. Test Coverage](#7-test-coverage) |
| ⚠️ TST-2 | Tests | **Med** | Tests depend on gitignored generated `data/splits/*` → fresh-clone `pytest` fails; `pytest` not installed in `.venv` | [7. Test Coverage](#7-test-coverage) |
| CQ-1 | Quality | **Medium** | Hardcoded machine-specific absolute paths in data scripts | [3. Code Quality](#3-code-quality) |
| CQ-2 | Quality | **Medium** | ✅ resolved — `set_seed` seeds python/numpy/torch CPU+CUDA in all 3 training scripts | [3. Code Quality](#3-code-quality) |
| DEP-1 | Dependencies | **Medium** | ⚠️ partial — lockfile exists, but `pandas>=2.0` missing from it; `*.json` ignore drops result JSONs | [5. Dependencies](#5-dependency-health) |
| DEP-2 | Dependencies | **Medium** | ✅ resolved — Python 3.11 standardized; CUDA index install documented | [5. Dependencies](#5-dependency-health) |
| ARCH-2 | Architecture | **Medium** | ✅ resolved — `OVERVIEW.md` filled; phase/status docs created | [6. Architecture](#6-architecture-consistency) |
| ARCH-3 | Architecture | **Low** | ⚠️ regression — `outputs/`, `models/`, `scratch/` undocumented; `outputs/` duplicates `experiments/results/` | [6. Architecture](#6-architecture-consistency) |
| ARCH-4 | Architecture | **Low** | ✅ resolved — `.feynman/` gitignored and untracked | [6. Architecture](#6-architecture-consistency) |
| PERF-1 | Performance | **Low** | ✅ resolved — `num_workers>0` + `pin_memory` in all training DataLoaders | [8. Performance](#8-performance-bottlenecks) |
| PERF-2 | Performance | **Low** | ✅ resolved — opt-in `--amp` (bfloat16, auto-disabled on MPS) | [8. Performance](#8-performance-bottlenecks) |
| ⚠️ DOC-1 | Docs | **Low** | `OVERVIEW.md` links to nonexistent notebooks; `EVAL_STATUS.md` stale; method-summary/metadata drift | [9. Documentation](#9-documentation--cross-reference-drift) |

Positive findings (Info): eval loops guard with `torch.no_grad()` (PERF-3),
no hardcoded secrets (SEC-3), data-prep zero-leakage assertions + tests
(DATA-1), `DF40_ROOT` env-configurable source root, clear missing-source
errors in `eval_df40_vit_cnn.py`.

Cross-reference: severity totals → [risk §11](#11-detailed-risk-analysis);
compliance gaps → [§10](#10-compliance-with-policies-and-procedures).

---

## 3. Code Quality

### CQ-1: Hardcoded machine-specific absolute paths — ⚠️ regressed in new scripts
- **Severity:** Medium
- **Description:** The previous `/Volumes/quangmanh/...` (Mac SMB) paths are
  gone (✅). But the new Celeb-DF extractors hardcode this box's layout:
  `/workspace/hoangtuan/deepfake-ViT/...` for output/splits dirs and
  `/workspace/data/...` for source roots. `prepare_df40_splits.py` is the
  good pattern (`DF40_ROOT` env with sane default); the extract scripts are
  not portable.
- **Affected:** [src/data/extract_celeb_df_frames.py](../src/data/extract_celeb_df_frames.py) (L157-159),
  [src/data/extract_all_celeb_datasets.py](../src/data/extract_all_celeb_datasets.py) (L160-162),
  [src/data/extract_celeb_df_test_suite.py](../src/data/extract_celeb_df_test_suite.py) (L153-155)
- **Remediation:** Default output/splits dirs relative to the repo
  (`data/processed/...`, `data/splits`), source root via `DF40_ROOT`/`--celeb-root`.
  Tracked in [Action P1.1](#13-prioritized-action-plan).

### CQ-2: RNG seeding — ✅ resolved
- **Severity:** Medium (was) → Resolved
- **Description:** [src/utils/seeding.py](../src/utils/seeding.py) seeds
  python, numpy, torch CPU + CUDA; all three training entry points call
  `set_seed(args.seed)` before building loaders. Docstring correctly notes
  DataLoader workers are auto-seeded by PyTorch and that `PYTHONHASHSEED`
  must be set before interpreter start.
- **Remediation:** None. (Optional: document `PYTHONHASHSEED` in the run
  command for byte-exact determinism.)

---

## 4. Security Vulnerabilities

### SEC-1: `torch.load` without `weights_only=True` — ✅ resolved in `src/`, ⚠️ present in notebook
- **Severity:** High (was) → Resolved in source
- **Description:** All 6 `torch.load` calls in `src/` now pass
  `weights_only=True` (verified by grep: `train.py`, `finetune_compare.py`,
  `analyze_threshold.py`, `eval_df40_fake.py`, `eval_finetuned.py`,
  `eval_finetuned_identity_disjoint.py`). **However**,
  `notebooks/01_full_pipeline.ipynb` (cell 19) loads a checkpoint with
  `torch.load(checkpoint_path, map_location=device, weights_only=False)` —
  a regression inside the notebook (see SEC-4).
- **Affected:** [notebooks/01_full_pipeline.ipynb](../notebooks/01_full_pipeline.ipynb)
- **Remediation:** Flip to `weights_only=True` in the notebook cell.

### SEC-2: `np.load(..., allow_pickle=True)` — ✅ resolved
- **Severity:** Medium (was) → Resolved
- **Description:** `evaluate.py`/`predict.py` now call plain
  `np.load(path)` (no `allow_pickle=True`); no `allow_pickle` anywhere in
  `src/` (grep-verified). New data scripts use `np.linspace`/`Image`, no
  untrusted deserialization.
- **Remediation:** None.

### SEC-3: No hardcoded secrets; safe subprocess — Info (positive)
- **Severity:** Info
- **Description:** No keys/tokens in code; `HF_TOKEN` appears only as
  placeholder in [MODELS.md](../MODELS.md). `subprocess` calls use argument
  lists without `shell=True`. `.env` is gitignored.
- **Remediation:** None — keep secrets via env vars.

### SEC-4: Notebook `01_full_pipeline.ipynb` — insecure load + training in notebook — ⚠️ new
- **Severity:** Medium
- **Description:** The notebook (a) calls `torch.load(..., weights_only=False)`
  (cell 19) and (b) contains a full training engine (`optimizer.step()`,
  `loss.backward()`, `torch.save` best-state, `for epoch`) in cell 15. Both
  violate [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md)
  §1 ("Notebooks never train") and the `weights_only=True` mandate; the
  checkpoint it saves is best-only state (same non-compliance as ARCH-1).
- **Affected:** [notebooks/01_full_pipeline.ipynb](../notebooks/01_full_pipeline.ipynb)
- **Remediation:** Delete/move the training cell to
  `src/training/train.py` and load saved artifacts in the notebook; fix
  `weights_only`. Tracked in [Action P0.1](#13-prioritized-action-plan).

---

## 5. Dependency Health

### DEP-1: Lockfile drift + global `*.json` ignore — ⚠️ partial
- **Severity:** Medium
- **Description:** [requirements.lock.txt](../requirements.lock.txt) exists
  (✅) and `requirements.txt` documents the CUDA-index torch install (✅).
  **New drift:** `pandas>=2.0` was added to
  [requirements.txt](../requirements.txt) (needed by the extractors) but is
  **not** in the lockfile. Separately, `.gitignore` line 7 ignores `*.json`
  globally, so eval result JSONs (`experiments/results/exp01_max_accuracy_report.json`,
  `final_test_report.json`, `split_info.json`, `methods_summary.json`) are
  silently **untracked** — the repo can't reproduce or review them. Only 3
  benchmark JSONs are tracked (committed before the rule).
- **Affected:** [requirements.lock.txt](../requirements.lock.txt),
  [.gitignore](../.gitignore), [experiments/results/](../experiments/results/)
- **Remediation:** Regenerate the lockfile (note in header) including
  `pandas`; narrow the JSON ignore to `.ipynb_checkpoints/` or allowlist
  `experiments/results/**/*.json`. Tracked in [Action P1.2](#13-prioritized-action-plan).

### DEP-2: Python 3.11 — ✅ resolved
- **Severity:** Medium (was) → Resolved
- **Description:** Setup docs (README/MODELS) standardize on Python 3.11
  (`src/utils/setup_ubuntu.sh`), and the lockfile documents `torch>=2.7.1`
  with the `cu124` index.
- **Remediation:** None.

---

## 6. Architecture Consistency

### ARCH-1: Logging/checkpoint rules vs training code — still open (on hold)
- **Severity:** High
- **Description:** [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md)
  still references nonexistent `src/training/train_model.py`,
  `src/utils/run_logger.py`, `src/utils/checkpoint_utils.py` (grep-verified:
  all three absent). Training still saves best-only dicts
  (`{"state_dict","epoch","val_metrics"}`) with no `_last.pt`, no
  optimizer/scheduler/RNG state, no JSONL history, no resume, no
  `RunLogger` — in [src/training/train.py](../src/training/train.py) (L212),
  [finetune_lora.py](../src/training/finetune_lora.py) (L215),
  [finetune_compare.py](../src/training/finetune_compare.py) (L215).
  The notebook training cell (SEC-4) repeats the same anti-pattern.
- **Affected:** `src/training/*.py`, `agents/rules/LOGGING_CHECKPOINT_RULES.md`
- **Remediation:** Either implement the rules (full-state checkpoint +
  resume + JSONL) or rescope the rules to this repo's loop. Decision still
  pending (user deferred). Tracked in [Action P0.3](#13-prioritized-action-plan).

### ARCH-2: `OVERVIEW.md` + phase/status docs — ✅ resolved
- **Severity:** Medium (was) → Resolved
- **Description:** [agents/OVERVIEW.md](OVERVIEW.md) is fully filled; all four
  phase docs and four status files exist and link correctly.
- **Remediation:** None for existence — but see DOC-1 for stale notebook links
  inside it.

### ARCH-3: Undocumented top-level trees + artifact duplication — ⚠️ regression
- **Severity:** Low
- **Description:** Previous audit said "no undocumented top-level folders".
  Now three exist: `outputs/` (research docs, eval reports, checkpoints,
  features, finetune, benchmark, logs), `models/` (3 pretrained weights,
  symlinked from `experiments/checkpoints/weights/*`), and `scratch/`
  (gitignored throwaway scripts). `outputs/` **duplicates**
  `experiments/results/`: 7/8 research docs byte-identical (1 differs only
  in internal links), `lora_probs.npz` and `report_40_methods_v3.md` exist
  in both trees. Duplication → divergence risk.
- **Affected:** [outputs/](../outputs/), [models/](../models/),
  [experiments/results/](../experiments/results/)
- **Remediation:** Pick one canonical results tree (`experiments/results/`),
  gitignore `outputs/` docs or remove the copies; document `models/` in
  [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md). Tracked in
  [Action P1.3](#13-prioritized-action-plan).

### ARCH-4: `.feynman/` tool cache — ✅ resolved
- **Severity:** Low (was) → Resolved
- **Description:** `.feynman/` is in `.gitignore` and no longer tracked
  (`git ls-files .feynman` = 0; `git check-ignore` succeeds).
- **Remediation:** None.

---

## 7. Test Coverage

### TST-1: No automated tests — ✅ resolved
- **Severity:** High (was) → Resolved
- **Description:** [tests/conftest.py](../tests/conftest.py),
  [tests/test_smoke.py](../tests/test_smoke.py),
  [tests/test_data_prep.py](../tests/test_data_prep.py), and
  [pytest.ini](../pytest.ini) exist. `test_data_prep.py` covers split-file
  existence, per-method suites, **0% identity/image leakage**, exact 1:1
  balance, DataLoader batching, and split metadata. Ran
  `python -m unittest tests.test_data_prep -v` → **7/7 pass**.
- **Remediation:** None for existence.

### TST-2: Tests depend on untracked generated data; `pytest` missing — ⚠️ new
- **Severity:** Medium
- **Description:** `data/splits/*.csv` and `split_info.json` /
  `methods_summary.json` are gitignored (`*.csv`, `*.json`) and **not**
  tracked (`git ls-files data/splits` = 0). `tests/test_data_prep.py`
  hard-requires them, so a **fresh clone fails `pytest`** until the data-prep
  pipeline runs (which itself needs `/workspace/data` sources). Separately,
  `pytest` is not installed in `.venv` and not in
  [requirements.txt](../requirements.txt) — `python -m pytest` errors out
  (only `unittest` works today).
- **Affected:** [tests/test_data_prep.py](../tests/test_data_prep.py),
  [.gitignore](../.gitignore), [pytest.ini](../pytest.ini)
- **Remediation:** (a) commit small `split_info.json`/`methods_summary.json`
  + a tiny fixture subset, or add a "skip if splits missing" marker; (b) add
  `pytest` to requirements (or keep unittest as the documented runner).
  Tracked in [Action P0.2](#13-prioritized-action-plan).

---

## 8. Performance Bottlenecks

### PERF-1: `num_workers=0` — ✅ resolved
- **Severity:** Low (was) → Resolved
- **Description:** All three training scripts expose `--num-workers`
  (default 2) and pass `num_workers=args.num_workers, pin_memory=True` to
  every DataLoader. Verified in `train.py`, `finetune_lora.py`,
  `finetune_compare.py`.
- **Remediation:** None.

### PERF-2: No AMP — ✅ resolved
- **Severity:** Low (was) → Resolved
- **Description:** `--amp` opt-in flag wraps forward passes in
  `torch.autocast(device_type, dtype=torch.bfloat16)`, auto-disabled on MPS
  (`enabled=args.amp and device != "mps"`). Eval loops correctly guard with
  `torch.no_grad()` (PERF-3, still positive).
- **Remediation:** None.

---

## 9. Documentation & Cross-Reference Drift

### DOC-1: Stale notebook links, stale status file, metadata drift — ⚠️ new
- **Severity:** Low
- **Description:**
  - [agents/OVERVIEW.md](OVERVIEW.md) links `notebooks/full_pipeline.ipynb`
    and `notebooks/02_advanced_accuracy_finetuning.ipynb` — **neither
    exists**; the real notebooks are `00_comprehensive_dataset_eda.ipynb`
    and `01_full_pipeline.ipynb`. The same dead `02_...` link appears in
    `phases/TRAINING_INFO.md`, `progress/TRAINING_STATUS.md`, and
    `experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md` (its "Target
    Deliverable" is a notebook that was never created).
  - [agents/progress/EVAL_STATUS.md](progress/EVAL_STATUS.md) says "Last
    updated 2026-08-18" but major eval work (40-method report, leakage
    analysis, lora evals) landed 2026-08-21/22 in
    [outputs/eval/](../outputs/eval/). `DATA_PREP_STATUS.md` was updated;
    `EVAL_STATUS.md` was not.
  - Metadata drift: `data/splits/methods_summary.json` lists 39 methods, but
    `data/splits/methods/` holds **200 files** including `*CelebDFv2*` sets
    absent from the summary.
- **Affected:** [agents/OVERVIEW.md](OVERVIEW.md), [agents/progress/EVAL_STATUS.md](progress/EVAL_STATUS.md), [agents/experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md](experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md), [data/splits/methods_summary.json](../data/splits/methods_summary.json)
- **Remediation:** Point `OVERVIEW.md` at the real notebooks (or remove);
  update `EVAL_STATUS.md`; regenerate/annotate `methods_summary.json`.
  Tracked in [Action P2.1](#13-prioritized-action-plan).

---

## 10. Compliance with Policies and Procedures

Assessed against the project rulebase (`agents/rules/*`).

| Policy / procedure | Compliance | Evidence / gap | Related finding |
|---|---|---|---|
| `FOLDER_STRUCTURE.md` | **Partial** | `outputs/`, `models/`, `scratch/`, `data/splits/` not documented; results duplicated | [ARCH-3](#6-architecture-consistency) |
| `NAMING_CONVENTION.md` | **Compliant** | New scripts snake_case; splits follow `test_<method>_balanced.csv` pattern | — |
| `MD_CONVENTION.md` | **Partial** | Header/TOC rules followed in new docs; dead notebook links violate cross-reference rule | [DOC-1](#9-documentation--cross-reference-drift) |
| `LOGGING_CHECKPOINT_RULES.md` | **Non-compliant** | Training best-only dicts, no resume/JSONL; notebook trains; `weights_only=False` in notebook | [ARCH-1](#6-architecture-consistency), [SEC-4](#4-security-vulnerabilities) |
| `RESULTS_REPORTING.md` | **Partial** | Eval reports carry protocol/5W1H context (leakage doc is thorough); 40-method report lacks an explicit 5W1H block | — |
| `NOTEBOOK_HEADER_CONVENTION.md` | **Partial** | Both notebooks have title/roadmap/references; `01` contains a training loop → violates output-persistence & no-training rules | [SEC-4](#4-security-vulnerabilities) |
| `CODEBASE_AUDIT.md` | **Compliant** | This report is the output of the mandated procedure | — |
| `SMOKE_TEST_CHECKLIST.md` | **Partial** | Smoke tests exist; but `pytest` not installed and tests need untracked splits | [TST-2](#7-test-coverage) |

Cross-reference: non-compliance maps to [risk §11](#11-detailed-risk-analysis)
and [action §13](#13-prioritized-action-plan).

---

## 11. Detailed Risk Analysis

| Risk | Likelihood | Impact | Overall | Description & mitigation | Related finding |
|---|---|---|---|---|---|
| Fresh-clone tests/results unavailable | High | Med | **High** | `data/splits/*` + result JSONs gitignored → `pytest` fails and eval results are invisible to the repo; team can't verify or reproduce numbers. Mitigate: allowlist result JSONs, commit/fixture splits, add pytest | [TST-2](#7-test-coverage), [DEP-1](#5-dependency-health) |
| Rulebase/code drift (logging) | High | Med | **High** | Rules document a checkpoint/resume system that neither scripts nor the notebook implement. Mitigate: implement or rescope | [ARCH-1](#6-architecture-consistency), [SEC-4](#4-security-vulnerabilities) |
| Pickle RCE via notebook load | Low | High | **Med** | `weights_only=False` in a committed notebook → unsafe if checkpoint is untrusted. Mitigate: fix cell | [SEC-4](#4-security-vulnerabilities) |
| Non-portable data pipeline | Med | Med | **Med** | New extractors hardcode `/workspace/...`; running on another host silently uses wrong inputs. Mitigate: repo-relative defaults + env roots | [CQ-1](#3-code-quality) |
| Duplicate artifact trees diverge | Med | Low | **Low-Med** | `outputs/` vs `experiments/results/` already differ in one file's internal links. Mitigate: single canonical tree | [ARCH-3](#6-architecture-consistency) |
| Doc drift misleads agents/humans | Med | Low | **Low** | Dead notebook links + stale `EVAL_STATUS.md` + method-summary mismatch. Mitigate: fix links, update status | [DOC-1](#9-documentation--cross-reference-drift) |

Cross-reference: severities originate in [§2](#2-findings-summary);
mitigations scheduled in [§13](#13-prioritized-action-plan).

---

## 12. Overall Project Health

| Dimension | Rating | Notes |
|---|---|---|
| Code quality | **Good** | Clean model code and new split logic; hardcoded absolute paths returned ([CQ-1](#3-code-quality)) |
| Security | **Good** | `weights_only=True` in all `src/` loads; one notebook regression ([SEC-4](#4-security-vulnerabilities)) |
| Dependencies | **Fair** | Lockfile exists but stale (`pandas`); global `*.json` ignore hides results ([DEP-1](#5-dependency-health)) |
| Architecture | **Good** | Clean tree; but `outputs/`/`models/` undocumented + duplication, logging rules still unrealized ([ARCH-1](#6-architecture-consistency), [ARCH-3](#6-architecture-consistency)) |
| Tests | **Good** | 7/7 pass locally, real leakage/balance assertions; fresh-clone portability gap ([TST-2](#7-test-coverage)) |
| Performance | **Good** | `num_workers`/`pin_memory`/`--amp` in place |
| Reproducibility | **Fair** | Full seeding; but splits/JSONs untracked and lockfile stale |
| Documentation | **Good** | Phase/status docs exist; dead notebook links + stale EVAL status ([DOC-1](#9-documentation--cross-reference-drift)) |

**Overall rating: Good** (up from Fair). The previous audit's P0/P1/P2 items
are genuinely closed; the remaining open item is ARCH-1 (user-held) plus new
drift introduced by the DF40 data-prep expansion.

Cross-reference: per-dimension evidence in
[§3](#3-code-quality)–[§9](#9-documentation--cross-reference-drift).

---

## 13. Prioritized Action Plan

### P0 — Fix now (blocks trust / reproducibility / security)
- **P0.1** Remove the training loop from
  `notebooks/01_full_pipeline.ipynb` and fix `weights_only=False` — addresses
  [SEC-4](#4-security-vulnerabilities), [ARCH-1](#6-architecture-consistency)
- **P0.2** Make tests fresh-clone safe: allowlist result JSONs in
  `.gitignore`, add `pytest` to requirements, and add a skip-if-splits-missing
  guard or commit split metadata — addresses [TST-2](#7-test-coverage),
  [DEP-1](#5-dependency-health)
- **P0.3** Decide ARCH-1: implement `LOGGING_CHECKPOINT_RULES` in
  `src/training/*.py` **or** rescope the rules to the actual loop — addresses
  [ARCH-1](#6-architecture-consistency)

### P1 — Next iteration (raises confidence)
- **P1.1** Replace `/workspace/...` defaults in the three extract scripts
  with repo-relative paths + `DF40_ROOT`/`--celeb-root` — addresses
  [CQ-1](#3-code-quality)
- **P1.2** Regenerate `requirements.lock.txt` (add `pandas`) — addresses
  [DEP-1](#5-dependency-health)
- **P1.3** Consolidate `outputs/` into `experiments/results/` (or gitignore
  the duplicate docs) and document `models/` in
  [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md) — addresses
  [ARCH-3](#6-architecture-consistency)

### P2 — Polish (when time permits)
- **P2.1** Fix `OVERVIEW.md` notebook links, update `EVAL_STATUS.md`, and
  reconcile `methods_summary.json` (39 vs 40 incl. CelebDFv2) — addresses
  [DOC-1](#9-documentation--cross-reference-drift)

> Per the AI-era perspective, no lint/tooling items are gating actions here.

Cross-reference: each item links back to its finding; the summary table is in
[§2](#2-findings-summary).

---

## Self-review checklist

- [x] All 5 header fields present
- [x] TOC anchors resolve (lowercase, strip punctuation, spaces → hyphens)
- [x] Cross-reference links between related sections resolve (summary ↔ detail ↔ action plan)
- [x] Metrics match source (test run 7/7, grep counts verified)
- [x] File/notebook paths relative to project root; dates in `YYYY-MM-DD`
- [x] Previous findings marked resolved only when grep/test-verified

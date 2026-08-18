# Codebase Audit Report — deepfake-ViT (main @ f3d521f)

- **Motivation/Background**: This audit runs immediately after a large
  repository re-organization (scripts/ → `src/`, artifacts → `experiments/`)
  and the addition of the `agents/` rulebase, so the goal is to verify the
  reorganized tree, catch drift between the rulebase and the real code, and
  establish a baseline of correctness, security, and reproducibility before
  further work toward the `>95%` test-accuracy target.
- **Purpose**: Establish a baseline of code quality, security, dependencies,
  architecture, tests, performance, and rulebase compliance for `main`.
- **Overview Pipeline**: Audit process = git-tree inspection + static grep of
  `src/`, `tests/`, `experiments/`, `agents/` + dependency review against
  `requirements.txt` + compliance check against the project rulebase
  (`agents/rules/*`).
- **Detailed Plan**: Executive Summary → Findings Summary → per-area findings
  (code quality, security, dependencies, architecture, tests, performance) →
  Compliance → Risk Analysis → Overall Health → Prioritized Action Plan.
- **References**: `git`, `grep`, `pip`, project rulebase
  (`agents/rules/*`), `agents/templates/CODEBASE_AUDIT_TEMPLATE.md`,
  `requirements.txt`, previous audit (none).

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
- [9. Compliance with Policies and Procedures](#9-compliance-with-policies-and-procedures)
- [10. Detailed Risk Analysis](#10-detailed-risk-analysis)
- [11. Overall Project Health](#11-overall-project-health)
- [12. Prioritized Action Plan](#12-prioritized-action-plan)

---

## 1. Executive Summary

> **Scope:** branch `main` @ `f3d521f` (2026-08-18); method = git-tree
> inspection + static grep of `src/`, `tests/`, `experiments/`, `agents/` +
> dependency/compliance review.

**Verdict:** The reorganized structure now cleanly matches the template, and
the DINOv3 model code (`src/models/`) is well-built and readable. What blocks
maturity:

- **Untrusted-file deserialization**: `torch.load` without `weights_only=True`
  in 6 files and `np.load(allow_pickle=True)` in 2 — pickle RCE risk.
- **Rulebase/code drift**: `agents/rules/LOGGING_CHECKPOINT_RULES.md` and
  `agents/rules/MD_CONVENTION.md` reference modules that do not exist
  (`train_model.py`, `run_logger.py`, `checkpoint_utils.py`), and `OVERVIEW.md`
  is still a placeholder referencing missing phase/status docs.
- **Zero automated tests** and **unpinned dependencies** undermine
  reproducibility.

Overall health: **Fair** (see [§11](#11-overall-project-health)). Top actions:
[§12 P0](#12-prioritized-action-plan).

---

## 2. Findings Summary

| ID | Area | Severity | Title | Section |
|---|---|---|---|---|
| SEC-1 | Security | **High** | `torch.load` without `weights_only=True` (pickle RCE) | [4. Security](#4-security-vulnerabilities) |
| TST-1 | Tests | **High** | No automated tests; `tests/` is empty scaffolding | [7. Test Coverage](#7-test-coverage) |
| ARCH-1 | Architecture | **High** | Logging/checkpoint rules reference nonexistent modules; training is non-compliant | [6. Architecture](#6-architecture-consistency) |
| SEC-2 | Security | **Medium** | `np.load(..., allow_pickle=True)` on untrusted NPZ | [4. Security](#4-security-vulnerabilities) |
| CQ-1 | Code quality | **Medium** | Hardcoded absolute Mac SMB path in 4 data scripts | [3. Code Quality](#3-code-quality) |
| CQ-2 | Code quality | **Medium** | Incomplete RNG seeding / no reproducible environment | [3. Code Quality](#3-code-quality) |
| DEP-1 | Dependencies | **Medium** | Unpinned `>=` deps; `torch`/`torchvision` excluded from lockfile | [5. Dependencies](#5-dependency-health) |
| DEP-2 | Dependencies | **Medium** | Python 3.9 EOL vs DINOv3 requiring `torch>=2.7.1` | [5. Dependencies](#5-dependency-health) |
| ARCH-2 | Architecture | **Medium** | `OVERVIEW.md` placeholder; phase/status docs missing | [6. Architecture](#6-architecture-consistency) |
| PERF-1 | Performance | **Low** | `num_workers=0` in all DataLoaders | [8. Performance](#8-performance-bottlenecks) |
| PERF-2 | Performance | **Low** | No mixed-precision (AMP) training | [8. Performance](#8-performance-bottlenecks) |
| ARCH-4 | Architecture | **Low** | `.feynman/` tool cache tracked in git, not ignored | [6. Architecture](#6-architecture-consistency) |

Positive findings (Info): structure matches
[FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md) (ARCH-3), eval loops guard with
`torch.no_grad()` (PERF-3), no secrets hardcoded (SEC-3).

Cross-reference: severity totals → [risk §10](#10-detailed-risk-analysis);
compliance gaps → [§9](#9-compliance-with-policies-and-procedures).

---

## 3. Code Quality

> Per the AI-era perspective, pure lint/style issues are treated as Info and
> omitted; Medium+ is reserved for duplication with drift risk, dead code that
> breaks behavior, or maintainability blocking correctness/reproducibility.

### CQ-1: Hardcoded absolute Mac SMB path in data build scripts
- **Severity:** Medium
- **Description:** Four `src/data/` scripts hardcode
  `/Volumes/quangmanh/Downloads/DF40` as the DF40 source root (a MacBook Air
  SMB mount). This is non-portable: the pipeline will fail on the RTX 4060
  machine or any non-Mac host unless the user edits the source, and it is a
  silent reproducibility hazard (different machines → different inputs).
- **Affected:** [src/data/build_df40_balanced.py](../src/data/build_df40_balanced.py),
  [src/data/build_df40_subset.py](../src/data/build_df40_subset.py),
  [src/data/build_test_data.py](../src/data/build_test_data.py),
  [src/data/build_test_data_v2.py](../src/data/build_test_data_v2.py)
- **Remediation:** Replace the constant with a `--src`/`DF40_ROOT` argument
  or env var defaulting to `data/raw/DF40`; never bake a machine-specific
  path. Tracked in [Action P1.1](#12-prioritized-action-plan).

### CQ-2: Incomplete RNG seeding and no reproducible environment
- **Severity:** Medium
- **Description:** Training scripts set `torch.manual_seed` and
  `np.random.seed` but not `torch.cuda.manual_seed_all` / `random.seed`, and
  DataLoader shuffling is not deterministically seeded. Combined with unpinned
  dependencies ([DEP-1](#5-dependency-health)), exact run reproducibility is
  not guaranteed — relevant to the `>95%` rubric claim.
- **Affected:** [src/training/train.py](../src/training/train.py),
  [src/training/finetune_lora.py](../src/training/finetune_lora.py),
  [src/training/finetune_compare.py](../src/training/finetune_compare.py)
- **Remediation:** Set all RNG seeds (torch CPU/CUDA, numpy, python `random`,
  `generator=` in DataLoaders) and record them in run metadata. Tracked in
  [Action P2.3](#12-prioritized-action-plan).

---

## 4. Security Vulnerabilities

### SEC-1: `torch.load` without `weights_only=True` (pickle RCE)
- **Severity:** High
- **Description:** Six scripts load checkpoints with
  `torch.load(..., map_location=...)` and no `weights_only=True`. Loading an
  untrusted `.pt` file can execute arbitrary code via pickle. The rulebase
  [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md) mandates
  `weights_only=True`; the code violates it (also a compliance gap, §9).
- **Affected:** [src/training/train.py](../src/training/train.py) (L204),
  [src/training/finetune_compare.py](../src/training/finetune_compare.py) (L212),
  [src/eval/analyze_threshold.py](../src/eval/analyze_threshold.py) (L59),
  [src/eval/eval_df40_fake.py](../src/eval/eval_df40_fake.py) (L85),
  [src/eval/eval_finetuned.py](../src/eval/eval_finetuned.py) (L132),
  [src/eval/eval_finetuned_identity_disjoint.py](../src/eval/eval_finetuned_identity_disjoint.py) (L128)
- **Remediation:** Add `weights_only=True` to every `torch.load` call (safe
  for the plain dict checkpoints used here). Tracked in
  [Action P0.1](#12-prioritized-action-plan).

### SEC-2: `np.load(..., allow_pickle=True)` on untrusted NPZ
- **Severity:** Medium
- **Description:** Two eval scripts load NPZ feature files with
  `allow_pickle=True`, which permits pickle-based code execution if the NPZ is
  untrusted. Prefer loading with `allow_pickle=False` (safe for numeric
  arrays) or a trusted source check.
- **Affected:** [src/eval/evaluate.py](../src/eval/evaluate.py) (L30),
  [src/eval/predict.py](../src/eval/predict.py) (L32)
- **Remediation:** Drop `allow_pickle=True` unless object arrays are
  genuinely required; otherwise validate provenance. Tracked in
  [Action P0.2](#12-prioritized-action-plan).

### SEC-3: No hardcoded secrets; subprocess used safely — Info (positive)
- **Severity:** Info
- **Description:** No API keys/tokens/passwords are hardcoded in code or shell
  scripts (`HF_TOKEN` appears only as the `hf_xxx` placeholder in
  [RUNPOD.md](../RUNPOD.md)). `subprocess.run` calls use argument-list form
  (no `shell=True`) with fixed binaries (`unzip`, `gdown`, `ffmpeg`).
- **Affected:** repo-wide
- **Remediation:** None required — keep secrets out of code; note tokens only
  via env vars.

---

## 5. Dependency Health

### DEP-1: Unpinned dependencies; `torch`/`torchvision` excluded from lockfile
- **Severity:** Medium
- **Description:** [requirements.txt](../requirements.txt) uses unpinned
  `>=` floors only, and `torch`/`torchvision` are deliberately excluded
  (installed separately with a CUDA index, `cu124`). There is no lockfile, so
  two machines can install different versions and yield different results —
  a direct threat to the reproducibility the coursework report claims.
- **Affected:** [requirements.txt](../requirements.txt)
- **Remediation:** Pin exact versions (or add a lockfile), and document the
  exact `torch`/`torchvision` CUDA install command + versions in `README.md`.
  Tracked in [Action P1.3](#12-prioritized-action-plan).

### DEP-2: Python 3.9 EOL vs DINOv3's `torch>=2.7.1`
- **Severity:** Medium
- **Description:** [README.md](../README.md) states the project runs on Python
  3.9, which is EOL (2025-10). DINOv3 requires `torch>=2.7.1`/`timm>=1.0.20`;
  some combinations may require newer Python. The README already flags this.
- **Affected:** [README.md](../README.md), environment setup
- **Remediation:** Standardize on Python 3.10/3.11 in setup docs and CI, and
  confirm the minimum supported Python for the pinned torch/timm. Tracked in
  [Action P1.4](#12-prioritized-action-plan).

---

## 6. Architecture Consistency

### ARCH-1: Logging/checkpoint rules reference nonexistent modules; training is non-compliant
- **Severity:** High
- **Description:** [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md)
  and [MD_CONVENTION.md](rules/MD_CONVENTION.md) reference
  `src/training/train_model.py`, `src/utils/run_logger.py`, and
  `src/utils/checkpoint_utils.py` — **none exist** (grep/glob confirmed). The
  actual training entry point is [src/training/train.py](../src/training/train.py),
  which does **not** implement the rules: it saves only a best-state dict
  (`{"state_dict","epoch","val_metrics"}`) with no optimizer/scheduler/RNG
  state, no `_last.pt`, no resume, no per-epoch JSONL history/config, and no
  `RunLogger`. This is documented-but-missing **and** code-not-following-rules
  drift.
- **Affected:** [src/training/train.py](../src/training/train.py),
  [src/training/finetune_compare.py](../src/training/finetune_compare.py),
  [src/training/finetune_lora.py](../src/training/finetune_lora.py);
  rules `LOGGING_CHECKPOINT_RULES.md`, `MD_CONVENTION.md`
- **Remediation:** Either implement the rules in `train.py` (full-state
  checkpoints + resume + `_last.pt` + history/config JSONL + `run_logger.py`),
  or explicitly rescope the rules to match this repo's simpler training loop.
  Do not leave both true simultaneously. Tracked in
  [Action P0.3](#12-prioritized-action-plan).

### ARCH-2: `OVERVIEW.md` is a placeholder and references missing phase/status docs
- **Severity:** Medium
- **Description:** [agents/OVERVIEW.md](OVERVIEW.md) still contains the
  template `[Fill in]` placeholders and links to phase docs
  (`phases/DATA_PREP.md`, `MODEL.md`, `TRAINING_INFO.md`, `EVAL.md`) and
  status files (`progress/*_STATUS.md`) that do not exist — only the
  `PHASE_TEMPLATE.md` and `PROGRESS_TEMPLATE.md` are present.
- **Affected:** [agents/OVERVIEW.md](OVERVIEW.md),
  [agents/phases/](phases/), [agents/progress/](progress/)
- **Remediation:** Fill `OVERVIEW.md` from the locked
  [PURPOSE.md](PURPOSE.md), and either create the phase/status docs or remove
  the dead links. Tracked in [Action P1.2](#12-prioritized-action-plan).

### ARCH-3: `FOLDER_STRUCTURE.md` matches the actual tree — Info (positive)
- **Severity:** Info
- **Description:** After the re-org, `src/{data,eval,experiments,models,training,utils}`,
  `data/{raw,processed,external}`, and `experiments/{checkpoints,plots,results,runs}`
  match [FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md). No top-level
  undocumented folders remain (root extras `RUNPOD.md` and `.feynman/` are
  noted separately).
- **Affected:** repo structure
- **Remediation:** None required; keep the tree in sync going forward.

### ARCH-4: `.feynman/` tool cache tracked in git, not ignored
- **Severity:** Low
- **Description:** `.feynman/cache/fetch-content/arxiv-200404730.md` is
  tracked in git (a tool cache committed by `git add -A`); `check-ignore`
  confirms `.feynman/` is not gitignored. Tool caches should not be versioned.
- **Affected:** `.feynman/`
- **Remediation:** Add `.feynman/` to `.gitignore` and `git rm --cached` the
  cache. Tracked in [Action P2.2](#12-prioritized-action-plan).

---

## 7. Test Coverage

### TST-1: No automated tests; `tests/` is empty scaffolding
- **Severity:** High
- **Description:** [tests/](../tests/) contains only `.gitkeep` and
  `__init__.py` — zero test modules, no `conftest.py`, no
  `test_smoke.py`, and no `pytest.ini`/`pyproject.toml` to run them. Given a
  hard coursework deadline and a large recent re-org (moves + path rewrites),
  there is no safety net against regressions. The template and the
  `FOLDER_STRUCTURE`/`SMOKE_TEST_CHECKLIST` rules expect tests and a smoke test.
- **Affected:** [tests/](../tests/)
- **Remediation:** Add `conftest.py` (root on `sys.path`) + `test_smoke.py`
  that imports the `src` packages and smoke-loads the model graph, plus a
  `pytest.ini`. Tracked in [Action P0.4](#12-prioritized-action-plan).

---

## 8. Performance Bottlenecks

### PERF-1: `num_workers=0` in all DataLoaders
- **Severity:** Low
- **Description:** Every training/eval DataLoader uses `num_workers=0`,
  loading images on the main thread — a serialized CPU bottleneck during GPU
  training, especially on the RTX 4060 with a 256×256 dataset.
- **Affected:** [src/training/train.py](../src/training/train.py),
  [src/training/finetune_lora.py](../src/training/finetune_lora.py),
  [src/training/finetune_compare.py](../src/training/finetune_compare.py)
- **Remediation:** Set `num_workers` to a small value (e.g. 2–4) and add
  `pin_memory=True`. Tracked in [Action P2.1](#12-prioritized-action-plan).

### PERF-2: No mixed-precision (AMP) training
- **Severity:** Low
- **Description:** Training uses full FP32; on an 8 GB card this limits batch
  size / input resolution. `torch.autocast` would speed up training and reduce
  memory. Eval loops correctly guard with `torch.no_grad()` (positive).
- **Affected:** [src/training/train.py](../src/training/train.py),
  [src/training/finetune_lora.py](../src/training/finetune_lora.py),
  [src/training/finetune_compare.py](../src/training/finetune_compare.py)
- **Remediation:** Add `torch.autocast(device_type, dtype=torch.bfloat16)` (or
  fp16 with GradScaler) as an opt-in flag. Tracked in
  [Action P2.1](#12-prioritized-action-plan).

---

## 9. Compliance with Policies and Procedures

Assessed against the project rulebase (`agents/rules/*`).

| Policy / procedure | Compliance | Evidence / gap | Related finding |
|---|---|---|---|
| `FOLDER_STRUCTURE.md` | **Compliant** | Actual tree matches the documented layout after re-org | [ARCH-3](#6-architecture-consistency) |
| `NAMING_CONVENTION.md` | **Partial** | Scripts/classes `snake_case`/`PascalCase` OK; checkpoint naming deviates from `LOGGING` (only best state, no `_last.pt`) | [ARCH-1](#6-architecture-consistency) |
| `MD_CONVENTION.md` | **Partial** | `PURPOSE.md` conforms; `OVERVIEW.md` unfilled; some rule cross-refs point to nonexistent modules (`train_model.py`) | [ARCH-1](#6-architecture-consistency), [ARCH-2](#6-architecture-consistency) |
| `LOGGING_CHECKPOINT_RULES.md` | **Non-compliant** | Training saves best-only dict; no `weights_only=True`, resume, `_last.pt`, JSONL history/config, `RunLogger` | [SEC-1](#4-security-vulnerabilities), [ARCH-1](#6-architecture-consistency) |
| `RESULTS_REPORTING.md` | **Partial** | Eval scripts write JSON reports and `experiments/results/README.md` exists, but 5W1H context blocks are sparse | — |
| `NOTEBOOK_HEADER_CONVENTION.md` | **N/A** | `notebooks/` is empty (no notebooks yet) | — |
| `CODEBASE_AUDIT.md` | **Compliant** | This report is the output of the mandated procedure | — |
| `SMOKE_TEST_CHECKLIST.md` | **Non-compliant** | No smoke test exists to run before long runs | [TST-1](#7-test-coverage) |

Cross-reference: non-compliance maps to
[risk §10](#10-detailed-risk-analysis) and [action §12](#12-prioritized-action-plan).

---

## 10. Detailed Risk Analysis

| Risk | Likelihood | Impact | Overall | Description & mitigation | Related finding |
|---|---|---|---|---|---|
| Pickle RCE via untrusted checkpoints | Med | High | **High** | `torch.load`/`np.load(allow_pickle=True)` on untrusted files → arbitrary code execution. Mitigate: `weights_only=True`, drop `allow_pickle` | [SEC-1](#4-security-vulnerabilities), [SEC-2](#4-security-vulnerabilities) |
| Rulebase/code drift (logging) | High | Med | **High** | Rules document a checkpoint/resume system the code does not implement; any agent/human building on the rules will assume wrong behavior. Mitigate: align code or rules | [ARCH-1](#6-architecture-consistency) |
| Regression before deadline (no tests) | High | Med | **High** | Large re-org with zero tests → silent breakage of path/import rewrites. Mitigate: smoke test + pytest | [TST-1](#7-test-coverage) |
| Non-reproducible results | Med | Med | **Med** | Unpinned deps + hardcoded machine paths → different machines give different numbers, undermining the `>95%` claim. Mitigate: pin deps, configurable data root, full seeding | [DEP-1](#5-dependency-health), [CQ-1](#3-code-quality), [CQ-2](#3-code-quality) |
| Portability failure on target GPU | Med | Med | **Med** | `/Volumes/...` source paths and Python 3.9 EOL break setup on the RTX 4060 host. Mitigate: env-configured roots, Python 3.10+ | [CQ-1](#3-code-quality), [DEP-2](#5-dependency-health) |
| Doc drift misleads future work | Med | Low | **Low-Med** | `OVERVIEW.md` placeholder + missing phase/status docs. Mitigate: fill docs or remove dead links | [ARCH-2](#6-architecture-consistency) |

Cross-reference: severities originate in
[§2](#2-findings-summary); mitigations scheduled in [§12](#12-prioritized-action-plan).

---

## 11. Overall Project Health

| Dimension | Rating | Notes |
|---|---|---|
| Code quality | **Good** | Clear, readable DINOv3/ConvNeXt/LoRA model code; modular `src/`; but hardcoded absolute paths ([CQ-1](#3-code-quality)) |
| Security | **Fair** | No secrets hardcoded; but untrusted deserialization in 8 sites ([SEC-1](#4-security-vulnerabilities), [SEC-2](#4-security-vulnerabilities)) |
| Dependencies | **Fair** | Sensible, minimal set; but unpinned and `torch` excluded from lockfile ([DEP-1](#5-dependency-health)) |
| Architecture | **Good** | Clean template-conformant tree; but rules reference nonexistent modules and training is non-compliant ([ARCH-1](#6-architecture-consistency)) |
| Tests | **None** | Zero automated tests ([TST-1](#7-test-coverage)) |
| Performance | **Fair** | Correct `no_grad` in eval; but `num_workers=0` and no AMP ([PERF-1](#8-performance-bottlenecks), [PERF-2](#8-performance-bottlenecks)) |
| Reproducibility | **Fair** | Partial seeding; unpinned env; machine-specific paths |
| Documentation | **Fair** | `PURPOSE.md` locked; `OVERVIEW.md`/phase/status docs missing ([ARCH-2](#6-architecture-consistency)) |

**Overall rating: Fair.**

Cross-reference: per-dimension evidence in
[§3](#3-code-quality)–[§8](#8-performance-bottlenecks).

---

## 12. Prioritized Action Plan

### P0 — Fix now (blocks trust / reproducibility / security)
- **P0.1** Add `weights_only=True` to all six `torch.load` calls — addresses [SEC-1](#4-security-vulnerabilities)
- **P0.2** Remove `allow_pickle=True` from `evaluate.py`/`predict.py` — addresses [SEC-2](#4-security-vulnerabilities)
- **P0.3** Align `src/training/train.py` with `LOGGING_CHECKPOINT_RULES.md`
  (full-state checkpoints + resume + `_last.pt` + JSONL history/config +
  `run_logger.py`) **or** rescope the rules to the actual loop — addresses [ARCH-1](#6-architecture-consistency)
- **P0.4** Add `tests/conftest.py` + `tests/test_smoke.py` + `pytest.ini` and
  smoke-test the reorganized imports — addresses [TST-1](#7-test-coverage)

### P1 — Next iteration (raises confidence)
- **P1.1** Replace hardcoded `/Volumes/quangmanh/Downloads/DF40` with a
  configurable `--src`/env root — addresses [CQ-1](#3-code-quality)
- **P1.2** Fill `OVERVIEW.md` from `PURPOSE.md` and create (or prune) the
  phase/status docs — addresses [ARCH-2](#6-architecture-consistency)
- **P1.3** Pin dependencies / add a lockfile and document the exact
  `torch`/`torchvision` CUDA install — addresses [DEP-1](#5-dependency-health)
- **P1.4** Standardize on Python 3.10+ — addresses [DEP-2](#5-dependency-health)

### P2 — Polish (when time permits)
- **P2.1** Set `num_workers>0`/`pin_memory` and add optional AMP — addresses [PERF-1](#8-performance-bottlenecks), [PERF-2](#8-performance-bottlenecks)
- **P2.2** Gitignore and untrack `.feynman/` — addresses [ARCH-4](#6-architecture-consistency)
- **P2.3** Full RNG seeding (CPU/CUDA/python/data-loader) + record seeds in run metadata — addresses [CQ-2](#3-code-quality)

> Per the AI-era perspective, no lint/tooling items are gating actions here.

Cross-reference: each item links back to its finding; the summary table is in
[§2](#2-findings-summary).

---

## Self-review checklist

- [x] All 5 header fields present
- [x] TOC anchors resolve (lowercase, strip punctuation, spaces → hyphens)
- [x] Cross-reference links between related sections resolve (summary ↔ detail ↔ action plan)
- [x] Metrics match source; file/notebook paths relative to project root
- [x] Dates in `YYYY-MM-DD`; `---` separators between major sections

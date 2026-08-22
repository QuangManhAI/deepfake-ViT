# Codebase Audit — Action Tracker (agents/progress)

- **Title:** Codebase audit remediation tracker (main @ f3d521f → 08d20d7)
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-22
- **Description:** Tracks resolution of every finding in
  [CODEBASE_AUDIT.md](../CODEBASE_AUDIT.md); this file is the living process
  that turns the audit's P0/P1/P2 plan into tracked, verified work items.
- **Status:** Complete (all findings resolved as of 2026-08-22)
- **Source audit:** [../CODEBASE_AUDIT.md](../CODEBASE_AUDIT.md)

---

## Process (how each problem is addressed)

1. **Work one finding at a time.** Move its status below to `In Progress`
   before editing, `Done` only after its **Verification** passes.
2. **Verification is mandatory before `Done`** — each finding lists the exact
   check (e.g. re-grep, run smoke test, compile). No finding closes on intent.
3. **Each fix lands as its own commit** with a conventional message
   (type + scope + imperative), e.g. `fix(security): load checkpoints with weights_only=True`.
4. **Log every change** in the [Log](#log) section with date + finding ID.
5. **Re-run the audit periodically** ([CODEBASE_AUDIT.md](../CODEBASE_AUDIT.md))
   and tick the summary here; close items that no longer apply.

---

## Findings tracking

Legend: `To Do` → `In Progress` → `Done` (or `Wontfix`). Priority tier in parens.

| ID | Area | Sev | Action (commit to make) | Verification | Status |
|---|---|---|---|---|---|
| SEC-1 | Security | High | Add `weights_only=True` to 6 `torch.load` calls | `grep -rn "torch.load" src/` shows `weights_only=True` on all | Done |
| SEC-2 | Security | Med | Drop `allow_pickle=True` in `evaluate.py`/`predict.py` | grep shows no `allow_pickle=True`; eval smoke-run loads NPZ | Done |
| ARCH-1 | Architecture | High | Implement `LOGGING_CHECKPOINT_RULES` in `train.py` OR rescope rules | `train.py` saves full-state + `_last.pt` + JSONL, OR rules updated to match | On Hold |
| TST-1 | Tests | High | Add `tests/conftest.py`, `tests/test_smoke.py`, `pytest.ini` | `pytest` passes | Done |
| CQ-1 | Quality | Med | Replace `/Volumes/...` with configurable `--src`/env root | grep shows no `/Volumes/quangmanh`; smoke `--help` shows arg | Done |
| CQ-2 | Quality | Med | Full RNG seeding (CPU/CUDA/python/dataloader) | grep shows `manual_seed_all`/`generator=`; seed logged | Done |
| DEP-1 | Dependencies | Med | Pin deps / add lockfile; document torch cu124 install | `pip freeze`-based lockfile present; README notes versions | Done |
| DEP-2 | Dependencies | Med | Standardize Python 3.10+ in setup docs | README/SETUP state 3.10+ | Done |
| ARCH-2 | Architecture | Med | Fill `OVERVIEW.md`; create or prune phase/status docs | OVERVIEW has no `[Fill in]`; links resolve | Done |
| PERF-1 | Performance | Low | Set `num_workers>0` + `pin_memory` | grep shows non-zero workers | Done |
| PERF-2 | Performance | Low | Add opt-in AMP (`torch.autocast`) | `--amp` flag present | Done |
| ARCH-4 | Architecture | Low | Gitignore + untrack `.feynman/` | `git check-ignore .feynman/` succeeds; file untracked | Done |

Positive findings (SEC-3, ARCH-3, PERF-3): **no action** — keep as-is.

## Re-audit findings (2026-08-22, main @ 08d20d7) — NEW

Re-audit from `881f8cf` → `08d20d7` verified all 11 closable items above are
genuinely resolved. New findings from the DF40 data-prep expansion:

| ID | Area | Sev | Action | Verification | Status |
|---|---|---|---|---|---|
| SEC-4 | Security | Med | Notebook `01_full_pipeline.ipynb`: fix `torch.load(weights_only=False)` + remove training loop | grep shows `weights_only=True` in notebook; no `optimizer.step`/`loss.backward` | Done |
| TST-2 | Tests | Med | Tests need gitignored `data/splits/*`; fresh clone fails `pytest`; `pytest` not in requirements/.venv | fresh `git clone` + `pytest` passes | Done |
| DEP-1r | Dependencies | Med | Lockfile missing `pandas>=2.0`; global `*.json` ignore hides result JSONs (regression of DEP-1) | `pip freeze`-lock includes pandas; `git ls-files` shows result JSONs | Done |
| CQ-1r | Quality | Med | `/workspace/...` absolute paths in `extract_celeb_df_*.py` (regression of CQ-1) | grep shows no `/workspace/hoangtuan` in `src/` | Done |
| ARCH-3 | Architecture | Low | `outputs/`/`models/` undocumented; `outputs/` duplicates `experiments/results/` | FOLDER_STRUCTURE documents them; single canonical results tree | Done |
| DOC-1 | Docs | Low | Fix `OVERVIEW.md` dead notebook links; update `EVAL_STATUS.md`; reconcile `methods_summary.json` | links resolve; EVAL_STATUS last-updated current | Done |

ARCH-1 resolved 2026-08-22: implemented full-state checkpointing + resume +
`_last.pt` + JSONL history + config in `src/training/train.py` via
`src/utils/run_logger.py`, and updated `LOGGING_CHECKPOINT_RULES.md` to match
reality (removed references to nonexistent `train_model.py`/`checkpoint_utils.py`).

---

## Log

- 2026-08-18: Audit report created and committed (`7c6f75d`); tracker created.
- 2026-08-18: Tracker initialized with all 12 findings as `To Do`.
- 2026-08-18: SEC-1 done (`2f8b595`); SEC-2 done (`a1f9cad`); TST-1 done (`be374a0`).
- 2026-08-18: ARCH-1 deferred by user; P1 started — CQ-1 (hardcoded paths) first.
- 2026-08-18: CQ-1 done (`cb2dcee`); CQ-2 done (`d08f658`).
- 2026-08-18: DEP-1 done (`bae8f1a`); DEP-2 done (`47be5fc`); ARCH-2 done (`588c9fc`). P1 complete.
- 2026-08-18: PERF-1 + PERF-2 done (`d525890`); ARCH-4 done (`d1dc6bf`). P2 complete.
- 2026-08-22: Re-audit `881f8cf` → `08d20d7`. All 11 closable findings verified resolved; added 6 new findings (SEC-4, TST-2, DEP-1r, CQ-1r, ARCH-3, DOC-1) to the tracker as `To Do`.
- 2026-08-22: ARCH-1 fixed — `train.py` upgraded to full-state checkpoints +
  resume + `_last.pt` + JSONL history + config via new `src/utils/run_logger.py`;
  `LOGGING_CHECKPOINT_RULES.md` aligned (dropped `train_model.py`/`checkpoint_utils.py` refs).
- 2026-08-22: SEC-4 fixed — notebook `01_full_pipeline.ipynb` training loop
  replaced with a loader stub; `weights_only=True`.
- 2026-08-22: TST-2 fixed — `pytest` added to requirements; data-prep tests skip
  when splits missing; result JSONs allowlisted in `.gitignore`; `pytest` passes (13/13).
- 2026-08-22: DEP-1r fixed — `requirements.lock.txt` regenerated (pandas, pytest).
- 2026-08-22: CQ-1r fixed — `/workspace/...` defaults replaced with repo-relative
  paths + `DF40_ROOT`/`CELEB_DF_ROOT` env.
- 2026-08-22: ARCH-3 fixed — `outputs/` docs moved to `experiments/results/`;
  `models/` + `outputs/` documented in `FOLDER_STRUCTURE.md`.
- 2026-08-22: DOC-1 fixed — `OVERVIEW.md` notebook links corrected;
  `EVAL_STATUS.md` updated; `methods_summary.json` reconciled (40 methods incl. CelebDFv2).

## Blockers (if any)

- ARCH-1 needs a decision: **implement** the logging/checkpoint rules in
  `train.py`, or **rescore** the rules to match the current simple loop. Both
  paths are valid; pick one before starting that item.

## Decisions

- Use `agents/progress/CODEBASE_AUDIT_STATUS.md` as the single source of truth
  for remediation status — it mirrors the P0/P1/P2 plan in
  [../CODEBASE_AUDIT.md](../CODEBASE_AUDIT.md) one-to-one.
- 2026-08-18: **ARCH-1 deferred** until after P1 (user decision). It remains
  the only open P0 item; revisit before the deadline.

## Next step

- All findings resolved as of the 2026-08-22 re-audit + remediation pass:
  original 11/12 (P0-P2) verified resolved, ARCH-1 implemented, and all 6 new
  re-audit findings (SEC-4, TST-2, DEP-1r, CQ-1r, ARCH-3, DOC-1) fixed.
- Re-run the audit (codebase_audit) on the next session to confirm the tree,
  `pytest` (13/13), and the new checkpoint/resume flow hold.

## Links

- Audit report: [../CODEBASE_AUDIT.md](../CODEBASE_AUDIT.md)
- Rules: [../rules/CODEBASE_AUDIT.md](../rules/CODEBASE_AUDIT.md),
  [../rules/LOGGING_CHECKPOINT_RULES.md](../rules/LOGGING_CHECKPOINT_RULES.md)

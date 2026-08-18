# Codebase Audit — Action Tracker (agents/progress)

- **Title:** Codebase audit remediation tracker (main @ f3d521f → 7c6f75d)
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-18
- **Description:** Tracks resolution of every finding in
  [CODEBASE_AUDIT.md](../CODEBASE_AUDIT.md); this file is the living process
  that turns the audit's P0/P1/P2 plan into tracked, verified work items.
- **Status:** In Progress
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

---

## Log

- 2026-08-18: Audit report created and committed (`7c6f75d`); tracker created.
- 2026-08-18: Tracker initialized with all 12 findings as `To Do`.
- 2026-08-18: SEC-1 done (`2f8b595`); SEC-2 done (`a1f9cad`); TST-1 done (`be374a0`).
- 2026-08-18: ARCH-1 deferred by user; P1 started — CQ-1 (hardcoded paths) first.
- 2026-08-18: CQ-1 done (`cb2dcee`); CQ-2 done (`d08f658`).
- 2026-08-18: DEP-1 done (`bae8f1a`); DEP-2 done (`47be5fc`); ARCH-2 done (`588c9fc`). P1 complete.
- 2026-08-18: PERF-1 + PERF-2 done (`d525890`); ARCH-4 done (`d1dc6bf`). P2 complete.

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

- P1 ✅ and P2 ✅ complete (all 11 of 12 findings resolved).
- **Only open item: ARCH-1** (P0, on hold) — revisit before the deadline
  (implement `LOGGING_CHECKPOINT_RULES` in `train.py` OR rescope the rules).
- Re-run the audit (codebase_audit) to confirm the tree/verifications hold.

## Links

- Audit report: [../CODEBASE_AUDIT.md](../CODEBASE_AUDIT.md)
- Rules: [../rules/CODEBASE_AUDIT.md](../rules/CODEBASE_AUDIT.md),
  [../rules/LOGGING_CHECKPOINT_RULES.md](../rules/LOGGING_CHECKPOINT_RULES.md)

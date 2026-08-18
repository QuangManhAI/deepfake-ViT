# CODEBASE_AUDIT.md
Run this procedure before starting any task that touches more than
one file, or at the start of a new session. Goal: catch drift between
what the docs claim and what the code actually does, BEFORE acting on it.

## Steps
1. List all files under `src/`, `data/`, `agents/` — diff against
   `FOLDER_STRUCTURE.md`. Note anything present but undocumented,
   or documented but missing.
2. For every function/class name referenced in `agents/**/*.md`, grep for
   it in `src/` or `data/`. Flag any that don't exist or whose signature changed.
3. Cross-check `agents/progress/*_STATUS.md` against real artifacts —
   e.g. if `TRAINING_STATUS.md` says "Done," does a checkpoint file
   actually exist? If not, flag it.
4. Check naming in recently changed files against `NAMING_CONVENTION.md`.
5. Check for orphaned experiment outputs (runs with no corresponding
   entry in progress files).

## Output format
Summarize findings in under 150 words as a short list:
- ✅ No discrepancies found, OR
- ⚠️ [file/claim] says X, but code shows Y

## Hard rule
Do NOT proceed with the requested task until discrepancies are either
resolved or the human explicitly says "proceed anyway." This audit is
cheap compared to the cost of building on a wrong assumption.

## When to run this
- Start of a new session
- Before any multi-file refactor or pipeline change
- NOT before every trivial single-line edit (too expensive to run
  constantly — use judgment)

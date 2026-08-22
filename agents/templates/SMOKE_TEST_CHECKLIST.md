# SMOKE_TEST_CHECKLIST.md
Every script must pass a smoke test before being run "for real"
(full data, full epochs, production config). This is cheap insurance
against burning an hour of compute on a bug in line 3.

## What counts as a smoke test
- Data scripts: run on a tiny sample (e.g. 1% or 100 rows), assert
  output shape/schema/dtypes are as expected, no NaNs where unexpected
- Training scripts: 1 epoch (or a handful of steps), tiny batch size,
  assert loss is finite and decreasing (or at least not NaN/exploding)
- Eval scripts: run on a small held-out slice, assert metric functions
  return values in expected range (e.g. accuracy in [0,1])
- Any script: assert it runs end-to-end without crashing, on a
  reduced input, in under ~1 minute

## Agent rule
Before running any script "for real" (full dataset / full training),
the agent must either:
1. Confirm a smoke test already passed for this exact script version, or
2. Run the smoke test first and show the result before proceeding

## Not required for
- Read-only inspection scripts (EDA, `df.info()`, `df.describe()`, `df.shape`)
- Trivial one-line utility scripts with no side effects

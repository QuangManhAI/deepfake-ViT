# DATA_PREP_STATUS.md — Data Preparation

- **Title:** Data Preparation (DF40)
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-18
- **Description:** Status of DF40 build/split/transform work.
- **Status:** In Progress
- **Phase doc:** [../phases/DATA_PREP.md](../phases/DATA_PREP.md)

## Log

- 2026-08-18: Source root made configurable via `DF40_ROOT` (CQ-1).

## Blockers (if any)

- Large raw data (~74.7 GB) is not committed; must be present locally via
  `DF40_ROOT` before build scripts run.

## Decisions

- Binary real/fake only; image size 256×256.

## Next step

- Generate final train/val/test split CSVs from the built subsets.

## Links

- Phase doc: [../phases/DATA_PREP.md](../phases/DATA_PREP.md)
- Overview: [../OVERVIEW.md](../OVERVIEW.md)

# DATA_PREP_STATUS.md — Data Preparation

- **Title:** Data Preparation (DF40 Benchmark & Multi-Method Evaluation)
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-22
- **Description:** Status of DF40 build/split/transform work.
- **Status:** Done
- **Phase doc:** [../phases/DATA_PREP.md](../phases/DATA_PREP.md)

## Log

- 2026-08-18: Source root made configurable via `DF40_ROOT` (CQ-1).
- 2026-08-21: External `/workspace/data` preserved as read-only.
- 2026-08-22: Upgraded `src/data/prepare_df40_splits.py` to generate complete per-method test sets (`test_<method>_balanced.csv`, `test_<method>_full.csv`, `test_<method>_detailed.csv`, `benchmark_test_<method>_balanced.csv`) across all active DF40 deepfake generation methods under `data/splits/methods/` (195 files).
- 2026-08-22: Generated 100% reproducible identity-disjoint train/val/test splits (70/15/15 ratio) across 23,237 unique identities with verified 0% identity leakage.
- 2026-08-22: Safely merged 22,418 disjoint FaceForensics++ Real frames (excluding 298 held-out test/val video folders) into high-scale training pools (`train_pool_693k.csv` with 643k images and `train_combined_balanced.csv` with 40.3k images).
- 2026-08-22: Extracted 10,336 clean Real frames ($256 \times 256$) from 690 Celeb-DF-v2 training videos (`data/processed/celeb_df_extracted/`) and 2,590 test frames from 518 official test videos (`data/processed/celeb_df_test_extracted/`).
- 2026-08-22: Upgraded high-scale 1:1 balanced training pool (`train_combined_balanced.csv`) to 58,958 images (20,219 FF++ Real + 9,268 Celeb-DF Real + 29,471 DF40 Fake) with exact 1:1 balance.
- 2026-08-22: Unified master evaluation suite (`test_full.csv`) to 32,281 images covering all 40 DF40 deepfake generation methods and official Celeb-DF-v2 test benchmark.
- 2026-08-22: Automated unit testing in `tests/test_data_prep.py` verified 100% (7/7 tests passed).

## Blockers (if any)

- None.

## Decisions

- Binary real/fake classification only; image size 256×256 with standard ImageNet normalization.
- External source root (`/workspace/data`) remains strictly READ-ONLY.
- Zero-leakage protocol strictly enforced at the Subject/Identity/Video level across all training, validation, and testing partitions (22,237 identities).
- All 4 data sources unified into a single coherent training and testing pipeline.

## Next step

- Proceed to Model Architecture & Training / Evaluation phases (Phase 2 & Phase 3).

## Links

- Master Summary: [../DATA_SPLIT_SUMMARIZE.md](../DATA_SPLIT_SUMMARIZE.md)
- Technical Report: [../DATA_PREP_SUMMARY_REPORT.md](../DATA_PREP_SUMMARY_REPORT.md)
- Phase doc: [../phases/DATA_PREP.md](../phases/DATA_PREP.md)
- Overview: [../OVERVIEW.md](../OVERVIEW.md)
- Unit Tests: [../../tests/test_data_prep.py](../../tests/test_data_prep.py)




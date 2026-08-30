# Primary Dataset Protocol

**Version:** identity_clean_v1
**Source:** test_data_v3
**Strategy:** identity-disjoint + exact-duplicate-aware
**Random seed:** 42

## Files

- `train.csv` / `train_detailed.csv`
- `val.csv` / `val_detailed.csv`
- `test.csv` / `test_detailed.csv`
- `protocol_metadata.json`

## Sizes

- Train: 20991
- Val:   4498
- Test:  4499
- Total: 29988

## Constraints

- identity(train) ∩ identity(val) = ∅
- identity(train) ∩ identity(test) = ∅
- identity(val) ∩ identity(test) = ∅
- exact-duplicate overlap across splits = 0

## Known Limitations

- **Video/source overlap** remains across splits (1,509 train↔val, 1,542 train↔test, 930 val↔test).
- **Near-duplicate overlap** is documented but not removed (4,215 cross-split groups).
- **Class imbalance** is ~24.5:1 fake:real.

## Reproducibility

This protocol was generated from:
- `src/data/build_clean_splits.py`
- `src/data/validate_protocol.py`
- `data/splits_identity_clean/`

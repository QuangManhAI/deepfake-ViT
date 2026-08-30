# DF40 `test_data_v3` Dataset Source & Structure Verification

## Summary

The correct authoritative dataset for the main DINOv3 ViT pipeline is **`test_data_v3`** from the Hugging Face dataset repo:

**`ManhQuangAI/df40-test-data-v3`**

It contains a single `test_data_v3.zip` (~4.53 GB / 4.22 GiB) which, when unzipped, produces the exact `test_data_v3/` tree that `src/data/prepare_df40_splits.py` requires.

## 1. What `prepare_df40_splits.py` Expects

### Expected root directory

Default: `data/raw` (or override with `DF40_ROOT` env, or `--df40-root`)

### Expected subdirectories / files

```
$DF40_ROOT/
├── test_data_v3/
│   ├── manifest.csv
│   ├── real/
│   │   └── <frame_id>.jpg   (1,177 real images)
│   ├── <method1>/
│   │   └── fake/
│   │       └── <frame_id>.jpg
│   ├── <method2>/
│   │   └── fake/
│   │       └── <frame_id>.jpg
│   ... 40 fake methods total
│
└── DF40_train_manifest.csv   (optional, only for Protocol 2 large training pool)
```

The script also looks for optional Celeb-DF-v2 / FaceForensics++ data if the full combined pool (Protocol 2) is being built, but the core requirement for generating the 30,691-sample identity-disjoint splits is only `test_data_v3/`.

### Expected `test_data_v3/manifest.csv` format

Columns (exact fields used by `load_test_manifest`):

| Column | Meaning |
|--------|---------|
| `path` | Relative image path inside `test_data_v3/` (e.g., `real/0000.jpg` or `insight/fake/0000.jpg`) |
| `label` | `0` for real, `1` for fake |
| `method` | Method name, or `real` |
| `identity` | Unique identity token |
| `domain` | Source domain (`real`, `fake`, `cdc`, `ffc`, etc.) |
| `video` | Video/source identifier |

### Expected naming convention

- Real images: `test_data_v3/real/<basename>`
- Fake images: `test_data_v3/<method>/fake/<basename>`
- Method string in `manifest.csv` matches the `<method>` folder name

### Expected number/type of methods

- **40 fake methods** as listed in `split_info.json`
- **Real:** single `real/` folder shared by all methods
- 1,177 unique real faces
- 30,691 total images

### Expected CSV output schema

`prepare_df40_splits.py` generates:

- `data/splits/train.csv`, `val.csv`, `test.csv`  → `path,label`
- `data/splits/train_detailed.csv` etc.         → `path,label,method,identity,domain,video`
- `data/splits/methods/*.csv`                   → `path,label`
- `data/splits/split_info.json`                 → statistics
- `data/splits/methods_summary.json`            → method counts

## 2. Project Documentation

### README.md

Lists the DF40 training corpus from HF:

```bash
hf download ManhQuangAI/DF40_train --repo-type dataset --local-dir data/raw/DF40
```

This is the **full DF40 training corpus**, NOT the `test_data_v3` evaluation set. The `test_data_v3` download is documented elsewhere.

### MODELS.md

Explicitly documents the test data source:

- **Dataset repo:** `ManhQuangAI/df40-test-data-v3`
- **File:** `test_data_v3.zip`
- **Size:** 4.53 GB
- **Commands:**

```bash
pip install -U "huggingface_hub[cli]"
hf download ManhQuangAI/df40-test-data-v3 --repo-type dataset --include "test_data_v3.zip" --local-dir .
unzip -o test_data_v3.zip -d .
ls test_data_v3/           # manifest.csv + real/ + <method>/ dirs
```

### RUNPOD.md

Same source and size:

```bash
hf download ManhQuangAI/df40-test-data-v3 --repo-type dataset --include test_data_v3.zip
unzip test_data_v3.zip -d .
```

### src/utils/push_dataset_to_hub.sh

This script was used to **push** `test_data_v3.zip` to the dataset repo. It confirms:

- Repo: `ManhQuangAI/df40-test-data-v3`
- Local zip: `test_data_v3.zip`
- Zip size: ~4.2 GB

### src/data/restructure_test_data_v3.py

Generates `test_data_v3/` from `test_data_v2/` using hard links. Confirms the same directory structure:

```
test_data_v3/
  manifest.csv
  real/
  <method>/fake/
```

## 3. Git History Findings

### Where `test_data_v3` came from

- `src/data/restructure_test_data_v3.py` — commit `4ec91b8` ("Add benchmark scripts and test datasets...")
- `src/utils/push_dataset_to_hub.sh` — commit `4ec91b8`
- `src/data/prepare_df40_splits.py` — commit `61ae0ca` (added the consumer script)
- `data/splits/split_info.json` and `data/splits/methods_summary.json` — commit `61ae0ca`

The dataset files themselves were **never committed** to Git. They are excluded by `.gitignore`:

```gitignore
*.zip
*.csv
data/raw/DF40/
```

The CSV split files referenced by `split_info.json` were generated on a local/remote workspace and are not in the repository.

## 4. Local Machine Search

Searched the following locations for `test_data_v3`, `DF40`, or `DF40_train`:

- `/Users/pickapu/Documents`
- `/Users/pickapu/Downloads`
- `/Users/pickapu/Datasets`
- `/Users/pickapu/data`

**Result:** No existing local copy found.

## 5. Hugging Face Source Verification

Using the project `.venv/hf` CLI:

```bash
.venv/bin/hf datasets ls ManhQuangAI/df40-test-data-v3
```

Output:

```
     2504  2026-08-17 15:23:08  .gitattributes
4532994873  2026-08-17 15:49:18  test_data_v3.zip
```

- **Repo exists:** ✓
- **File:** `test_data_v3.zip`
- **Decimal size:** 4.53 GB
- **Binary size:** 4.22 GiB
- **Last modified:** 2026-08-17 15:49:18
- **Public/Private:** Private (requires HF token or access grant)

## 6. Disk Space Check

Current disk:

```
Filesystem      Size    Used   Avail Capacity
/dev/disk3s5   460Gi   134Gi   303Gi    31%
```

**Available free space:** 303 GiB

**Required for `test_data_v3`:**

- Zip file: ~4.22 GiB
- Unzipped: approximately the same or slightly larger (estimated 5–10 GiB total for zip + unzipped)
- **Conclusion:** more than enough space.

## 7. Download Method

The exact, verified download commands are:

```bash
# 1. Ensure the Hugging Face CLI is installed and authenticated
.venv/bin/pip install -U "huggingface_hub[cli]"
.venv/bin/hf login              # or set HF_TOKEN env

# 2. Download the test dataset zip to the project root
.venv/bin/hf download ManhQuangAI/df40-test-data-v3 \
    --repo-type dataset \
    --include "test_data_v3.zip" \
    --local-dir .

# 3. Unzip
unzip -o test_data_v3.zip -d .

# 4. Verify contents
ls test_data_v3/                # should show manifest.csv, real/, <method>/ dirs
wc -l test_data_v3/manifest.csv # should be ~30,692 lines (header + 30,691 images)
```

## 8. Required Before `prepare_df40_splits.py`

After the steps above, run:

```bash
.venv/bin/python src/data/prepare_df40_splits.py --seed 42
```

This will:

1. Load `test_data_v3/manifest.csv`
2. Verify 50 sample images decode
3. Generate identity-disjoint train/val/test splits
4. Generate 40 method-specific test sets
5. Generate convenience splits (`train_insight.csv`, `val_insight.csv`, etc.)
6. Write `data/splits/split_info.json` and `data/splits/methods_summary.json`
7. Optionally build the full combined training pool if `DF40_train_manifest.csv` and FF++ data are present

## Final Report

| Item | Value |
|------|-------|
| **Authoritative dataset** | `test_data_v3` |
| **Source** | Hugging Face dataset `ManhQuangAI/df40-test-data-v3` |
| **Version** | `test_data_v3.zip` uploaded 2026-08-17 |
| **Expected local path** | `test_data_v3/` in project root (or `data/raw/test_data_v3/` if `DF40_ROOT=data/raw`) |
| **Expected structure** | `test_data_v3/manifest.csv`, `test_data_v3/real/`, `test_data_v3/<method>/fake/` |
| **Approximate size** | 4.53 GB / 4.22 GiB (zip); ~5–10 GiB total after unzip |
| **Current local status** | Not present |
| **Required download command** | `hf download ManhQuangAI/df40-test-data-v3 --repo-type dataset --include "test_data_v3.zip" --local-dir .` |
| **Required disk space** | ~10 GiB (zip + unzip) — 303 GiB available, so OK |
| **Next step** | Download and unzip `test_data_v3.zip`, then run `prepare_df40_splits.py` |

**Note:** The repo is private. The user must have a Hugging Face token with access to `ManhQuangAI/df40-test-data-v3`, or the dataset must be made public, or an access request must be approved before the download can succeed.
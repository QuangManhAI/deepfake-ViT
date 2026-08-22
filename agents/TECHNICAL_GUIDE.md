# TECHNICAL_GUIDE.md — Current State, Training/Fine-tuning/Experiment Workflows & Data Quality

- **Motivation/Background**: The project has matured (full DF40 data pipeline,
  full-state checkpointing, tests), but there is no single up-to-date
  reference that explains how to *actually run* training, fine-tuning, and
  experiments, and how to keep noisy/imbalanced data from hurting results.
- **Purpose**: One authoritative operator's guide covering (a) the current
  architecture and entry points, (b) exact commands for training,
  fine-tuning, and experiments, and (c) concrete data cleaning,
  preprocessing, and validation strategies to protect training stability and
  generalization on the "clumsy"/noisy DF40 dataset.
- **Overview Pipeline**: Derived from a codebase audit of `src/training`,
  `src/eval`, `src/experiments`, `src/data`, `data/splits`, and the agent
  rulebase at commit `61ae0ca`.
- **Detailed Plan**: §1 current state; §2 conventions that gate all runs;
  §3 training workflows; §4 fine-tuning workflows; §5 experiment workflows;
  §6 data quality problem statement; §7 data cleaning; §8 preprocessing;
  §9 validation & leakage; §10 run verification checklist.
- **References**: [src/training/train.py](../src/training/train.py),
  [src/training/finetune_lora.py](../src/training/finetune_lora.py),
  [src/training/finetune_compare.py](../src/training/finetune_compare.py),
  [src/utils/run_logger.py](../src/utils/run_logger.py),
  [src/data/prepare_df40_splits.py](../src/data/prepare_df40_splits.py),
  [LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md),
  [RESULTS_REPORTING.md](rules/RESULTS_REPORTING.md),
  [DATA_PREP_SUMMARY_REPORT.md](DATA_PREP_SUMMARY_REPORT.md),
  [tests/test_data_prep.py](../tests/test_data_prep.py).

---

## Table of Contents

- [1. Current State](#1-current-state)
- [2. Conventions That Gate Every Run](#2-conventions-that-gate-every-run)
- [3. Training Workflow](#3-training-workflow)
- [4. Fine-Tuning Workflows](#4-fine-tuning-workflows)
- [5. Experiment Workflows](#5-experiment-workflows)
- [6. Data Quality: Problem Statement](#6-data-quality-problem-statement)
- [7. Data Cleaning Strategies](#7-data-cleaning-strategies)
- [8. Preprocessing & Augmentation](#8-preprocessing--augmentation)
- [9. Validation & Leakage Prevention](#9-validation--leakage-prevention)
- [10. Run Verification Checklist](#10-run-verification-checklist)

---

## 1. Current State

- **Task**: binary face-deepfake classification (real=0 / fake=1) at 256×256.
- **Models**: DINOv3 ViT-S/16 (`src/models/dinov3_vit.py`), matched-size
  ConvNeXt-Tiny (`src/models/dinov3_convnext.py`), LoRA adapter
  (`src/models/lora.py`).
- **Backbones**: gitignored `.safetensors` under
  `experiments/checkpoints/weights/` (see [MODELS.md](../MODELS.md) §5 for
  download).
- **Data**: 4 unified sources — FF++ real (22.4k), Celeb-DF-v2 real (10.3k),
  DF40 train pool (692k fake), and `test_data_v3` (30,691 benchmark images).
  Identity-disjoint 70/15/15 splits (22,237 identities, 0% leakage) plus a
  high-scale 1:1 balanced pool (`train_combined_balanced.csv`, 58,958) and
  per-method test suites (`data/splits/methods/`, 200 files).
- **Checkpointing**: `train.py` now implements full-state checkpoints
  (model+optimizer+scheduler+RNG+history) + `--resume`/`--force-resume` via
  [src/utils/run_logger.py](../src/utils/run_logger.py). `finetune_*` are
  still best-state only (legacy).
- **Tests**: 13 pass (`pytest`); data-prep tests auto-skip if splits are not
  generated.
- **Headline result**: ~95.2% test acc (linear probe) ViT on `test_data_v3`;
  target > 97.5% via the EXP-01 optimization plan
  ([EXP_01_ACCURACY_OPTIMIZATION_PLAN.md](experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md)).

---

## 2. Conventions That Gate Every Run

These are project rules — not suggestions:

1. **No training in notebooks.** Notebooks load artifacts produced by
   scripts. The only training entry points are the `src/training/*.py`
   scripts ([LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md) §1).
2. **Run from the repo root.** Scripts resolve `data/splits/*`, model paths,
   and `sys.path` relative to the repo root.
3. **Every result carries 5W1H** context (what/why/when/where/who/how) —
   [RESULTS_REPORTING.md](rules/RESULTS_REPORTING.md). Never report a bare number.
4. **Seed everything.** All training scripts call `set_seed`; for
   byte-exact reproducibility set `PYTHONHASHSEED` before interpreter start.
5. **`num_workers` + `--amp` are opt-in flags**; on MPS, AMP auto-disables.
6. **Raw data is read-only.** `data/raw/*` and `/workspace/data/*` are never
   written by scripts; all outputs go to `data/processed/`, `data/splits/`,
   `experiments/results/`, and `experiments/runs/`.
7. **Smoke-test before long runs.** Verify a tiny slice or a few epochs
   completes before committing GPU hours.

---

## 3. Training Workflow

### 3.1 Standard full fine-tune (`train.py`)

Full-state checkpointing with resume — the recommended path.

```bash
# From repo root
.venv/bin/python src/training/train.py \
    --train-csv data/splits/train.csv \
    --val-csv   data/splits/val.csv \
    --test-csv  data/splits/test.csv \
    --model     experiments/checkpoints/weights/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors \
    --epochs 5 --batch-size 32 --num-workers 4 --amp \
    --run-name dinov3_finetuned
```

- **Outputs** (created under `experiments/results/checkpoints/<ts>_<run_name>/`):
  - `checkpoints/<run>_best.pt`, `checkpoints/<run>_last.pt` (full state)
  - `logs/<run>.log`, `metrics/<run>_history.jsonl`, `metrics/<run>_config.json`
  - a legacy best-state copy `experiments/results/checkpoints/dinov3_finetuned.pt`
    so the old eval scripts keep working unchanged.

**Resume after interruption** (at most one epoch lost):

```bash
# exact resume from _last.pt
.venv/bin/python src/training/train.py ... --resume
# rewind to the best epoch with a fresh budget
.venv/bin/python src/training/train.py ... --force-resume
```

**Important**: `--resume` reuses the latest `*_<run_name>/` directory, so run
the same `--run-name` to continue that run. Using a new run name starts fresh.

### 3.2 Choosing the split manifest

| Goal | Use |
|---|---|
| Benchmark generalization | `train.csv` / `val.csv` / `test.csv` (identity-disjoint, imbalanced 24:1) |
| Rapid, balanced convergence | `train_balanced.csv` / `val_balanced.csv` / `test_balanced.csv` (1:1) |
| High-scale robustness | `train_combined_balanced.csv` (58.9k, 1:1) |
| Per-method eval | `data/splits/methods/test_<method>_balanced.csv` |

### 3.3 Model selection

- ViT backbone default: `experiments/checkpoints/weights/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors`
- `train.py` default `--model` points at `model.safetensors` (small) — pass
  the full path explicitly to use the larger ViT-S/16 Plus.

---

## 4. Fine-Tuning Workflows

### 4.1 LoRA (`finetune_lora.py`) — parameter-efficient

Frozen backbone + low-rank adapters on q/v projections + head. Best when
compute is tight or you want to preserve generic features.

```bash
.venv/bin/python src/training/finetune_lora.py \
    --train-csv data_train/train.csv \
    --val-csv   data_train/val.csv \
    --lora-rank 16 --lora-alpha 32 \
    --epochs 4 --batch-size 32 --num-workers 4 --amp \
    --output-dir experiments/results/finetune
```

- Saves `vit_lora_finetuned.pt` (includes `lora_config` so eval can rebuild
  the adapter) + `vit_lora_finetune_report.json`.
- **Note**: best-state only; no `--resume`. Treated as legacy until migrated.

### 4.2 Matched ViT-vs-CNN fine-tune (`finetune_compare.py`)

Fine-tunes the same classifier head on both backbones for an apples-to-apples
comparison under identical conditions.

```bash
.venv/bin/python src/training/finetune_compare.py --model-type vit \
    --train-csv data/splits/train_insight.csv \
    --val-csv data/splits/val_insight.csv \
    --test-csv data/splits/test_insight.csv \
    --epochs 4 --num-workers 4 --amp
.venv/bin/python src/training/finetune_compare.py --model-type cnn ... # same flags
```

- Outputs: `vit_finetuned.pt` / `cnn_finetuned.pt` + per-type JSON reports.
- **Note**: best-state only; no `--resume`.

---

## 5. Experiment Workflows

### 5.1 Linear-probe comparison (`compare_models.py`)

Frozen backbones + logistic regression (standardized, balanced). Fast,
low-VRAM, good first signal of representation quality.

```bash
.venv/bin/python src/experiments/compare_models.py \
    --train-csv data/splits/train.csv \
    --test-csv  data/splits/test.csv \
    --output experiments/results/comparison_report.json
```

### 5.2 Identity-disjoint linear-probe eval (`eval_identity_disjoint.py`)

The canonical eval: split by identity so no subject is seen in train; reports
overall + per-domain + per-method + paired-only. Streams features to memmap to
bound RAM.

```bash
.venv/bin/python src/eval/eval_identity_disjoint.py --model vit --tag test_data_v3
.venv/bin/python src/eval/eval_identity_disjoint.py --model cnn --tag test_data_v3
```

### 5.3 Evaluate a fine-tuned checkpoint

```bash
# ViT-S/16 full fine-tune (needs the legacy .pt from train.py)
.venv/bin/python src/eval/eval_finetuned.py --device cuda
# 40-method suite on the 30.6k benchmark
.venv/bin/python src/eval/eval_df40_all_methods.py
# ViT vs CNN on DF40 (needs raw FF++/blendface/ddim frames)
.venv/bin/python src/eval/eval_df40_vit_cnn.py
# threshold analysis
.venv/bin/python src/eval/analyze_threshold.py --ckpt experiments/checkpoints/finetune/vit_lora_finetuned.pt
```

### 5.4 Feature extraction & attention

```bash
# Dump CLS-token features to NPZ for downstream probing
.venv/bin/python src/experiments/extract_features.py
# Visualize ViT attention (CLS→patches, skip registers)
.venv/bin/python src/experiments/visualize_attention.py \
    --model experiments/checkpoints/weights/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors \
    --data-dir test_data_v3/real --output-dir experiments/plots/attention
```

### 5.5 Running the EXP-01 accuracy-optimization plan

The 6 pillars (50:50 batch balancing, LLRD, artifact-preserving
augmentations, label smoothing, threshold tuning, TTA + ViT/CNN ensemble)
are documented in
[EXP_01_ACCURACY_OPTIMIZATION_PLAN.md](experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md).
Execute the fine-tune via `train.py` with those hyperparameters, then the
`analyze_threshold.py` + eval scripts for TTA/ensemble.

---

## 6. Data Quality: Problem Statement

The dataset is "clumsy" in ways that measurably threaten training stability
and generalization:

1. **Extreme class imbalance** — the identity-disjoint splits are ~24:1
   fake:real (20,019 fake vs 834 real in train). Unweighted CE collapses to
   always-fake, near-zero real accuracy.
2. **Noise & artifacts** — face crops may contain partial faces, heavy
   compression (JPEG blocking), blur, overlays, watermark/caption text, and
   inconsistent face centering across the 40 methods.
3. **Method-domain shortcuts** — models can latch onto color/compression
   artifacts rather than manipulation seams; this overfits to seen methods
   and fails on unseen (zero-shot) methods.
4. **Identity leakage risk** — the same person appearing in train and test
   inflates accuracy. Mitigated by identity-disjoint splits, but must be
   re-verified whenever new data is added.
5. **Broken/empty/corrupt images** — missing files, zero-byte or corrupt
   PNGs break DataLoader batches non-deterministically.

---

## 7. Data Cleaning Strategies

Goal: keep only decodable, correctly-labeled, leakage-free samples before
training.

1. **Verify decodability & non-emptiness.** The split generator already
   checks a random sample (`verify_split_integrity` in
   [prepare_df40_splits.py](../src/data/prepare_df40_splits.py)). Before long
   runs, sweep **all** train rows and drop images that fail to decode, are
   zero-byte, or have unexpected channels/dims. Add a durable check to
   `src/data` (extend `tests/test_data_prep.py`) so it stays enforced.
2. **Label sanity.** For each method, cross-check `label` vs the source
   manifest; flag samples whose `method` column is `real` but live under a
   fake method dir (and vice-versa). Report the counts, don't silently fix.
3. **Deduplicate.** Use SHA-256 of image bytes (or perceptual hash) to find
   exact/near-duplicate frames; drop duplicates across the fake pool so one
   method can't dominate by sheer replication.
4. **Detect near-black / saturated / constant images.** Compute per-image
   std-dev and entropy; drop/flag images whose variance is ~0 (blank frames
   from failed extraction).
5. **Filter by method confidence.** If a method folder has many undecodable
   or zero-variance frames, either clean it or down-weight it in the
   manifest rather than training on garbage.
6. **Keep the manifest as the single source of truth.** Do not edit CSVs
   ad-hoc; regenerate splits from a cleaned manifest so `data/splits/*`
   remain reproducible.

> **Rule**: any cleaning step must be reproducible (a script + seed), never
> a one-off manual edit, and its output must be re-validated by
> `tests/test_data_prep.py`.

---

## 8. Preprocessing & Augmentation

### 8.1 Fixed preprocessing (never optional)

- Resize to 256×256.
- Convert to RGB (drop alpha).
- ImageNet normalize (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).
- `ToTensor()` after PIL load.

### 8.2 Regularization & imbalance handling (protects stability)

- **Class weighting or balanced sampling.** Either `CrossEntropyLoss(weight=...)`
  (inverse-frequency, as `finetune_*` do) or a `WeightedRandomSampler` for
  50:50 batches (EXP-01 Pillar 1). This is the single most important fix for
  the 24:1 imbalance.
- **Label smoothing** (`eps=0.05`) to curb overconfident logits and improve
  zero-shot robustness (EXP-01 Pillar 4).
- **Layer-wise LR decay (LLRD)** `gamma=0.80` — low LR on early layers so
  generic features aren't destroyed while adapting deep layers (EXP-01
  Pillar 2). Protects against catastrophic forgetting on small real class.

### 8.3 Artifact-preserving augmentation (helps noisy data without removing seams)

Keep augmentation **subtle** so it doesn't destroy the manipulation seams
the model must learn:

- `RandomHorizontalFlip(p=0.5)`
- `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)`
- `GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))`
- Optional `RandomJPEGCompression(quality=(75, 95))` to mimic social-media
  recompression noise.

> Avoid heavy geometric aug (large rotations/crops) — it can wash out the
> subtle fake/real cues. The current `TRAIN_TF` in the scripts already
> applies the mild version.

### 8.4 Deterministic data pipeline

- Set `num_workers>0` + `pin_memory=True` (already the default) for speed.
- Seed loaders via the main-process seed (`set_seed`) so shuffling is
  reproducible.
- If exact reproducibility matters, set `PYTHONHASHSEED` before launch.

---

## 9. Validation & Leakage Prevention

1. **Split at the identity level, not the image level.** The pipeline groups
   by `identity` key (22,237 subjects) before splitting 70/15/15, and asserts
   `train ∩ val ∩ test = ∅` in
   [tests/test_data_prep.py](../tests/test_data_prep.py). Re-run those tests
   after any manifest change.
2. **Verify identity-disjointness on every new dataset.** When adding a new
   source (e.g. a new Celeb-DF split), check that its video/identity keys do
   not overlap held-out test/val identities (the extractors already exclude
   held-out folders).
3. **Track seen vs unseen methods.** The benchmark distinguishes 31 seen vs
   9 zero-shot methods. Report per-method detection separately — an
   aggregate number hides zero-shot collapse.
4. **Use a paired-only check.** Evaluate only on identities that have both
   real and fake images (`eval_identity_disjoint.py` "paired-only"); this is
   the strictest generalization signal.
5. **Hold the test set sacred.** Train/val are for model selection; the test
   split is evaluated only at the end. Do not tune hyperparameters on the
   test split.
6. **Watch for "easy" shortcuts.** If validation accuracy is implausibly high
   (near 1.0), suspect a shortcut (compression artifacts, a method whose
   images are trivially separable) or leakage — investigate with per-method
   and per-domain breakdowns before trusting the number.

---

## 10. Run Verification Checklist

Before and after every run, confirm:

- [ ] Repo root; venv active (`source .venv/bin/activate`).
- [ ] Required split CSVs exist; `pytest` passes (or data-prep tests skip
      because splits aren't generated yet — expected on fresh clone).
- [ ] Backbone `.safetensors` present under
      `experiments/checkpoints/weights/`.
- [ ] No training in notebooks — only `src/training/*.py` scripts.
- [ ] Seed set; (optional) `PYTHONHASHSEED` exported.
- [ ] `--num-workers` and `--amp` chosen; device correct
      (`auto`/`cuda`/`mps`/`cpu`).
- [ ] Smoke run: a few epochs or a capped dataset completes without errors.
- [ ] `train.py` writes `_best.pt` + `_last.pt` + log + history + config; a
      `--resume` re-run continues from the same run dir.
- [ ] Result reported with 5W1H; artifacts in `experiments/results/`.

---

## Self-review checklist

- [x] All 5 header fields present
- [x] TOC anchors resolve
- [x] Every file/script path is a working cross-reference link (relative to `agents/`)
- [x] Commands verified against actual `--help` output and source
- [x] Dates `YYYY-MM-DD`; `---` separators between major sections

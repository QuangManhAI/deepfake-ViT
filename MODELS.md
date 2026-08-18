# MODELS.md — Download & Test the Published Models

- **Motivation/Background**: The teammate's weights are not in the git repo
  (they are gitignored); they are published on Hugging Face Hub. This guide
  covers downloading the published models + test data and testing them locally.
- **Purpose**: One self-contained runbook to reproduce model evaluation on any
  machine (written for macOS).
- **Overview Pipeline**: Clone repo → create env → download 3 pretrained
  backbones from HF → download `test_data_v3` → run eval/experiment scripts.
- **Detailed Plan**: §1 availability; §2 prerequisites; §3 code; §4 env;
  §5 models; §6 test data; §7 test commands; §8 fine-tuned checkpoints.
- **References**: `src/eval/*`, `src/experiments/*`, HF
  `ManhQuangAI/dinov3-deepfake-detection`, `ManhQuangAI/df40-test-data-v3`,
  [README.md](README.md), [RUNPOD.md](RUNPOD.md).

---

## Table of Contents

- [1. What Is Actually Published](#1-what-is-actually-published)
- [2. Prerequisites (Mac)](#2-prerequisites-mac)
- [3. Get the Code](#3-get-the-code)
- [4. Environment Setup](#4-environment-setup)
- [5. Download the Pretrained Models](#5-download-the-pretrained-models)
- [6. Download the Test Data](#6-download-the-test-data)
- [7. Test the Models](#7-test-the-models)
- [8. Fine-Tuned Checkpoints (Not Published)](#8-fine-tuned-checkpoints-not-published)

---

## 1. What Is Actually Published

The teammate's GitHub repo (`github.com/zombieTDV/deepfake-ViT`) contains no
weights (`.safetensors`/`.pt` are gitignored). The published artifacts live on
**Hugging Face Hub**:

- **Model repo** `ManhQuangAI/dinov3-deepfake-detection` — **3 pretrained
  backbones only** (no fine-tuned checkpoints):
  - `models/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors` (ViT)
  - `models/dinov3_next_cnn/model-2.safetensors` (ConvNeXt CNN)
  - `models/dinov3_small/model.safetensors` (DINOv3 small)
- **Dataset repo** `ManhQuangAI/df40-test-data-v3` — `test_data_v3.zip` (4.53 GB).

> **Fine-tuned `.pt` checkpoints are NOT published.** `dinov3_finetuned.pt`,
> `vit_finetuned.pt`, `cnn_finetuned.pt`, `vit_lora_finetuned.pt` are gitignored
> and were not uploaded. Anything requiring them (see §8) needs the files from
> the teammate or a retrain.

## 2. Prerequisites (Mac)

```bash
# Python 3.10/3.11 recommended
python3 --version

brew install git git-lfs && git lfs install
```

## 3. Get the Code

```bash
git clone https://github.com/zombieTDV/deepfake-ViT.git
cd deepfake-ViT
```

## 4. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

# On Mac, plain pip install of torch/torchvision is correct (no CUDA index):
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
# Optional pinned: pip install -r requirements.lock.txt
```

## 5. Download the Pretrained Models

Create the expected folders and pull each file to the exact path the code
expects:

```bash
mkdir -p experiments/checkpoints/weights/dinov3-vits16plus-pretrain-lvd1689m \
         experiments/checkpoints/weights/dinov3_next_cnn \
         experiments/checkpoints/weights/dinov3_small

curl -L -o experiments/checkpoints/weights/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors \
  "https://huggingface.co/ManhQuangAI/dinov3-deepfake-detection/resolve/main/models/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors"

curl -L -o experiments/checkpoints/weights/dinov3_next_cnn/model-2.safetensors \
  "https://huggingface.co/ManhQuangAI/dinov3-deepfake-detection/resolve/main/models/dinov3_next_cnn/model-2.safetensors"

curl -L -o experiments/checkpoints/weights/dinov3_small/model.safetensors \
  "https://huggingface.co/ManhQuangAI/dinov3-deepfake-detection/resolve/main/models/dinov3_small/model.safetensors"

find experiments/checkpoints/weights -name "*.safetensors" -ls   # expect 3 files
```

## 6. Download the Test Data

```bash
pip install -U "huggingface_hub[cli]"
hf download ManhQuangAI/df40-test-data-v3 --repo-type dataset --include "test_data_v3.zip" --local-dir .
unzip -o test_data_v3.zip -d .
ls test_data_v3/           # manifest.csv + real/ + <method>/ dirs
```

## 7. Test the Models

Works with the 3 backbones (no fine-tuned checkpoint needed). Run from the
repo root; on Mac use `--device mps` (`--amp` is auto-disabled on MPS):

```bash
# Smoke tests
python -m pytest

# Identity-disjoint linear-probe eval (ViT then CNN) on test_data_v3
python src/eval/eval_identity_disjoint.py --model vit --root test_data_v3 --tag test_data_v3 --device mps
python src/eval/eval_identity_disjoint.py --model cnn --root test_data_v3 --tag test_data_v3 --device mps

# ViT vs CNN comparison
python src/eval/eval_df40_vit_cnn.py --device mps
python src/experiments/compare_models.py --device mps

# Attention visualization
python src/experiments/visualize_attention.py \
  --model experiments/checkpoints/weights/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors \
  --output-dir experiments/plots/attention

# Feature extraction + predict using dinov3_small
python src/experiments/extract_features.py --model experiments/checkpoints/weights/dinov3_small/model.safetensors
python src/eval/predict.py --model experiments/checkpoints/weights/dinov3_small/model.safetensors
```

> `eval_deepfaketimit.py`, `eval_challenge.py`, and `eval_df40_all_methods.py`
> also run off the backbones but need extra data (DeepFake-TIMIT, challenge
> data, or `data/df40_subset`) to obtain/build first.

## 8. Fine-Tuned Checkpoints (Not Published)

These four `.pt` files are **not on GitHub or HF**. Get them from the teammate
(or retrain), then place them at:

```bash
# From train.py:            experiments/checkpoints/dinov3_finetuned.pt
# From finetune_lora.py:    experiments/checkpoints/finetune/vit_lora_finetuned.pt
# From finetune_compare.py: experiments/checkpoints/finetune/vit_finetuned.pt
#                            experiments/checkpoints/finetune/cnn_finetuned.pt
```

Once present, test with:

```bash
python src/eval/eval_finetuned.py --device mps
python src/eval/eval_finetuned_identity_disjoint.py --device mps
python src/eval/eval_df40_fake.py --device mps
python src/eval/analyze_threshold.py --ckpt experiments/checkpoints/finetune/vit_lora_finetuned.pt
```

To retrain on the Mac (needs a train/val/test split CSV from step 5's data):

```bash
python src/training/train.py --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv --test-csv data/splits/test.csv \
  --device mps --num-workers 0
```

---

## Notes

- **Mac worker processes:** if any script errors with a multiprocessing/spawn
  issue, add `--num-workers 0` (training scripts only).
- **Results:** eval JSON/MD reports go to `experiments/results/`; figures to
  `experiments/plots/` and `experiments/results/report/figures/`.

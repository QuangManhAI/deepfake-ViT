"""Download the DINOv3 Small checkpoint to the expected local path."""

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

REPO_ID = "ManhQuangAI/dinov3-deepfake-detection"
FILENAME = "models/dinov3_small/model.safetensors"
OUT_DIR = PROJECT_ROOT / "experiments" / "checkpoints" / "weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = OUT_DIR / "model.safetensors"

print(f"Downloading {REPO_ID}/{FILENAME} ...")
print("This may take several minutes; the file is several hundred MB.")

downloaded = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=str(OUT_DIR),
    local_dir_use_symlinks=False,
)

# hf_hub_download keeps the repo relative path, so the file ends up in
# experiments/checkpoints/weights/models/dinov3_small/model.safetensors
# Move it to the expected experiments/checkpoints/weights/model.safetensors
src = Path(downloaded)
if src != TARGET:
    if TARGET.exists():
        TARGET.unlink()
    shutil.move(str(src), str(TARGET))
    # clean up empty subdirs if any
    try:
        (OUT_DIR / "models").rmdir()
    except OSError:
        pass

print(f"Saved DINOv3 weights to: {TARGET}")
print(f"File size: {TARGET.stat().st_size / (1024**2):.1f} MB")

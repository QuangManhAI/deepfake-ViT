"""Smoke test the primary protocol through Dataset → DataLoader."""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import ImageDataset, EVAL_TF  # the same Dataset used by train.py


def main():
    print("="*80)
    print("Protocol DataLoader smoke test")
    print("="*80)
    
    # Load protocol config
    config_path = PROJECT_ROOT / "data" / "protocol" / "protocol_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"Protocol: {config['DATA_PROTOCOL']}")
    print(f"  train: {config['train_csv']}")
    print(f"  val:   {config['val_csv']}")
    print(f"  test:  {config['test_csv']}")
    
    # Test each split
    for split, csv_path in [("train", config["train_csv"]), ("val", config["val_csv"]), ("test", config["test_csv"])]:
        print(f"\n--- {split} ---")
        ds = ImageDataset(csv_path, EVAL_TF)
        print(f"  Dataset size: {len(ds)}")
        
        # Load 3 samples
        for i in range(min(3, len(ds))):
            x, y = ds[i]
            assert x.shape == torch.Size([3, 256, 256]), f"bad shape: {x.shape}"
            assert y in (0, 1), f"bad label: {y}"
            assert x.dtype == torch.float32
        print(f"  3 samples loaded; shapes/labels OK")
        
        # DataLoader one batch
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        xb, yb = next(iter(loader))
        assert xb.shape == torch.Size([4, 3, 256, 256]), f"bad batch shape: {xb.shape}"
        assert yb.shape == torch.Size([4]), f"bad label shape: {yb.shape}"
        assert yb.dtype == torch.int64, f"bad label dtype: {yb.dtype}"
        print(f"  one batch loaded: x={xb.shape}, y={yb.shape}, labels={yb.tolist()}")
    
    print("\n" + "="*80)
    print("SMOKE TEST PASS")
    print("="*80)


if __name__ == "__main__":
    main()
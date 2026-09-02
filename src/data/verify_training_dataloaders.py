"""Verify train.py Dataset/DataLoader can load the generated splits."""

import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torchvision import transforms as T


def simple_dataset(csv_path, transform):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    
    class DS:
        def __init__(self, rows, transform):
            self.rows = rows
            self.transform = transform
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            p = Path(r["path"])
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, int(r["label"])
    
    return DS(rows, transform)


def main():
    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "splits" / "val.csv"
    test_csv = PROJECT_ROOT / "data" / "splits" / "test.csv"
    
    tf = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    print("="*80)
    print("Training DataLoader Verification")
    print("="*80)
    
    for name, path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        print(f"\nLoading {name}: {path}")
        ds = simple_dataset(path, tf)
        print(f"  Dataset size: {len(ds)}")
        
        # Load first 3 samples
        for i in range(min(3, len(ds))):
            x, y = ds[i]
            print(f"  Sample {i}: shape={x.shape}, label={y}, type={type(x)}")
        
        # DataLoader batch
        loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)
        x, y = next(iter(loader))
        print(f"  DataLoader batch: x.shape={x.shape}, y={y.tolist()}")


if __name__ == "__main__":
    main()
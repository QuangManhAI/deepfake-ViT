"""Unit and integration tests for DF40 Data Preparation & Dataset Loading."""
import csv
import json
import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from src.training.train import EVAL_TF, IMG_SIZE, MEAN, STD, TRAIN_TF, ImageDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestDataPreparation(unittest.TestCase):
    def test_split_files_exist(self):
        """Verify that all required split CSVs and manifest JSONs exist."""
        splits_dir = PROJECT_ROOT / "data" / "splits"
        required_files = [
            "train.csv",
            "val.csv",
            "test.csv",
            "test_full.csv",
            "train_balanced.csv",
            "val_balanced.csv",
            "test_balanced.csv",
            "train_insight.csv",
            "val_insight.csv",
            "test_insight.csv",
            "train_faceswap.csv",
            "val_faceswap.csv",
            "test_faceswap.csv",
            "train_simswap.csv",
            "val_simswap.csv",
            "test_simswap.csv",
            "train_pool_693k.csv",
            "val_pool.csv",
            "train_combined_balanced.csv",
            "val_combined_balanced.csv",
            "celeb_df_extracted_real_frames.csv",
            "split_info.json",
            "methods_summary.json",
        ]


        for filename in required_files:
            p = splits_dir / filename
            self.assertTrue(p.exists(), f"Missing split file: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Empty file: {p}")

    def test_method_specific_splits_exist(self):
        """Verify that per-method test splits are generated for all methods."""
        methods_dir = PROJECT_ROOT / "data" / "splits" / "methods"
        self.assertTrue(methods_dir.exists(), "Methods directory does not exist")

        methods_summary_p = PROJECT_ROOT / "data" / "splits" / "methods_summary.json"
        self.assertTrue(methods_summary_p.exists(), "Missing methods_summary.json")
        with open(methods_summary_p) as f:
            methods_dict = json.load(f)

        self.assertGreaterEqual(len(methods_dict), 39, f"Expected >= 39 methods, got {len(methods_dict)}")

        for method_name in methods_dict.keys():
            bal_p = methods_dir / f"test_{method_name}_balanced.csv"
            full_p = methods_dir / f"test_{method_name}_full.csv"
            detailed_p = methods_dir / f"test_{method_name}_detailed.csv"
            bench_bal_p = methods_dir / f"benchmark_test_{method_name}_balanced.csv"
            bench_full_p = methods_dir / f"benchmark_test_{method_name}_full.csv"

            self.assertTrue(bal_p.exists(), f"Missing {bal_p}")
            self.assertTrue(full_p.exists(), f"Missing {full_p}")
            self.assertTrue(detailed_p.exists(), f"Missing {detailed_p}")
            self.assertTrue(bench_bal_p.exists(), f"Missing {bench_bal_p}")
            self.assertTrue(bench_full_p.exists(), f"Missing {bench_full_p}")

    def test_identity_disjoint_leakage(self):
        """Verify 0% identity and image overlap across train, val, and test splits."""
        splits_dir = PROJECT_ROOT / "data" / "splits"

        def load_paths_and_ids(csv_name):
            paths = set()
            ids = set()
            detailed_csv = splits_dir / f"{csv_name}_detailed.csv"
            self.assertTrue(detailed_csv.exists(), f"Missing {detailed_csv}")
            with open(detailed_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    paths.add(r["path"])
                    ids.add(r["identity"])
            return paths, ids

        tr_paths, tr_ids = load_paths_and_ids("train")
        val_paths, val_ids = load_paths_and_ids("val")
        te_paths, te_ids = load_paths_and_ids("test")

        # Zero image path overlap
        self.assertEqual(len(tr_paths & val_paths), 0, "Image overlap between Train & Val")
        self.assertEqual(len(tr_paths & te_paths), 0, "Image overlap between Train & Test")
        self.assertEqual(len(val_paths & te_paths), 0, "Image overlap between Val & Test")

        # Zero identity leakage
        self.assertEqual(len(tr_ids & val_ids), 0, "Identity leakage between Train & Val")
        self.assertEqual(len(tr_ids & te_ids), 0, "Identity leakage between Train & Test")
        self.assertEqual(len(val_ids & te_ids), 0, "Identity leakage between Val & Test")

    def test_balanced_splits_exact_ratio(self):
        """Verify exact 1:1 real:fake balance in balanced splits."""
        splits_dir = PROJECT_ROOT / "data" / "splits"

        for split_name in ["train_balanced", "val_balanced", "test_balanced"]:
            p = splits_dir / f"{split_name}.csv"
            reals, fakes = 0, 0
            with open(p, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # header
                for row in reader:
                    if int(row[1]) == 0:
                        reals += 1
                    else:
                        fakes += 1
            self.assertEqual(reals, fakes, f"Imbalance in {split_name}: {reals} reals vs {fakes} fakes")

    def test_method_balanced_exact_ratio(self):
        """Verify exact 1:1 real:fake balance across method-specific balanced test sets."""
        methods_dir = PROJECT_ROOT / "data" / "splits" / "methods"
        methods_summary_p = PROJECT_ROOT / "data" / "splits" / "methods_summary.json"
        with open(methods_summary_p) as f:
            methods_dict = json.load(f)

        for m in methods_dict.keys():
            p = methods_dir / f"test_{m}_balanced.csv"
            reals, fakes = 0, 0
            with open(p, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if int(row[1]) == 0:
                        reals += 1
                    else:
                        fakes += 1
            self.assertEqual(reals, fakes, f"Method {m} test balanced imbalance: {reals} reals vs {fakes} fakes")

    def test_dataloader_batch_loading(self):
        """Verify that PyTorch DataLoader loads batches correctly with transforms if torch is available."""
        if not HAS_TORCH:
            self.skipTest("PyTorch / torchvision not installed in current env")

        train_csv = str(PROJECT_ROOT / "data" / "splits" / "train_insight.csv")
        val_csv = str(PROJECT_ROOT / "data" / "splits" / "val_insight.csv")

        train_ds = ImageDataset(train_csv, transform=TRAIN_TF)
        val_ds = ImageDataset(val_csv, transform=EVAL_TF)

        self.assertGreater(len(train_ds), 0, "Train dataset is empty")
        self.assertGreater(len(val_ds), 0, "Val dataset is empty")

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        images, labels = next(iter(train_loader))

        self.assertEqual(images.shape, (8, 3, IMG_SIZE, IMG_SIZE), f"Unexpected image batch shape: {images.shape}")
        self.assertEqual(labels.shape, (8,), f"Unexpected label batch shape: {labels.shape}")
        self.assertEqual(images.dtype, torch.float32, "Image tensor should be float32")
        self.assertTrue(set(labels.tolist()).issubset({0, 1}), "Labels must be binary 0 or 1")

    def test_split_info_metadata(self):
        """Verify the validity of split_info.json."""
        split_info_p = PROJECT_ROOT / "data" / "splits" / "split_info.json"
        self.assertTrue(split_info_p.exists())
        with open(split_info_p) as f:
            meta = json.load(f)

        self.assertEqual(meta["seed"], 42)
        self.assertIn("identity_disjoint_splits", meta)
        self.assertGreater(meta["identity_disjoint_splits"]["train"]["total"], 0)
        self.assertGreater(meta["identity_disjoint_splits"]["val"]["total"], 0)
        self.assertGreater(meta["identity_disjoint_splits"]["test"]["total"], 0)
        self.assertGreaterEqual(meta["total_methods"], 39)


if __name__ == "__main__":
    unittest.main()


"""Canonical data loading.

The ONLY place that should read protocol CSVs into DataFrames/Datasets, so
notebooks and scripts never grow their own loaders.

    from src.data import loader

    df = loader.load_split("test", detailed=True)
    ds = loader.build_dataset("train", train=True)
    dl = loader.build_dataloader("val", batch_size=32)
    m  = loader.load_predictions_with_metadata(preds_csv)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data import protocol as P
from src.data.preprocessing import get_transforms

QUALITY_CSV = (P.PROJECT_ROOT / "experiments" / "results" / "data_quality"
               / "sample_quality.csv")

DETAILED_COLUMNS = ["path", "label", "method", "identity", "video", "domain"]

QUALITY_METRIC_COLUMNS = [
    "width", "height", "brightness", "contrast", "edge", "bits_per_pixel", "file_size",
]


# ---------------------------------------------------------------------------
# Manifest / split loading
# ---------------------------------------------------------------------------
def load_manifest() -> pd.DataFrame:
    """Full dataset manifest (pre-duplicate-cleanup). EDA only, NOT a split."""
    if not P.MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest not found: {P.MANIFEST_PATH}")
    return pd.read_csv(P.MANIFEST_PATH)


def load_split(split: str, detailed: bool = True) -> pd.DataFrame:
    """Load one canonical split as a DataFrame (adds a 'split' column)."""
    proto = P.load_protocol()
    csv_path = proto.csv(split, detailed=detailed)
    if not csv_path.exists():
        raise FileNotFoundError(f"protocol CSV missing: {csv_path}")
    df = pd.read_csv(csv_path)
    df["split"] = split
    return df


def load_all_splits(detailed: bool = True) -> dict[str, pd.DataFrame]:
    return {s: load_split(s, detailed=detailed) for s in P.SPLITS}


def load_splits_concat(detailed: bool = True) -> pd.DataFrame:
    return pd.concat(load_all_splits(detailed=detailed).values(), ignore_index=True)


def split_sizes() -> dict[str, int]:
    return {s: len(load_split(s, detailed=False)) for s in P.SPLITS}


def class_counts(split: str) -> dict[str, int]:
    df = load_split(split, detailed=False)
    vc = df["label"].value_counts().to_dict()
    return {
        "real": int(vc.get(P.LABEL_REAL, 0)),
        "fake": int(vc.get(P.LABEL_FAKE, 0)),
        "total": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Quality metadata
# ---------------------------------------------------------------------------
def load_sample_quality() -> pd.DataFrame:
    if not QUALITY_CSV.exists():
        raise FileNotFoundError(
            f"sample_quality.csv not found: {QUALITY_CSV}\n"
            "Run: .venv/bin/python src/data/data_quality_analysis.py"
        )
    return pd.read_csv(QUALITY_CSV)


def attach_quality(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Left-join the quality profile onto any frame with a 'path' column."""
    if "path" not in df.columns:
        raise KeyError("frame must contain a 'path' column to join quality metadata")
    q = load_sample_quality()
    cols = columns or QUALITY_METRIC_COLUMNS
    keep = ["path"] + [c for c in cols if c in q.columns]
    for extra in ("weakness_score", "weakness_reasons", "outlier_class"):
        if extra in q.columns and extra not in keep:
            keep.append(extra)
    out = df.copy()
    out["path"] = out["path"].astype(str)
    q = q.copy()
    q["path"] = q["path"].astype(str)
    return out.merge(q[keep], on="path", how="left")


# ---------------------------------------------------------------------------
# Predictions join (error-analysis data contract)
# ---------------------------------------------------------------------------
def load_predictions_with_metadata(predictions_csv: str | Path,
                                   split: str = "test",
                                   with_quality: bool = True) -> pd.DataFrame:
    """Predictions joined to canonical metadata and image quality.

    Verifies the prediction population IS the canonical split population, so an
    analysis can never silently run on a different test set.
    """
    preds = pd.read_csv(predictions_csv)
    preds["path"] = preds["path"].astype(str)

    meta = load_split(split, detailed=True)
    meta["path"] = meta["path"].astype(str)

    verify_matches_protocol(preds, split=split)

    meta_cols = [c for c in DETAILED_COLUMNS if c != "label"]
    overlap = [c for c in meta_cols if c != "path" and c in preds.columns]
    merged = preds.merge(meta[meta_cols], on="path", how="left", suffixes=("", "_meta"))
    for c in overlap:
        if f"{c}_meta" in merged.columns:
            merged[c] = merged[f"{c}_meta"]
            merged = merged.drop(columns=[f"{c}_meta"])

    if with_quality:
        merged = attach_quality(merged)
    return merged


def verify_matches_protocol(df: pd.DataFrame, split: str = "test") -> None:
    """Raise if `df` is not exactly the canonical split population."""
    canonical = load_split(split, detailed=False)
    want = set(canonical["path"].astype(str))
    got = set(df["path"].astype(str))
    if want != got:
        raise ValueError(
            f"population does NOT match canonical {split} split.\n"
            f"  canonical rows : {len(want)}\n"
            f"  provided rows  : {len(got)}\n"
            f"  missing        : {len(want - got)}\n"
            f"  unexpected     : {len(got - want)}\n"
            "Analyses must use the canonical protocol split."
        )
    if df["path"].duplicated().any():
        raise ValueError("provided frame contains duplicate paths")


# ---------------------------------------------------------------------------
# Torch Dataset / DataLoader
# ---------------------------------------------------------------------------
class ProtocolImageDataset(Dataset):
    """Image dataset backed by a canonical protocol split."""

    def __init__(self, split: str, train: bool = False, transform=None,
                 return_path: bool = False, dataframe: pd.DataFrame | None = None):
        self.split = split
        self.return_path = return_path
        self.transform = transform if transform is not None else get_transforms(train=train)
        df = dataframe if dataframe is not None else load_split(split, detailed=True)
        self.df = df.reset_index(drop=True)
        self.paths = self.df["path"].astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        p = Path(self.paths[i])
        if not p.is_absolute():
            p = P.PROJECT_ROOT / p
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[i]
        if self.return_path:
            return img, label, self.paths[i]
        return img, label


def build_dataset(split: str, train: bool = False, return_path: bool = False,
                  dataframe: pd.DataFrame | None = None) -> ProtocolImageDataset:
    """Dataset for a canonical split using canonical preprocessing.

    ``train=True`` selects augmented transforms. It does NOT change the split
    population -- augmentation is a training-time concern only.
    """
    return ProtocolImageDataset(split, train=train, return_path=return_path,
                                dataframe=dataframe)


def build_dataloader(split: str, batch_size: int = 32, train: bool = False,
                     shuffle: bool | None = None, num_workers: int = 0,
                     sampler=None, return_path: bool = False,
                     dataframe: pd.DataFrame | None = None) -> DataLoader:
    ds = build_dataset(split, train=train, return_path=return_path, dataframe=dataframe)
    if shuffle is None:
        shuffle = (split == "train")
    if sampler is not None:
        shuffle = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, sampler=sampler, pin_memory=False)


# ---------------------------------------------------------------------------
# Training-time class balancing (TRAIN SPLIT ONLY)
# ---------------------------------------------------------------------------
def class_weights(split: str = "train", device=None):
    """Inverse-frequency class weights for a weighted loss.

    Changes the LOSS, not the dataset. Val/test untouched.
    """
    import torch

    counts = class_counts(split)
    n_total = counts["total"]
    w = [n_total / max(1, counts["real"]) / 2.0,
         n_total / max(1, counts["fake"]) / 2.0]
    t = torch.tensor(w, dtype=torch.float32)
    return t.to(device) if device is not None else t


def weighted_random_sampler(split: str = "train", replacement: bool = True):
    """WeightedRandomSampler equalizing class frequency in TRAIN batches.

    Changes SAMPLING of training batches only. Never adds, removes or
    duplicates rows in the split definition, and must never touch val/test.
    """
    import torch
    from torch.utils.data import WeightedRandomSampler

    if split != "train":
        raise ValueError(
            "Balancing may only be applied to the train split. "
            "Validation and test populations are immutable."
        )
    df = load_split(split, detailed=False)
    counts = df["label"].value_counts().to_dict()
    per_class = {c: 1.0 / n for c, n in counts.items()}
    weights = df["label"].map(per_class).astype("float64")
    return WeightedRandomSampler(
        weights=torch.tensor(weights.to_numpy(copy=True), dtype=torch.double),
        num_samples=len(df),
        replacement=replacement,
    )


def describe() -> str:
    sizes = split_sizes()
    lines = ["CANONICAL SPLITS (from protocol)"]
    for s in P.SPLITS:
        c = class_counts(s)
        lines.append(f"  {s:5s}: {sizes[s]:6d} rows  real={c['real']:5d}  fake={c['fake']:6d}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(P.describe())
    print()
    print(describe())

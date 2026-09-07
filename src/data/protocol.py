"""Canonical dataset-protocol resolver.

Single source of truth for WHICH train/val/test split the project uses.
The active protocol is declared in ``data/protocol/protocol_config.json``;
this module is the typed accessor for it.

Rules enforced by convention (see DATA_PIPELINE.md):
  - Nothing may re-split the dataset.
  - Nothing may call train_test_split() on the protocol data.
  - Nothing may point at another CSV as if it were the protocol.
  - Validation and test populations are immutable.

Note: ``src/data/get_protocol_paths.py`` is a small pre-existing reader of the
same JSON config. Both agree because the JSON is the one source of truth; this
module adds typing, detailed-CSV resolution and metadata helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PROTOCOL_DIR = PROJECT_ROOT / "data" / "protocol"
PROTOCOL_CONFIG = PROTOCOL_DIR / "protocol_config.json"
PROTOCOL_METADATA = PROTOCOL_DIR / "protocol_metadata.json"

DATASET_ROOT = PROJECT_ROOT / "test_data_v3"
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"

# ---------------------------------------------------------------------------
# Canonical label semantics. Do not redefine these anywhere else.
# ---------------------------------------------------------------------------
LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_MAPPING = {LABEL_REAL: "real", LABEL_FAKE: "fake"}
LABEL_NAMES = ["real", "fake"]

SPLITS = ("train", "val", "test")


class ProtocolError(RuntimeError):
    """Raised when the canonical protocol is missing or inconsistent."""


@dataclass(frozen=True)
class Protocol:
    name: str
    protocol_dir: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    config: dict

    @property
    def train_detailed_csv(self) -> Path:
        return _detailed(self.train_csv)

    @property
    def val_detailed_csv(self) -> Path:
        return _detailed(self.val_csv)

    @property
    def test_detailed_csv(self) -> Path:
        return _detailed(self.test_csv)

    def csv(self, split: str, detailed: bool = False) -> Path:
        _check_split(split)
        base = {"train": self.train_csv, "val": self.val_csv, "test": self.test_csv}[split]
        return _detailed(base) if detailed else base

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "protocol_dir": str(self.protocol_dir),
            "train_csv": str(self.train_csv),
            "val_csv": str(self.val_csv),
            "test_csv": str(self.test_csv),
        }

    def __str__(self) -> str:  # pragma: no cover
        return f"Protocol({self.name})"


def _detailed(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_detailed{csv_path.suffix}")


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_protocol(config_path: str | Path | None = None) -> Protocol:
    """Load the active protocol. Canonical entry point."""
    cfg_path = Path(config_path) if config_path else PROTOCOL_CONFIG
    if not cfg_path.exists():
        raise ProtocolError(
            f"canonical protocol config not found: {cfg_path}\n"
            "Run: .venv/bin/python src/data/validate_protocol.py"
        )
    with open(cfg_path) as f:
        cfg = json.load(f)

    proto = Protocol(
        name=cfg.get("DATA_PROTOCOL", "unknown"),
        protocol_dir=_resolve(cfg.get("protocol_dir", PROTOCOL_DIR)),
        train_csv=_resolve(cfg["train_csv"]),
        val_csv=_resolve(cfg["val_csv"]),
        test_csv=_resolve(cfg["test_csv"]),
        config=cfg,
    )
    missing = [str(p) for p in (proto.train_csv, proto.val_csv, proto.test_csv)
               if not p.exists()]
    if missing:
        raise ProtocolError(f"protocol CSVs missing: {missing}")
    return proto


def get_protocol_paths(config_path: str | Path | None = None) -> tuple[Path, Path, Path]:
    p = load_protocol(config_path)
    return p.train_csv, p.val_csv, p.test_csv


def get_detailed_paths(config_path: str | Path | None = None) -> tuple[Path, Path, Path]:
    p = load_protocol(config_path)
    return p.train_detailed_csv, p.val_detailed_csv, p.test_detailed_csv


def protocol_metadata() -> dict:
    if not PROTOCOL_METADATA.exists():
        raise ProtocolError(f"protocol metadata not found: {PROTOCOL_METADATA}")
    with open(PROTOCOL_METADATA) as f:
        return json.load(f)


def describe() -> str:
    p = load_protocol()
    try:
        meta = protocol_metadata()
    except ProtocolError:
        meta = {}
    return "\n".join([
        "CANONICAL DATASET PROTOCOL",
        f"  name              : {p.name}",
        f"  strategy          : {meta.get('split_strategy', 'n/a')}",
        f"  seed              : {meta.get('random_seed', 'n/a')}",
        f"  dataset root      : {DATASET_ROOT}",
        f"  protocol dir      : {p.protocol_dir}",
        f"  train / val / test: {meta.get('train_size','?')} / "
        f"{meta.get('val_size','?')} / {meta.get('test_size','?')}",
        f"  label mapping     : {LABEL_MAPPING}",
        f"  identity          : {meta.get('identity_constraint', 'n/a')}",
        f"  video             : {meta.get('video_constraint', 'n/a')}",
        f"  duplicates        : {meta.get('duplicate_constraint', 'n/a')}",
    ])


KNOWN_LIMITATIONS = [
    "Video/source overlap EXISTS across splits (1,509 train-val, 1,542 train-test, "
    "930 val-test). It is NOT safe by default.",
    "Near-duplicate overlap: 4,215 cross-split groups were quantified but NOT removed.",
    "Class imbalance is ~24.5:1 fake:real and is preserved in val/test by design.",
    "Exact-duplicate cross-split overlap is 0 (703 rows removed/reassigned, logged).",
    "Identity overlap across splits is 0.",
]


if __name__ == "__main__":  # pragma: no cover
    print(describe())
    print("\nKNOWN LIMITATIONS")
    for item in KNOWN_LIMITATIONS:
        print(f"  - {item}")

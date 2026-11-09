"""Smoke tests validating the repository structure after the re-organization.

Structural checks (no heavy deps) run anywhere; torch-dependent checks skip
gracefully when ``torch`` is not installed, so CI/local can at least validate
the tree even without the full GPU environment.
"""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_src_subpackages_import():
    """All six src subpackages must exist and import (empty __init__, no torch)."""
    import src.data
    import src.eval
    import src.experiments
    import src.models
    import src.training
    import src.utils


def test_data_layout_exists():
    for sub in ("raw", "processed", "external"):
        assert (ROOT / "data" / sub).is_dir(), f"missing data/{sub}"


def test_experiments_layout_exists():
    for sub in ("checkpoints", "plots", "results", "runs"):
        assert (ROOT / "experiments" / sub).is_dir(), f"missing experiments/{sub}"


def test_root_is_clean():
    """Scripts live in scripts/ or src/, docs/reports in docs/ -- not at the repo root."""
    allowed_md = {"README.md"}
    stray = [p.name for p in ROOT.iterdir() if p.is_file()
             and (p.suffix == ".py" or (p.suffix == ".md" and p.name not in allowed_md))]
    assert not stray, f"move these out of the repo root: {stray}"


def test_model_builders_exist():
    torch = pytest.importorskip("torch")  # noqa: F841 (ensures torch is present)
    from src.models import dinov3_convnext, dinov3_vit, lora

    assert hasattr(dinov3_vit, "load_dinov3")
    assert hasattr(dinov3_convnext, "load_dinov3_convnext")
    assert hasattr(lora, "apply_lora")


def test_dinov3_graph_builds():
    torch = pytest.importorskip("torch")
    from src.models.dinov3_vit import DinoViT

    model = DinoViT(img_size=256, patch_size=16)
    out = model(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 384)

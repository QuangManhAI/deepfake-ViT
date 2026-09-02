"""
EDA utility functions for deepfake dataset analysis.

This module provides reusable functions for dataset exploration, quality analysis,
and visualization for the deepfake-ViT project.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from collections import Counter, defaultdict


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_dataset_metadata(metadata_file: str = "split_info.json") -> Dict[str, Any]:
    """Load dataset metadata from JSON file.
    
    Args:
        metadata_file: Name of the metadata file in data/splits/
        
    Returns:
        Dictionary containing dataset metadata
    """
    project_root = get_project_root()
    metadata_path = project_root / "data" / "splits" / metadata_file
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def load_methods_summary() -> Dict[str, Any]:
    """Load methods summary from JSON file.
    
    Returns:
        Dictionary containing method-wise statistics
    """
    project_root = get_project_root()
    methods_path = project_root / "data" / "splits" / "methods_summary.json"
    
    if not methods_path.exists():
        raise FileNotFoundError(f"Methods summary not found: {methods_path}")
    
    with open(methods_path, 'r') as f:
        return json.load(f)


def analyze_class_distribution(metadata: Dict[str, Any]) -> pd.DataFrame:
    """Analyze class distribution from metadata.
    
    Args:
        metadata: Dataset metadata dictionary
        
    Returns:
        DataFrame with class distribution statistics
    """
    if "identity_disjoint_splits" not in metadata:
        return pd.DataFrame()
    
    splits_data = metadata["identity_disjoint_splits"]
    
    data = []
    for split_name, split_info in splits_data.items():
        data.append({
            "split": split_name,
            "total": split_info["total"],
            "real": split_info["real"],
            "fake": split_info["fake"],
            "identities": split_info["identities"],
            "ratio": split_info["ratio"]
        })
    
    return pd.DataFrame(data)


def analyze_method_distribution(methods_summary: Dict[str, Any]) -> pd.DataFrame:
    """Analyze method distribution from methods summary.
    
    Args:
        methods_summary: Methods summary dictionary
        
    Returns:
        DataFrame with method distribution statistics
    """
    data = []
    for method, stats in methods_summary.items():
        data.append({
            "method": method,
            "test_split_fakes": stats.get("test_split_fakes", 0),
            "test_split_balanced_total": stats.get("test_split_balanced_total", 0),
            "benchmark_fakes": stats.get("benchmark_fakes", 0),
            "benchmark_balanced_total": stats.get("benchmark_balanced_total", 0),
            "benchmark_full_total": stats.get("benchmark_full_total", 0)
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values("benchmark_full_total", ascending=False)
    return df


def calculate_imbalance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate imbalance metrics from distribution DataFrame.
    
    Args:
        df: DataFrame with distribution data
        
    Returns:
        Dictionary with imbalance metrics
    """
    if df.empty:
        return {}
    
    metrics = {}
    
    # Class imbalance
    if "real" in df.columns and "fake" in df.columns:
        total_real = df["real"].sum()
        total_fake = df["fake"].sum()
        total = total_real + total_fake
        
        metrics["class_imbalance_ratio"] = total_fake / max(total_real, 1)
        metrics["real_percentage"] = total_real / total * 100
        metrics["fake_percentage"] = total_fake / total * 100
    
    # Method imbalance
    if "benchmark_full_total" in df.columns:
        method_counts = df["benchmark_full_total"]
        metrics["method_count_min"] = method_counts.min()
        metrics["method_count_max"] = method_counts.max()
        metrics["method_count_mean"] = method_counts.mean()
        metrics["method_count_std"] = method_counts.std()
        metrics["method_imbalance_ratio"] = method_counts.max() / max(method_counts.min(), 1)
    
    return metrics


def identify_weak_methods(methods_df: pd.DataFrame, 
                         threshold_percentile: float = 25) -> List[str]:
    """Identify weak/underrepresented methods.
    
    Args:
        methods_df: DataFrame with method distribution
        threshold_percentile: Percentile threshold for weakness
        
    Returns:
        List of method names considered weak
    """
    if methods_df.empty or "benchmark_full_total" not in methods_df.columns:
        return []
    
    threshold = methods_df["benchmark_full_total"].quantile(threshold_percentile / 100)
    weak_methods = methods_df[methods_df["benchmark_full_total"] <= threshold]["method"].tolist()
    
    return weak_methods


def identify_strong_methods(methods_df: pd.DataFrame,
                           threshold_percentile: float = 75) -> List[str]:
    """Identify strong/well-represented methods.
    
    Args:
        methods_df: DataFrame with method distribution
        threshold_percentile: Percentile threshold for strength
        
    Returns:
        List of method names considered strong
    """
    if methods_df.empty or "benchmark_full_total" not in methods_df.columns:
        return []
    
    threshold = methods_df["benchmark_full_total"].quantile(threshold_percentile / 100)
    strong_methods = methods_df[methods_df["benchmark_full_total"] >= threshold]["method"].tolist()
    
    return strong_methods


def estimate_identity_leakage_risk(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate identity leakage risk based on metadata.
    
    Args:
        metadata: Dataset metadata dictionary
        
    Returns:
        Dictionary with leakage risk assessment
    """
    risk_assessment = {
        "risk_level": "UNKNOWN",
        "reasoning": [],
        "identity_info_available": False
    }
    
    if "identity_disjoint_splits" in metadata:
        risk_assessment["identity_info_available"] = True
        splits = metadata["identity_disjoint_splits"]
        
        # Check if identity counts are provided
        total_identities = sum(split.get("identities", 0) for split in splits.values())
        
        if total_identities > 0:
            risk_assessment["reasoning"].append(
                f"Identity information available: {total_identities} total identities"
            )
            
            # Check if identity-disjoint splitting was used
            if metadata.get("seed") is not None:
                risk_assessment["reasoning"].append(
                    "Identity-disjoint splitting was used with seed"
                )
                risk_assessment["risk_level"] = "LOW"
            else:
                risk_assessment["reasoning"].append("No seed information for identity splitting")
                risk_assessment["risk_level"] = "MEDIUM"
        else:
            risk_assessment["reasoning"].append("Identity counts are zero or missing")
            risk_assessment["risk_level"] = "HIGH"
    
    return risk_assessment


def assess_data_readiness(metadata: Dict[str, Any], 
                         methods_summary: Dict[str, Any]) -> Dict[str, str]:
    """Assess overall data readiness for training.
    
    Args:
        metadata: Dataset metadata dictionary
        methods_summary: Methods summary dictionary
        
    Returns:
        Dictionary with readiness assessment
    """
    assessment = {}
    
    # Balance assessment
    if "identity_disjoint_splits" in metadata:
        splits = metadata["identity_disjoint_splits"]
        total_real = sum(s["real"] for s in splits.values())
        total_fake = sum(s["fake"] for s in splits.values())
        ratio = total_fake / max(total_real, 1)
        
        if ratio < 2:
            assessment["balance"] = "GOOD"
        elif ratio < 5:
            assessment["balance"] = "WARNING"
        else:
            assessment["balance"] = "CRITICAL"
    
    # Method coverage assessment
    if methods_summary:
        num_methods = len(methods_summary)
        if num_methods >= 30:
            assessment["method_coverage"] = "GOOD"
        elif num_methods >= 15:
            assessment["method_coverage"] = "WARNING"
        else:
            assessment["method_coverage"] = "CRITICAL"
    
    # Overall readiness
    critical_count = sum(1 for v in assessment.values() if v == "CRITICAL")
    warning_count = sum(1 for v in assessment.values() if v == "WARNING")
    
    if critical_count > 0:
        assessment["overall"] = "CRITICAL"
    elif warning_count > 0:
        assessment["overall"] = "WARNING"
    else:
        assessment["overall"] = "GOOD"
    
    return assessment


def generate_data_quality_report(metadata: Dict[str, Any],
                                methods_summary: Dict[str, Any]) -> str:
    """Generate a comprehensive data quality report.
    
    Args:
        metadata: Dataset metadata dictionary
        methods_summary: Methods summary dictionary
        
    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("DATA QUALITY REPORT")
    report_lines.append("=" * 60)
    
    # Dataset overview
    if "identity_disjoint_splits" in metadata:
        splits = metadata["identity_disjoint_splits"]
        total_samples = sum(s["total"] for s in splits.values())
        total_identities = sum(s["identities"] for s in splits.values())
        
        report_lines.append(f"\nDATASET OVERVIEW:")
        report_lines.append(f"  Total samples: {total_samples:,}")
        report_lines.append(f"  Total identities: {total_identities:,}")
        report_lines.append(f"  Number of methods: {len(methods_summary)}")
    
    # Class distribution
    class_df = analyze_class_distribution(metadata)
    if not class_df.empty:
        report_lines.append(f"\nCLASS DISTRIBUTION:")
        for _, row in class_df.iterrows():
            report_lines.append(f"  {row['split']}: {row['real']:,} real, {row['fake']:,} fake "
                              f"(ratio {row['ratio']})")
    
    # Method distribution
    methods_df = analyze_method_distribution(methods_summary)
    if not methods_df.empty:
        report_lines.append(f"\nMETHOD DISTRIBUTION (Top 10):")
        for _, row in methods_df.head(10).iterrows():
            report_lines.append(f"  {row['method']}: {row['benchmark_full_total']:,} total samples")
    
    # Imbalance metrics
    imbalance = calculate_imbalance_metrics(class_df)
    if imbalance:
        report_lines.append(f"\nIMBALANCE METRICS:")
        for key, value in imbalance.items():
            if isinstance(value, float):
                report_lines.append(f"  {key}: {value:.3f}")
            else:
                report_lines.append(f"  {key}: {value}")
    
    # Data readiness
    readiness = assess_data_readiness(metadata, methods_summary)
    report_lines.append(f"\nDATA READINESS:")
    for key, value in readiness.items():
        report_lines.append(f"  {key}: {value}")
    
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


def check_data_availability() -> Dict[str, bool]:
    """Check availability of actual data files.
    
    Returns:
        Dictionary indicating availability of different data components
    """
    project_root = get_project_root()
    availability = {}
    
    # Check for raw data
    raw_dir = project_root / "data" / "raw"
    availability["raw_data"] = raw_dir.exists() and any(raw_dir.iterdir())
    
    # Check for processed data
    processed_dir = project_root / "data" / "processed"
    availability["processed_data"] = processed_dir.exists() and any(processed_dir.iterdir())
    
    # Check for CSV splits
    splits_dir = project_root / "data" / "splits"
    csv_files = list(splits_dir.glob("*.csv")) if splits_dir.exists() else []
    availability["csv_splits"] = len(csv_files) > 0
    
    # Check for metadata
    metadata_files = list(splits_dir.glob("*.json")) if splits_dir.exists() else []
    availability["metadata"] = len(metadata_files) > 0
    
    # Check for model weights
    weights_dir = project_root / "experiments" / "checkpoints" / "weights"
    availability["model_weights"] = weights_dir.exists() and any(weights_dir.iterdir())
    
    return availability


def get_recommended_actions(metadata: Dict[str, Any],
                           methods_summary: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate recommended actions based on data analysis.
    
    Args:
        metadata: Dataset metadata dictionary
        methods_summary: Methods summary dictionary
        
    Returns:
        List of recommended actions with priority and rationale
    """
    actions = []
    
    # Check data availability
    availability = check_data_availability()
    
    if not availability["raw_data"]:
        actions.append({
            "priority": "CRITICAL",
            "action": "Download raw dataset",
            "rationale": "No raw data found. Download DF40 dataset from Hugging Face Hub",
            "command": "hf download ManhQuangAI/DF40_train --repo-type dataset --local-dir data/raw/DF40"
        })
    
    if not availability["csv_splits"]:
        actions.append({
            "priority": "CRITICAL", 
            "action": "Generate CSV split files",
            "rationale": "No CSV split files found. Run data preparation scripts",
            "command": "python src/data/split_dataset.py"
        })
    
    # Check class imbalance
    class_df = analyze_class_distribution(metadata)
    if not class_df.empty:
        imbalance = calculate_imbalance_metrics(class_df)
        if imbalance.get("class_imbalance_ratio", 0) > 5:
            actions.append({
                "priority": "HIGH",
                "action": "Address class imbalance",
                "rationale": f"Severe class imbalance detected (ratio: {imbalance['class_imbalance_ratio']:.2f})",
                "solution": "Use class-weighted loss or oversample minority class"
            })
    
    # Check method coverage
    methods_df = analyze_method_distribution(methods_summary)
    if not methods_df.empty:
        weak_methods = identify_weak_methods(methods_df)
        if weak_methods:
            actions.append({
                "priority": "MEDIUM",
                "action": "Strengthen weak methods",
                "rationale": f"{len(weak_methods)} methods are underrepresented",
                "solution": f"Targeted data collection for: {', '.join(weak_methods[:5])}"
            })
    
    # Check identity leakage risk
    leakage_risk = estimate_identity_leakage_risk(metadata)
    if leakage_risk["risk_level"] in ["MEDIUM", "HIGH"]:
        actions.append({
            "priority": "HIGH",
            "action": "Prevent identity leakage",
            "rationale": f"Identity leakage risk: {leakage_risk['risk_level']}",
            "solution": "Ensure identity-disjoint train/val/test splitting"
        })
    
    return actions
"""
Critical audit of the EDA implementation.

This script performs an independent verification of all EDA claims
and separates "implemented" from "actually analyzed".
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_file_exists(relative_path: str) -> bool:
    """Check if a file exists relative to project root."""
    return (project_root / relative_path).exists()


def load_json(relative_path: str):
    """Load a JSON file if it exists."""
    path = project_root / relative_path
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


def audit_data_availability():
    """Audit actual data availability."""
    print("="*80)
    print("1. DATA AVAILABILITY AUDIT")
    print("="*80)
    
    results = {}
    
    # Check key directories
    directories = {
        'data/raw': project_root / 'data/raw',
        'data/hug': project_root / 'data/hug',
        'data/processed': project_root / 'data/processed',
        'data/external': project_root / 'data/external',
        'data/splits': project_root / 'data/splits',
        'data/df40_subset': project_root / 'data/df40_subset',
        'experiments/checkpoints/weights': project_root / 'experiments/checkpoints/weights',
    }
    
    for name, path in directories.items():
        exists = path.exists()
        has_contents = any(path.iterdir()) if exists and path.is_dir() else False
        results[name] = {'exists': exists, 'has_contents': has_contents}
        status = '✓' if (exists and has_contents) else '⚠' if exists else '✗'
        content_status = f"(has contents)" if has_contents else f"(empty)" if exists else f"(missing)"
        print(f"  {status} {name:40s} {content_status}")
    
    # Check CSV files
    splits_dir = project_root / 'data/splits'
    csv_files = list(splits_dir.glob('*.csv')) if splits_dir.exists() else []
    print(f"\n  CSV split files found: {len(csv_files)}")
    
    # Check metadata files
    json_files = list(splits_dir.glob('*.json')) if splits_dir.exists() else []
    print(f"  JSON metadata files found: {len(json_files)}")
    for jf in json_files:
        print(f"    - {jf.name} ({jf.stat().st_size:,} bytes)")
    
    return results, len(csv_files), json_files


def verify_25_ratio():
    """Verify the 25:1 fake/real ratio source."""
    print("\n" + "="*80)
    print("2. 25:1 FAKE/REAL RATIO VERIFICATION")
    print("="*80)
    
    split_info = load_json('data/splits/split_info.json')
    if not split_info:
        print("  ✗ split_info.json not found - cannot verify ratio")
        return None
    
    print("  Source: split_info.json (JSON metadata)")
    print("  Status: METADATA-LEVEL ESTIMATE ONLY")
    print("  NOT verified against actual images")
    print()
    
    splits = split_info.get('identity_disjoint_splits', {})
    
    print("  Split breakdown:")
    total_real = 0
    total_fake = 0
    total_samples = 0
    total_identities = 0
    
    for split_name, split_data in splits.items():
        real = split_data['real']
        fake = split_data['fake']
        total = split_data['total']
        identities = split_data['identities']
        ratio = fake / max(real, 1)
        
        total_real += real
        total_fake += fake
        total_samples += total
        total_identities += identities
        
        print(f"    {split_name:6s}: {real:6,} real, {fake:6,} fake, {identities:6,} identities -> {ratio:.2f}:1")
    
    overall_ratio = total_fake / max(total_real, 1)
    real_pct = total_real / max(total_samples, 1) * 100
    fake_pct = total_fake / max(total_samples, 1) * 100
    
    print()
    print(f"  TOTAL: {total_real:,} real, {total_fake:,} fake = {total_samples:,} samples")
    print(f"  Overall fake:real ratio: {overall_ratio:.2f}:1")
    print(f"  Real percentage: {real_pct:.2f}%")
    print(f"  Fake percentage: {fake_pct:.2f}%")
    print(f"  Total identities: {total_identities:,}")
    
    # Check against CSV files
    csv_files = list((project_root / 'data/splits').glob('*.csv'))
    print(f"\n  CSV files to verify against: {len(csv_files)}")
    if len(csv_files) == 0:
        print("  ✗ NO CSV FILES FOUND - cannot verify metadata against actual data")
        print("  ✗ This is a metadata-level estimate, not actual dataset distribution")
    
    return {
        'ratio_source': 'split_info.json metadata',
        'overall_ratio': overall_ratio,
        'total_real': total_real,
        'total_fake': total_fake,
        'total_samples': total_samples,
        'total_identities': total_identities,
        'verified_against_images': False,
        'verified_against_csv': False
    }


def verify_identity_leakage():
    """Verify identity leakage claims."""
    print("\n" + "="*80)
    print("3. IDENTITY LEAKAGE CLAIM VERIFICATION")
    print("="*80)
    
    split_info = load_json('data/splits/split_info.json')
    
    print("  How 'identity' is defined:")
    print("    - split_info.json contains an 'identity_disjoint_splits' section")
    print("    - It lists an 'identities' count per split")
    print("    - However, it does NOT specify the source of identity")
    print("    - Possible sources: filename, video_id, folder structure, face recognition")
    print("    - Actual source: UNKNOWN from metadata alone")
    print()
    
    print("  What we know from metadata:")
    splits = split_info.get('identity_disjoint_splits', {}) if split_info else {}
    total_identities = sum(s['identities'] for s in splits.values()) if splits else 0
    
    for split_name, split_data in splits.items():
        samples_per_id = split_data['total'] / max(split_data['identities'], 1)
        print(f"    {split_name:6s}: {split_data['identities']:,} identities, {samples_per_id:.2f} samples/identity")
    print(f"    Total: {total_identities:,} identities")
    print()
    
    print("  What we CANNOT verify without raw images:")
    print("    ✗ Whether 'identity' = actual face identity")
    print("    ✗ Whether different expressions of same person are split correctly")
    print("    ✗ Whether video-level separation equals identity-level separation")
    print("    ✗ Whether face recognition was used")
    print("    ✗ Whether the splitting is truly identity-disjoint")
    print()
    
    # Check split_dataset.py to understand its identity approach
    split_dataset_path = project_root / 'src/data/split_dataset.py'
    if split_dataset_path.exists():
        with open(split_dataset_path, 'r') as f:
            content = f.read()
        
        print("  From src/data/split_dataset.py:")
        if 'nhân vật (folder 00–99)' in content:
            print("    - Identity defined by folder name (e.g., 00-99)")
            print("    - Dataset expected at: data/hug")
            print("    - Categories: wiki (real), insight/inpainting/text2img (fake)")
            print("    - This is a DIFFERENT dataset structure from DF40!")
            
            hug_dir = project_root / 'data/hug'
            if not hug_dir.exists():
                print("    ✗ data/hug directory does not exist")
        
        if 'split_info.json' not in content:
            print("    - split_dataset.py does NOT read or write the current split_info.json")
            print("    - The current split_info.json is from a different pipeline")
    
    return {
        'identity_source': 'unknown',
        'total_identities_metadata': total_identities,
        'verified': False,
        'leakage_conclusion': 'Identity leakage cannot be fully verified without raw images'
    }


def check_project_commands():
    """Check whether recommended commands are valid."""
    print("\n" + "="*80)
    print("4. RECOMMENDED COMMAND VERIFICATION")
    print("="*80)
    
    commands = [
        ('hf download ManhQuangAI/DF40_train --repo-type dataset --local-dir data/raw/DF40',
         'From README.md - valid dataset source'),
        ('python src/data/split_dataset.py',
         'Exists but expects data/hug, not DF40')
    ]
    
    for cmd, note in commands:
        print(f"  Command: {cmd}")
        print(f"  Note: {note}")
        print()
    
    # Verify split_dataset.py
    split_dataset = project_root / 'src/data/split_dataset.py'
    print(f"  src/data/split_dataset.py exists: {split_dataset.exists()}")
    if split_dataset.exists():
        print("    - This script expects data/hug structure")
        print("    - Not compatible with DF40 (data/raw/DF40)")
        print("    - The correct split script for DF40 is likely prepare_df40_splits.py")
    
    # Check prepare_df40_splits
    prepare_df40 = project_root / 'src/data/prepare_df40_splits.py'
    print(f"  src/data/prepare_df40_splits.py exists: {prepare_df40.exists()}")
    if prepare_df40.exists():
        print("    - This is the likely correct script for DF40 split generation")
    
    return commands


def audit_pipeline_consistency():
    """Audit consistency between EDA and training pipeline."""
    print("\n" + "="*80)
    print("5. DATA PIPELINE CONSISTENCY AUDIT")
    print("="*80)
    
    # What training expects
    train_py = project_root / 'src/training/train.py'
    if train_py.exists():
        with open(train_py, 'r') as f:
            train_content = f.read()
        
        # Extract expected CSV files from docstring
        print("  Training pipeline expects CSV files:")
        for line in train_content.split('\n'):
            if '--train-csv' in line or '--val-csv' in line or '--test-csv' in line:
                if 'data/splits' in line:
                    print(f"    {line.strip()}")
    
    # What exists
    splits_dir = project_root / 'data/splits'
    csv_files = list(splits_dir.glob('*.csv')) if splits_dir.exists() else []
    
    print(f"\n  Actual CSV files in data/splits/: {len(csv_files)}")
    if not csv_files:
        print("  ✗ MISMATCH: Training expects CSVs but none exist")
    
    # Metadata says CSVs should exist
    split_info = load_json('data/splits/split_info.json')
    if split_info:
        referenced_csvs = split_info.get('split_files', [])
        print(f"\n  Metadata references {len(referenced_csvs)} CSV files")
        print(f"  Actual CSV files: {len(csv_files)}")
        if len(csv_files) == 0 and len(referenced_csvs) > 0:
            print("  ✗ CRITICAL MISMATCH: Metadata references CSVs that don't exist")
    
    # EDA notebook paths
    notebook_path = project_root / 'src/data/eda_deepfake_dataset.ipynb'
    if notebook_path.exists():
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        print("\n  Notebook references:")
        notebook_text = json.dumps(notebook)
        if 'data/splits' in notebook_text:
            print("    ✓ References data/splits/")
        if 'data/raw' in notebook_text:
            print("    ✓ References data/raw/")
        if 'split_info.json' in notebook_text:
            print("    ✓ References split_info.json")


def create_dependency_matrix():
    """Create analysis dependency matrix."""
    print("\n" + "="*80)
    print("6. ANALYSIS DEPENDENCY MATRIX")
    print("="*80)
    
    analyses = [
        ("Class balance", True, False, False, "ANALYZED using metadata"),
        ("Method balance", True, False, False, "ANALYZED using metadata"),
        ("Exact duplicates", False, True, False, "FRAMEWORK ONLY"),
        ("Near duplicates", False, True, False, "FRAMEWORK ONLY"),
        ("Blur/sharpness", False, True, False, "FRAMEWORK ONLY"),
        ("Visual inspection", False, True, False, "NOT IMPLEMENTED"),
        ("Identity leakage", True, True, False, "METADATA-LEVEL ESTIMATE"),
        ("Feature similarity", False, True, True, "FRAMEWORK ONLY"),
        ("Weak-data discovery", True, True, True, "PARTIAL - metadata only"),
        ("Image quality", False, True, False, "FRAMEWORK ONLY"),
        ("Video-level leakage", True, True, False, "NOT IMPLEMENTED"),
        ("Face attribute analysis", False, True, True, "NOT IMPLEMENTED"),
    ]
    
    print(f"  {'Analysis':<25} {'Metadata':<9} {'Images':<7} {'Embeddings':<11} {'Status':<30}")
    print("  " + "-"*84)
    for analysis, needs_meta, needs_images, needs_emb, status in analyses:
        meta = '✓' if needs_meta else ''
        img = '✓' if needs_images else ''
        emb = '✓' if needs_emb else ''
        print(f"  {analysis:<25} {meta:<9} {img:<7} {emb:<11} {status:<30}")
    
    return analyses


def classify_requirements():
    """Classify each EDA requirement as actually analyzed, metadata-only, framework, etc."""
    print("\n" + "="*80)
    print("7. REQUIREMENT CLASSIFICATION")
    print("="*80)
    
    requirements = [
        ("1. Dataset Overview", "ANALYZED using metadata only"),
        ("2. Dataset Distribution", "ANALYZED using metadata only"),
        ("3. Class / Method Balance", "ANALYZED using metadata only"),
        ("4. Data Quality", "FRAMEWORK IMPLEMENTED but NOT EXECUTED"),
        ("5. Duplicate Detection", "FRAMEWORK IMPLEMENTED but NOT EXECUTED"),
        ("6. Near-Duplicate Detection", "FRAMEWORK IMPLEMENTED but NOT EXECUTED"),
        ("7. Leakage Analysis", "ANALYZED using metadata only"),
        ("8. Identity / Subject Analysis", "ANALYZED using metadata only"),
        ("9. Visual Inspection", "NOT IMPLEMENTED"),
        ("10. Deepfake Method Analysis", "ANALYZED using metadata only"),
        ("11. Weak Data Analysis", "PARTIAL - uses method counts only"),
        ("12. Feature Similarity", "FRAMEWORK IMPLEMENTED but NOT EXECUTED"),
        ("13. Data Improvement Recommendations", "COMPLETED based on metadata"),
        ("14. Image Quality Metrics", "FRAMEWORK IMPLEMENTED but NOT EXECUTED"),
        ("15. Video-level Leakage", "NOT IMPLEMENTED"),
        ("16. 'Same Person Different Expression'", "NOT IMPLEMENTED - requires faces"),
    ]
    
    for req, status in requirements:
        print(f"  {req:<45} {status}")
    
    return requirements


def main():
    """Run the full audit."""
    print("="*80)
    print("EDA IMPLEMENTATION CRITICAL AUDIT")
    print("="*80)
    print(f"Project: {project_root}")
    print()
    
    # Run all audit sections
    audit_data_availability()
    verify_25_ratio()
    verify_identity_leakage()
    check_project_commands()
    audit_pipeline_consistency()
    create_dependency_matrix()
    classify_requirements()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL AUDIT SUMMARY")
    print("="*80)
    print("\nACTUALLY ANALYZED (using metadata):")
    print("  ✓ Dataset overview (from split_info.json)")
    print("  ✓ Class distribution (from split_info.json)")
    print("  ✓ Method distribution (from methods_summary.json)")
    print("  ✓ Method balance (from methods_summary.json)")
    print("  ✓ Weak/strong method identification (from methods_summary.json)")
    print("\nANALYZED using metadata only (cannot verify without images):")
    print("  ⚠ 25:1 fake/real ratio - metadata estimate only")
    print("  ⚠ Identity leakage risk - based on identity counts in metadata")
    print("  ⚠ Data readiness - based on file existence checks")
    print("\nFRAMEWORK IMPLEMENTED but NOT EXECUTED:")
    print("  ✗ Image quality analysis (no images)")
    print("  ✗ Duplicate detection (no images)")
    print("  ✗ Near-duplicate detection (no images)")
    print("  ✗ Feature similarity analysis (no images/embeddings)")
    print("\nNOT IMPLEMENTED:")
    print("  ✗ Visual inspection grids")
    print("  ✗ Video-level leakage verification")
    print("  ✗ 'Same person, different expression' analysis")
    print("  ✗ Face attribute / pose analysis")
    print("\nINTEGRATION PROBLEMS FOUND:")
    print("  ✗ Metadata references CSV files that don't exist")
    print("  ✗ split_dataset.py expects data/hug, not DF40")
    print("  ✗ Training pipeline expects CSVs that don't exist")
    print("  ✗ Identity definition not verified from metadata")
    print("\nCORRECTED CONCLUSIONS:")
    print("  - The 25:1 ratio is a metadata-level estimate from split_info.json")
    print("  - Identity leakage risk cannot be fully verified without raw images")
    print("  - Weak data analysis is limited to sample counts; not multi-dimensional")
    print("  - Feature similarity cannot be performed without actual images")


if __name__ == "__main__":
    main()
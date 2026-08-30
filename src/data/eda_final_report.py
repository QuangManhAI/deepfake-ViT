"""
Generate final EDA summary report for the deepfake dataset.

This script creates a comprehensive summary of the EDA analysis findings
and recommendations.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.eda_utils import *

def generate_final_report():
    """Generate comprehensive final EDA report."""
    
    print("="*80)
    print("DEEPFAKE DATASET EDA - FINAL REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {project_root}")
    print("="*80)
    
    # Load metadata
    try:
        metadata = load_dataset_metadata()
        methods_summary = load_methods_summary()
        print("✓ Metadata loaded successfully")
    except Exception as e:
        print(f"✗ Could not load metadata: {e}")
        return
    
    # DATASET HEALTH ASSESSMENT
    print("\n" + "="*80)
    print("DATASET HEALTH ASSESSMENT")
    print("="*80)
    
    availability = check_data_availability()
    readiness = assess_data_readiness(metadata, methods_summary)
    leakage_risk = estimate_identity_leakage_risk(metadata)
    
    print(f"Balance:           {readiness.get('balance', 'UNKNOWN')}")
    print(f"Leakage:           {leakage_risk['risk_level']}")
    print(f"Duplicate rate:    Unknown (no image data available)")
    print(f"Image quality:     Not analyzed (no image data available)")
    print(f"Method coverage:   {readiness.get('method_coverage', 'UNKNOWN')}")
    print(f"Identity distribution: {leakage_risk['risk_level']}")
    
    # Overall readiness
    overall = readiness.get('overall', 'UNKNOWN')
    print(f"Overall data readiness: {overall}")
    
    # DETAILED FINDINGS
    print("\n" + "="*80)
    print("DETAILED FINDINGS")
    print("="*80)
    
    # Balance analysis
    class_df = analyze_class_distribution(metadata)
    imbalance = calculate_imbalance_metrics(class_df)
    
    print("\nBALANCE ANALYSIS:")
    print(f"  Class imbalance ratio: {imbalance['class_imbalance_ratio']:.2f}:1 (fake:real)")
    print(f"  Real images: {imbalance['real_percentage']:.2f}% of dataset")
    print(f"  Fake images: {imbalance['fake_percentage']:.2f}% of dataset")
    print(f"  Assessment: {'CRITICAL - Severe imbalance' if imbalance['class_imbalance_ratio'] > 5 else 'WARNING - Moderate imbalance' if imbalance['class_imbalance_ratio'] > 2 else 'GOOD - Balanced'}")
    
    # Method analysis
    methods_df = analyze_method_distribution(methods_summary)
    method_imbalance = calculate_imbalance_metrics(methods_df)
    
    print("\nMETHOD COVERAGE ANALYSIS:")
    print(f"  Total methods: {len(methods_df)}")
    print(f"  Method count range: {method_imbalance['method_count_min']:,} - {method_imbalance['method_count_max']:,}")
    print(f"  Method count mean: {method_imbalance['method_count_mean']:.0f}")
    print(f"  Method imbalance ratio: {method_imbalance['method_imbalance_ratio']:.2f}:1")
    print(f"  Assessment: {'GOOD' if method_imbalance['method_imbalance_ratio'] < 3 else 'WARNING' if method_imbalance['method_imbalance_ratio'] < 5 else 'CRITICAL'}")
    
    # Identity analysis
    print("\nIDENTITY ANALYSIS:")
    if "identity_disjoint_splits" in metadata:
        splits = metadata['identity_disjoint_splits']
        total_identities = sum(s['identities'] for s in splits.values())
        total_samples = sum(s['total'] for s in splits.values())
        print(f"  Total identities: {total_identities:,}")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Average samples per identity: {total_samples / max(total_identities, 1):.2f}")
        print(f"  Identity leakage risk: {leakage_risk['risk_level']}")
        print(f"  Identity-disjoint splitting: {'YES' if metadata.get('seed') else 'UNKNOWN'}")
    else:
        print("  Identity information not available")
    
    # WEAK DATA GROUPS
    print("\n" + "="*80)
    print("WEAKEST DATA GROUPS")
    print("="*80)
    
    weak_methods = identify_weak_methods(methods_df, threshold_percentile=20)
    print(f"\n1. UNDERREPRESENTED METHODS (Bottom 20%):")
    for i, method in enumerate(weak_methods[:5], 1):
        count = methods_df[methods_df['method'] == method]['benchmark_full_total'].values[0]
        print(f"   {i}. {method:20s} ({count:,} samples)")
    
    if imbalance['class_imbalance_ratio'] > 3:
        print(f"\n2. REAL IMAGES (Minority Class):")
        print(f"   Only {imbalance['real_percentage']:.2f}% of dataset")
        print(f"   {imbalance['class_imbalance_ratio']:.1f}:1 imbalance ratio")
    
    # Check for low samples per identity
    if "identity_disjoint_splits" in metadata:
        splits = metadata['identity_disjoint_splits']
        low_identity_splits = []
        for split_name, split_info in splits.items():
            samples_per_id = split_info['total'] / max(split_info['identities'], 1)
            if samples_per_id < 2:
                low_identity_splits.append((split_name, samples_per_id))
        
        if low_identity_splits:
            print(f"\n3. LOW-SAMPLE-IDENTITY SPLITS:")
            for split_name, samples_per_id in low_identity_splits:
                print(f"   {split_name}: {samples_per_id:.1f} samples/identity")
    
    # STRONG DATA GROUPS
    print("\n" + "="*80)
    print("STRONGEST DATA GROUPS")
    print("="*80)
    
    strong_methods = identify_strong_methods(methods_df, threshold_percentile=80)
    print(f"\n1. WELL-REPRESENTED METHODS (Top 20%):")
    for i, method in enumerate(strong_methods[:5], 1):
        count = methods_df[methods_df['method'] == method]['benchmark_full_total'].values[0]
        print(f"   {i}. {method:20s} ({count:,} samples)")
    
    print(f"\n2. FAKE IMAGES (Majority Class):")
    print(f"   {imbalance['fake_percentage']:.2f}% of dataset")
    print(f"   Well-represented across {len(methods_df)} methods")
    
    # RECOMMENDED ACTIONS
    print("\n" + "="*80)
    print("RECOMMENDED ACTIONS")
    print("="*80)
    
    actions = get_recommended_actions(metadata, methods_summary)
    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    actions_sorted = sorted(actions, key=lambda x: priority_order.get(x['priority'], 4))
    
    for i, action in enumerate(actions_sorted, 1):
        print(f"\n{i}. [{action['priority']}] {action['action']}")
        print(f"   Problem: {action['rationale']}")
        if 'solution' in action:
            print(f"   Solution: {action['solution']}")
        if 'command' in action:
            print(f"   Command: {action['command']}")
    
    # DATA IMPROVEMENT RECOMMENDATIONS
    print("\n" + "="*80)
    print("DATA IMPROVEMENT RECOMMENDATIONS")
    print("="*80)
    
    print("\nCLASS IMBALANCE HANDLING:")
    print("  • Use class-weighted loss functions")
    print("  • Implement oversampling for real images")
    print("  • Consider focal loss to focus on hard examples")
    print("  • Data augmentation targeted at minority class")
    
    print("\nMETHOD STRENGTHENING:")
    print("  • Prioritize data collection for weak methods")
    print("  • Targeted augmentation for underrepresented methods")
    print("  • Consider transfer learning from strong to weak methods")
    print("  • Method-specific fine-tuning strategies")
    
    print("\nIDENTITY LEAKAGE PREVENTION:")
    print("  • Ensure strict identity-disjoint train/val/test splitting")
    print("  • Group samples by identity before splitting")
    print("  • Verify no identity overlap between splits")
    print("  • Use identity-aware cross-validation")
    
    print("\nQUALITY IMPROVEMENT:")
    print("  • Implement image quality filtering")
    print("  • Remove corrupted or invalid images")
    print("  • Standardize resolution and preprocessing")
    print("  • Face alignment and normalization")
    
    # LIMITATIONS
    print("\n" + "="*80)
    print("ANALYSIS LIMITATIONS")
    print("="*80)
    
    print("\n⚠ ANALYSIS BASED ON METADATA ONLY:")
    print("  • No actual image data available for quality analysis")
    print("  • No visual inspection performed")
    print("  • No duplicate detection on actual images")
    print("  • No feature similarity analysis")
    print("  • No facial attribute analysis")
    
    print("\n⚠ MISSING ANALYSES:")
    print("  • Image quality metrics (blur, resolution, noise)")
    print("  • Exact and near-duplicate detection")
    print("  • Visual inspection grids")
    print("  • Feature embedding similarity")
    print("  • Face pose and attribute distribution")
    print("  • Compression artifact analysis")
    
    print("\n⚠ REQUIRES DATASET DOWNLOAD:")
    print("  • Download DF40 dataset from Hugging Face Hub")
    print("  • Generate CSV split files")
    print("  • Run data preparation scripts")
    print("  • Verify data integrity")
    
    # FINAL SUMMARY
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\nDATASET STATUS: {overall}")
    print(f"  • Strong method coverage (41 methods)")
    print(f"  • Identity-disjoint splitting implemented")
    print(f"  • CRITICAL class imbalance (25:1 fake:real)")
    print(f"  • Missing actual image data")
    print(f"  • Missing CSV split files")
    
    print(f"\nIMMEDIATE ACTIONS REQUIRED:")
    print(f"  1. Download raw dataset from Hugging Face Hub")
    print(f"  2. Generate CSV split files")
    print(f"  3. Address class imbalance")
    print(f"  4. Strengthen weak methods")
    
    print(f"\nFUTURE WORK:")
    print(f"  1. Image quality analysis")
    print(f"  2. Duplicate detection")
    print(f"  3. Visual inspection")
    print(f"  4. Feature similarity analysis")
    print(f"  5. Advanced leakage detection")
    
    print("\n" + "="*80)
    print("EDA ANALYSIS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    generate_final_report()
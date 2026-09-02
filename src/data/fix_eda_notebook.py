"""
Audit and fix EDA notebook for Matplotlib/API compatibility.

This script:
1. Reads the EDA notebook
2. Fixes known API incompatibilities (e.g., boxplot labels -> tick_labels)
3. Adds data validation for weak/strong method plots
4. Writes the corrected notebook back
"""

import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).parent.parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "src" / "data" / "eda_deepfake_dataset.ipynb"


def fix_boxplot_labels(cell_source):
    """Fix boxplot 'labels' parameter to 'tick_labels' for newer Matplotlib."""
    # Replace boxplot(..., labels=[...]) with boxplot(..., tick_labels=[...])
    pattern = r"ax\.boxplot\(\[\s*weak_counts\s*,\s*strong_counts\s*\],\s*labels\s*=\s*\[\s*'Weak Methods'\s*,\s*'Strong Methods'\s*\]\s*,\s*patch_artist\s*=\s*True\s*\)"
    replacement = "ax.boxplot([weak_counts, strong_counts], tick_labels=['Weak Methods', 'Strong Methods'], patch_artist=True)"
    return re.sub(pattern, replacement, cell_source)


def add_data_validation_to_weak_strong(cell_source):
    """Add data validation before weak/strong method boxplot."""
    old_code = """    weak_counts = [methods_df[methods_df['method'] == m]['benchmark_full_total'].values[0] 
                       for m in weak_methods]
        strong_counts = [methods_df[methods_df['method'] == m]['benchmark_full_total'].values[0] 
                        for m in strong_methods]
        
        bp = ax.boxplot([weak_counts, strong_counts], tick_labels=['Weak Methods', 'Strong Methods'], patch_artist=True)"""
    
    new_code = """    # Validate data before plotting
    weak_counts = [methods_df[methods_df['method'] == m]['benchmark_full_total'].values[0] 
                       for m in weak_methods]
    strong_counts = [methods_df[methods_df['method'] == m]['benchmark_full_total'].values[0] 
                        for m in strong_methods]
    
    # Convert to numeric, drop NaN, ensure non-empty
    weak_counts = [float(x) for x in weak_counts if x is not None and not (isinstance(x, float) and pd.isna(x))]
    strong_counts = [float(x) for x in strong_counts if x is not None and not (isinstance(x, float) and pd.isna(x))]
    
    if not weak_counts or not strong_counts:
        print("⚠ Cannot create boxplot: weak or strong method list is empty or invalid")
        plt.close(fig)
    else:
        bp = ax.boxplot([weak_counts, strong_counts], tick_labels=['Weak Methods', 'Strong Methods'], patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][1].set_facecolor('green')
        
        ax.set_ylabel('Sample Count')
        ax.set_title('Sample Count Distribution: Weak vs Strong Methods')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()"""
    
    # Replace the full block
    full_old_block = old_code + """
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][1].set_facecolor('green')
        
        ax.set_ylabel('Sample Count')
        ax.set_title('Sample Count Distribution: Weak vs Strong Methods')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()"""
    
    if full_old_block in cell_source:
        return cell_source.replace(full_old_block, new_code)
    return cell_source


def audit_notebook(notebook):
    """Audit notebook cells for common Matplotlib API issues."""
    issues = []
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Check for boxplot with 'labels' parameter
        if 'ax.boxplot(' in source and 'labels=' in source:
            issues.append(f"Cell {i}: boxplot uses 'labels=' parameter, should be 'tick_labels='")
        
        # Check for deprecated seaborn/methods (basic checks)
        if 'plt.hist(' in source and 'density' not in source and 'normed' in source:
            issues.append(f"Cell {i}: uses deprecated 'normed' in hist")
        
        # Check for plt.tight_layout without show
        if 'plt.tight_layout()' in source and 'plt.show()' not in source:
            pass  # not necessarily an error
    
    return issues


def main():
    print("Loading EDA notebook...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Initial audit
    initial_issues = audit_notebook(notebook)
    print(f"Initial issues found: {len(initial_issues)}")
    for issue in initial_issues:
        print(f"  - {issue}")
    
    # Fix code cells
    fixed_cells = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        original = source
        
        # Fix boxplot labels
        source = fix_boxplot_labels(source)
        # Add data validation
        source = add_data_validation_to_weak_strong(source)
        
        if source != original:
            # Update cell source
            if isinstance(cell['source'], list):
                # Convert back to list of lines
                cell['source'] = source.split('\n')
                # Preserve line ending style (list items)
                if not source.endswith('\n'):
                    cell['source'] = [s + '\n' for s in cell['source'][:-1]] + [cell['source'][-1]]
            else:
                cell['source'] = source
            fixed_cells += 1
            print(f"Fixed cell with 'Strong Data Analysis' boxplot")
    
    # Re-audit
    final_issues = audit_notebook(notebook)
    print(f"\nFinal issues found: {len(final_issues)}")
    for issue in final_issues:
        print(f"  - {issue}")
    
    # Save
    print(f"\nSaving corrected notebook to {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Done. Fixed {fixed_cells} code cell(s).")


if __name__ == "__main__":
    main()
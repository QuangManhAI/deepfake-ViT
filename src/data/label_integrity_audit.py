"""Label and metadata integrity audit for the cleaned dataset."""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_ROOT = PROJECT_ROOT / "test_data_v3"
MANIFEST = TEST_ROOT / "manifest.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits_identity_clean"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "data_quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest():
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["label"] = int(r["label"])
            rows.append(r)
    return rows


def load_split_detailed():
    rows = []
    for split in ["train", "val", "test"]:
        with open(SPLITS_DIR / f"{split}_detailed.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["label"] = int(r["label"])
                r["split"] = split
                rows.append(r)
    return rows


def get_valid_methods():
    valid = set()
    for p in TEST_ROOT.iterdir():
        if p.is_dir():
            valid.add(p.name)
    return valid


def audit():
    print("="*80)
    print("Label & Metadata Integrity Audit")
    print("="*80)
    
    manifest_rows = load_manifest()
    split_rows = load_split_detailed()
    valid_methods = get_valid_methods()
    
    issues = []
    
    # 1. Manifest vs split consistency
    split_by_path = {r["path"]: r for r in split_rows}
    for r in manifest_rows:
        p = r["path"]
        if p in split_by_path:
            sr = split_by_path[p]
            for col in ["label", "method", "identity", "domain", "video"]:
                if str(r[col]) != str(sr[col]):
                    issues.append({
                        "path": p,
                        "issue_type": "manifest_split_mismatch",
                        "details": f"{col}: manifest={r[col]} split={sr[col]}",
                        "split": sr["split"],
                    })
    
    # 2. Folder/method consistency and label consistency
    all_rows = manifest_rows  # use manifest as the full source
    for r in all_rows:
        p = r["path"]
        parts = p.split("/")
        
        # Missing or invalid method
        if not r["method"]:
            issues.append({"path": p, "issue_type": "missing_method", "details": "method is empty", "split": ""})
        elif r["method"] not in valid_methods:
            issues.append({"path": p, "issue_type": "invalid_method", "details": f"method='{r['method']}' not in folder list", "split": ""})
        
        # Path structure
        if len(parts) < 2:
            issues.append({"path": p, "issue_type": "bad_path", "details": f"path has <2 components", "split": ""})
            continue
        
        top = parts[0]
        if top == "real":
            if len(parts) != 2:
                issues.append({"path": p, "issue_type": "bad_path_real", "details": "real/ should contain direct image", "split": ""})
            if r["label"] != 0:
                issues.append({"path": p, "issue_type": "real_folder_fake_label", "details": f"path in real/ but label={r['label']}", "split": ""})
            if r["method"] != "real":
                issues.append({"path": p, "issue_type": "real_folder_wrong_method", "details": f"path in real/ but method='{r['method']}'", "split": ""})
        else:
            # fake method folder
            if len(parts) < 3 or parts[1] != "fake":
                issues.append({"path": p, "issue_type": "bad_path_fake", "details": f"expected '<method>/fake/<file>', got {p}", "split": ""})
            if top != r["method"]:
                issues.append({"path": p, "issue_type": "path_method_mismatch", "details": f"folder={top} but method={r['method']}", "split": ""})
            if r["label"] != 1:
                issues.append({"path": p, "issue_type": "fake_folder_real_label", "details": f"path in fake/ but label={r['label']}", "split": ""})
        
        # Missing metadata
        if not r.get("identity"):
            issues.append({"path": p, "issue_type": "missing_identity", "details": "identity is empty", "split": ""})
        if not r.get("video"):
            issues.append({"path": p, "issue_type": "missing_video", "details": "video is empty", "split": ""})
        if not r.get("domain"):
            issues.append({"path": p, "issue_type": "missing_domain", "details": "domain is empty", "split": ""})
    
    # 3. Duplicate paths in manifest
    path_counts = Counter(r["path"] for r in all_rows)
    for p, c in path_counts.items():
        if c > 1:
            issues.append({"path": p, "issue_type": "duplicate_path", "details": f"path appears {c} times in manifest", "split": ""})
    
    # 4. Duplicate metadata records (same path with different label/method/identity/video)
    #    Already covered by duplicate paths, but we can also flag same MD5 if needed.
    
    # 5. Inconsistent identity-video mapping
    #    If same (video, method) has many different identities, that may be suspicious.
    #    But in DF40 a video may be used to generate many fakes from different source identities,
    #    so this is not necessarily an error. We do not flag by default.
    
    # Save issues
    issues_df = pd.DataFrame(issues)
    issues_df.to_csv(OUT_DIR / "label_integrity.csv", index=False)
    
    summary = {
        "total_manifest_rows": len(manifest_rows),
        "total_split_rows": len(split_rows),
        "total_issues": len(issues),
        "issue_counts": issues_df["issue_type"].value_counts().to_dict() if not issues_df.empty else {},
        "output": str(OUT_DIR / "label_integrity.csv"),
    }
    with open(OUT_DIR / "label_integrity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Manifest rows: {len(manifest_rows)}")
    print(f"Split rows: {len(split_rows)}")
    print(f"Issues found: {len(issues)}")
    if issues:
        print("Issue breakdown:")
        for k, v in summary["issue_counts"].items():
            print(f"  {k}: {v}")
    else:
        print("No issues found")
    print(f"Saved: {OUT_DIR / 'label_integrity.csv'}")


if __name__ == "__main__":
    audit()
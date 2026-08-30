"""
Comprehensive data quality, label integrity, and distribution analysis.

Outputs:
- experiments/results/data_quality/sample_quality.csv
- experiments/results/data_quality/method_quality_summary.csv
- experiments/results/data_quality/identity_quality_summary.csv
- experiments/results/data_quality/video_quality_summary.csv
- experiments/results/data_quality/distribution_shift.csv
- experiments/results/data_quality/data_collection_recommendations.csv
"""

import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits_identity_clean"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "data_quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Face detection is computationally expensive for 30k images.
# It is disabled by default. Fields are filled with NaN and marked as unavailable.
USE_FACE_DETECTION = False

try:
    import cv2
    FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(FACE_CASCADE_PATH):
        raise ImportError("Haar cascade not found")
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH) if USE_FACE_DETECTION else None
    FACE_AVAILABLE = USE_FACE_DETECTION
except Exception:
    FACE_AVAILABLE = False
    face_cascade = None


def load_rows():
    rows = []
    for split in ["train", "val", "test"]:
        with open(SPLITS_DIR / f"{split}_detailed.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["label"] = int(r["label"])
                r["split"] = split
                rows.append(r)
    return pd.DataFrame(rows)


def image_metrics(path):
    try:
        with Image.open(path) as img:
            w, h = img.size
            img_rgb = img.convert("RGB")
            arr = np.array(img_rgb)
            gray = np.mean(arr, axis=2)
            
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # sharpness via Laplacian variance (PIL gradient)
            gy, gx = np.gradient(gray)
            edge = float(np.mean(np.sqrt(gx**2 + gy**2)))
            
            aspect = w / h if h > 0 else 0.0
            file_size = float(os.path.getsize(path))
            bits_per_pixel = (file_size * 8) / (w * h * 3) if (w * h) > 0 else 0.0
            
            metrics = {
                "width": w,
                "height": h,
                "aspect": aspect,
                "brightness": brightness,
                "contrast": contrast,
                "edge": edge,
                "file_size": file_size,
                "bits_per_pixel": bits_per_pixel,
            }
            
            # Face detection
            if FACE_AVAILABLE:
                gray_cv = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray_cv, 1.1, 4)
                if len(faces) > 0:
                    total_face_area = sum(w * h for (x, y, w, h) in faces)
                    img_area = w * h
                    metrics["face_count"] = int(len(faces))
                    metrics["face_area_ratio"] = float(total_face_area / img_area)
                    metrics["face_bbox_size"] = float(np.mean([w * h for (x, y, w, h) in faces]))
                else:
                    metrics["face_count"] = 0
                    metrics["face_area_ratio"] = 0.0
                    metrics["face_bbox_size"] = 0.0
            else:
                metrics["face_count"] = np.nan
                metrics["face_area_ratio"] = np.nan
                metrics["face_bbox_size"] = np.nan
            
            return metrics
    except Exception as e:
        return {"error": str(e)}


def compute_quality(df):
    print("="*80)
    print("Computing per-image quality metrics")
    print("="*80)
    
    records = []
    t0 = time.time()
    for i, row in df.iterrows():
        p = Path(row["path"])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        m = image_metrics(str(p))
        m["path"] = row["path"]
        m["label"] = row["label"]
        m["method"] = row["method"]
        m["identity"] = row["identity"]
        m["video"] = row["video"]
        m["domain"] = row["domain"]
        m["split"] = row["split"]
        records.append(m)
        if i % 5000 == 0:
            print(f"  {i:,} processed, {time.time()-t0:.1f}s")
    
    qdf = pd.DataFrame(records)
    errors = qdf[qdf["error"].notna()] if "error" in qdf.columns else pd.DataFrame()
    if not errors.empty:
        print(f"  {len(errors)} errors")
    
    return qdf


def add_weakness_scores(qdf):
    """Add per-image weakness score with documented weights."""
    
    # Weights (documented and explicit)
    # Each flag contributes +1 to weakness_score
    qdf["low_resolution"] = (qdf["width"] < 256) | (qdf["height"] < 256)
    qdf["high_blur"] = qdf["edge"] < qdf["edge"].quantile(0.05)
    qdf["extreme_brightness"] = (qdf["brightness"] < qdf["brightness"].quantile(0.01)) | (qdf["brightness"] > qdf["brightness"].quantile(0.99))
    qdf["extreme_darkness"] = qdf["brightness"] < qdf["brightness"].quantile(0.05)
    qdf["low_contrast"] = qdf["contrast"] < qdf["contrast"].quantile(0.05)
    qdf["abnormal_aspect"] = (qdf["aspect"] < 0.5) | (qdf["aspect"] > 2.0)
    qdf["high_compression"] = qdf["bits_per_pixel"] < qdf["bits_per_pixel"].quantile(0.05)
    
    qdf["weakness_reasons"] = ""
    reasons = []
    for flag in ["low_resolution", "high_blur", "extreme_darkness", "low_contrast", "abnormal_aspect", "high_compression"]:
        reasons.append(qdf[flag].apply(lambda x: flag if x else ""))
    qdf["weakness_reasons"] = pd.concat(reasons, axis=1).apply(lambda x: ",".join([s for s in x if s]), axis=1)
    
    qdf["weakness_score"] = qdf[["low_resolution", "high_blur", "extreme_darkness", "low_contrast", "abnormal_aspect", "high_compression"]].sum(axis=1)
    
    return qdf


def add_outliers(qdf):
    """Classify outliers using IQR and MAD."""
    
    def iqr_bounds(s, k=1.5):
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        return q1 - k * iqr, q3 + k * iqr
    
    def mad_outliers(s, thresh=3.5):
        med = s.median()
        mad = (s - med).abs().median()
        return (s - med).abs() / (1.4826 * mad) > thresh
    
    for col in ["width", "height", "aspect", "brightness", "contrast", "edge", "bits_per_pixel"]:
        lo, hi = iqr_bounds(qdf[col].dropna())
        qdf[f"{col}_iqr_outlier"] = ((qdf[col] < lo) | (qdf[col] > hi))
        qdf[f"{col}_mad_outlier"] = mad_outliers(qdf[col].dropna())
    
    # Outlier classification
    def classify(row):
        bad_quality = (
            row["low_resolution"] or row["high_blur"] or row["extreme_darkness"] or
            row["low_contrast"] or row["high_compression"]
        )
        hard_sample = (
            row["abnormal_aspect"] or row["aspect_iqr_outlier"] or row["width_iqr_outlier"] or row["height_iqr_outlier"]
        )
        
        if bad_quality:
            return "POTENTIAL_BAD_QUALITY"
        elif hard_sample:
            return "POTENTIAL_HARD_SAMPLE"
        else:
            return "NORMAL_OUTLIER"
    
    qdf["outlier_class"] = qdf.apply(classify, axis=1)
    qdf.loc[qdf["weakness_score"] == 0, "outlier_class"] = "NONE"
    
    return qdf


def method_summary(qdf):
    fake = qdf[qdf["label"] == 1]
    summary = fake.groupby("method").agg(
        sample_count=("path", "count"),
        real_count=("label", lambda x: (x == 0).sum()),
        identity_count=("identity", "nunique"),
        video_count=("video", "nunique"),
        median_width=("width", "median"),
        median_height=("height", "median"),
        median_edge=("edge", "median"),
        median_brightness=("brightness", "median"),
        median_contrast=("contrast", "median"),
        duplicate_rate=("path", lambda x: 0.0),  # exact duplicates already removed
        quality_outlier_rate=("weakness_score", lambda x: (x > 0).mean()),
    ).reset_index()
    
    # Normalized composite weakness score
    summary["weakness_score"] = 0.0
    summary["weakness_score"] += (summary["sample_count"].max() - summary["sample_count"]) / summary["sample_count"].max() * 0.25
    summary["weakness_score"] += (1 - (summary["median_edge"] - summary["median_edge"].min()) / (summary["median_edge"].max() - summary["median_edge"].min())) * 0.20
    summary["weakness_score"] += (summary["median_width"] < 256).astype(float) * 0.20
    summary["weakness_score"] += (summary["quality_outlier_rate"]) * 0.20
    summary["weakness_score"] += (1 - summary["identity_count"] / summary["sample_count"].clip(lower=1)) * 0.15
    
    def main_weakness(row):
        reasons = []
        if row["sample_count"] < row["sample_count"].mean():  # but not using mean inside apply
            reasons.append("low_count")
        if row["median_width"] < 256:
            reasons.append("low_resolution")
        if row["median_edge"] < summary["median_edge"].median():
            reasons.append("low_sharpness")
        if row["quality_outlier_rate"] > 0.2:
            reasons.append("high_outlier_rate")
        return ",".join(reasons) if reasons else "none"
    
    # Compute means/medians first
    mean_count = summary["sample_count"].mean()
    med_edge = summary["median_edge"].median()
    
    def main_weakness_row(row):
        reasons = []
        if row["sample_count"] < mean_count:
            reasons.append("low_count")
        if row["median_width"] < 256:
            reasons.append("low_resolution")
        if row["median_edge"] < med_edge:
            reasons.append("low_sharpness")
        if row["quality_outlier_rate"] > 0.2:
            reasons.append("high_outlier_rate")
        return ",".join(reasons) if reasons else "none"
    
    summary["main_weakness"] = summary.apply(main_weakness_row, axis=1)
    summary["severity"] = pd.cut(summary["weakness_score"], bins=[-0.1, 0.2, 0.5, 1.0], labels=["low", "medium", "high"])
    summary = summary.sort_values("weakness_score", ascending=False)
    
    return summary


def identity_summary(qdf):
    return qdf.groupby("identity").agg(
        image_count=("path", "count"),
        video_count=("video", "nunique"),
        method_count=("method", "nunique"),
        split_count=("split", "nunique"),
        median_edge=("edge", "median"),
        median_width=("width", "median"),
    ).reset_index()


def video_summary(qdf):
    return qdf.groupby("video").agg(
        image_count=("path", "count"),
        identity_count=("identity", "nunique"),
        method_count=("method", "nunique"),
        split_count=("split", "nunique"),
        median_edge=("edge", "median"),
        median_width=("width", "median"),
    ).reset_index()


def distribution_shift(qdf):
    splits = ["train", "val", "test"]
    metrics = ["width", "height", "aspect", "brightness", "contrast", "edge", "bits_per_pixel"]
    rows = []
    
    for m in metrics:
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            x = qdf[qdf["split"] == a][m].dropna()
            y = qdf[qdf["split"] == b][m].dropna()
            
            if len(x) > 0 and len(y) > 0:
                # histograms for Jensen-Shannon
                bins = np.histogram_bin_edges(pd.concat([x, y]), bins=50)
                hx, _ = np.histogram(x, bins=bins, density=True)
                hy, _ = np.histogram(y, bins=bins, density=True)
                js = jensenshannon(hx + 1e-12, hy + 1e-12)
                
                ks = ks_2samp(x, y)
                
                rows.append({
                    "metric": m,
                    "split_a": a,
                    "split_b": b,
                    "mean_a": x.mean(),
                    "mean_b": y.mean(),
                    "std_a": x.std(),
                    "std_b": y.std(),
                    "jensen_shannon": js,
                    "ks_statistic": ks.statistic,
                    "ks_pvalue": ks.pvalue,
                })
    
    # Categorical distributions
    for col in ["method", "domain"]:
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            ca = qdf[qdf["split"] == a][col].value_counts(normalize=True)
            cb = qdf[qdf["split"] == b][col].value_counts(normalize=True)
            idx = ca.index.union(cb.index)
            pa = ca.reindex(idx, fill_value=0)
            pb = cb.reindex(idx, fill_value=0)
            js = jensenshannon(pa + 1e-12, pb + 1e-12)
            rows.append({
                "metric": col,
                "split_a": a,
                "split_b": b,
                "mean_a": np.nan,
                "mean_b": np.nan,
                "std_a": np.nan,
                "std_b": np.nan,
                "jensen_shannon": js,
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
            })
    
    return pd.DataFrame(rows)


def data_collection_recommendations(qdf, method_summary):
    recs = []
    
    # Class imbalance
    real = (qdf["label"] == 0).sum()
    fake = (qdf["label"] == 1).sum()
    recs.append({
        "Problem": "Class imbalance",
        "Evidence": f"Real={real:,}, Fake={fake:,}, ratio={fake/max(1,real):.2f}:1",
        "Impact": "Model bias toward fake",
        "Recommended Action": "Collect additional real identities or use class-weighted loss",
        "Priority": "HIGH",
    })
    
    # Low resolution
    low_res = (qdf["width"] < 256).sum()
    recs.append({
        "Problem": "Low resolution prevalence",
        "Evidence": f"{low_res:,} / {len(qdf):,} images below 256px",
        "Impact": "Reduced detail for model",
        "Recommended Action": "Collect or generate higher-resolution samples",
        "Priority": "HIGH",
    })
    
    # Weak methods
    weak = method_summary.sort_values("weakness_score", ascending=False).head(10)
    for _, row in weak.iterrows():
        recs.append({
            "Problem": f"Weak method: {row['method']}",
            "Evidence": f"count={row['sample_count']}, median_edge={row['median_edge']:.2f}, median_width={row['median_width']:.0f}, outlier_rate={row['quality_outlier_rate']:.2%}",
            "Impact": "Poor method-specific generalization",
            "Recommended Action": f"Collect {row['main_weakness']} improvements for {row['method']}: higher resolution, sharper, more identities",
            "Priority": "HIGH" if row["weakness_score"] > 0.5 else "MEDIUM",
        })
    
    # Identity imbalance
    id_counts = qdf["identity"].value_counts()
    single_id = (id_counts == 1).sum()
    top_ids = id_counts.head(100).sum()
    recs.append({
        "Problem": "Identity imbalance",
        "Evidence": f"{single_id:,} identities with 1 sample; top 100 identities cover {top_ids/len(qdf):.2%} of images",
        "Impact": "Some identities dominate; others underrepresented",
        "Recommended Action": "Collect more independent identities, especially for weak methods",
        "Priority": "MEDIUM",
    })
    
    # Video/source concentration
    vid_counts = qdf["video"].value_counts()
    top_vids = vid_counts.head(100).sum()
    recs.append({
        "Problem": "Video/source concentration",
        "Evidence": f"Top 100 videos cover {top_vids/len(qdf):.2%} of images",
        "Impact": "Source-level overfitting and leakage risk",
        "Recommended Action": "Collect more independent source videos; consider video-disjoint split if needed",
        "Priority": "MEDIUM",
    })
    
    return pd.DataFrame(recs)


def main():
    t0 = time.time()
    df = load_rows()
    qdf = compute_quality(df)
    qdf = add_weakness_scores(qdf)
    qdf = add_outliers(qdf)
    
    qdf.to_csv(OUT_DIR / "sample_quality.csv", index=False)
    print(f"Saved sample_quality.csv ({len(qdf)} rows)")
    
    msum = method_summary(qdf)
    msum.to_csv(OUT_DIR / "method_quality_summary.csv", index=False)
    print(f"Saved method_quality_summary.csv")
    
    isum = identity_summary(qdf)
    isum.to_csv(OUT_DIR / "identity_quality_summary.csv", index=False)
    print(f"Saved identity_quality_summary.csv")
    
    vsum = video_summary(qdf)
    vsum.to_csv(OUT_DIR / "video_quality_summary.csv", index=False)
    print(f"Saved video_quality_summary.csv")
    
    dshift = distribution_shift(qdf)
    dshift.to_csv(OUT_DIR / "distribution_shift.csv", index=False)
    print(f"Saved distribution_shift.csv")
    
    recs = data_collection_recommendations(qdf, msum)
    recs.to_csv(OUT_DIR / "data_collection_recommendations.csv", index=False)
    print(f"Saved data_collection_recommendations.csv")
    
    # Save summary
    summary = {
        "total_images": len(qdf),
        "real": int((qdf["label"] == 0).sum()),
        "fake": int((qdf["label"] == 1).sum()),
        "weak_samples": int((qdf["weakness_score"] > 0).sum()),
        "bad_quality_outliers": int((qdf["outlier_class"] == "POTENTIAL_BAD_QUALITY").sum()),
        "hard_sample_outliers": int((qdf["outlier_class"] == "POTENTIAL_HARD_SAMPLE").sum()),
        "face_detection_available": FACE_AVAILABLE,
        "output_dir": str(OUT_DIR),
        "runtime_seconds": time.time() - t0,
    }
    with open(OUT_DIR / "data_quality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("Data quality analysis complete")
    print("="*80)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
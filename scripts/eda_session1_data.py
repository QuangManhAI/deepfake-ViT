#!/usr/bin/env python3
"""Session 1 — DF40 data aggregation.

Aggregates authoritative numbers from local manifests / feature memmaps and
computes pixel + identity statistics from locally available image frames.
Writes agents/figures/session1/summary.json for the plotting script.

RAM-conscious: samples a small image subset, never loads the whole dataset.
"""
import os, json, collections, random
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "agents", "figures", "session1")
os.makedirs(FIG, exist_ok=True)
rng = random.Random(42)

# --------------------------------------------------------------------------
# 1. Feature memmaps (test_data_v3): train split (21,459) + rest (9,232) = full 30,691
# --------------------------------------------------------------------------
def load_memmap_meta(split):
    metas = []
    for m in ("vit", "cnn"):
        p = os.path.join(ROOT, "outputs", "features", f"test_data_v3_{m}{split}.mmap.meta.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            metas.append({"labels": z["labels"], "methods": z["methods"]})
    # use the first available model's labels/methods (identical across vit/cnn)
    lab = metas[0]["labels"]; meth = metas[0]["methods"]
    return lab, meth

full_labels, full_methods = [], []
split_stats = {}
for split in ("", "_test"):
    lab, meth = load_memmap_meta(split)
    n_real = int(np.sum(lab == 0)); n_fake = int(np.sum(lab == 1))
    split_stats["train" if split == "" else "val+test"] = {
        "total": int(len(lab)), "real": n_real, "fake": n_fake,
        "methods": int(len(set(meth))),
    }
    full_labels.append(lab); full_methods.append(meth)
full_labels = np.concatenate(full_labels)
full_methods = np.concatenate(full_methods)

per_method_full = collections.Counter(m for m, l in zip(full_methods, full_labels) if l == 1)
per_method_train = collections.Counter()
for split in ("",):
    lab, meth = load_memmap_meta(split)
    per_method_train = collections.Counter(m for m, l in zip(meth, lab) if l == 1)

n_real_full = int(np.sum(full_labels == 0))
n_fake_full = int(np.sum(full_labels == 1))

# --------------------------------------------------------------------------
# 2. Manifests
# --------------------------------------------------------------------------
methods_summary = json.load(open(os.path.join(ROOT, "data/splits/methods_summary.json")))
split_info = json.load(open(os.path.join(ROOT, "data/splits/split_info.json")))

# eval json for domain + identity structure
eval_json = json.load(open(os.path.join(ROOT, "outputs/eval/identity_disjoint_v3_vit.json")))

# --------------------------------------------------------------------------
# 3. Family taxonomy (40 methods)
# --------------------------------------------------------------------------
FAMILIES = {
    "Face Swap": ["faceswap", "simswap", "inswap", "mobileswap", "facedancer",
                  "blendface", "uniface", "deepfacelab"],
    "Reenactment": ["sadtalker", "wav2lip", "fomm", "MRAA", "lia", "mcnet", "tpsm",
                    "facevid2vid", "hyperreenact", "pirender", "one_shot_free",
                    "danet", "fsgan", "heygen"],
    "Face Synthesis": ["DiT", "SiT", "StyleGAN2", "StyleGAN3", "StyleGANXL", "sd2.1",
                       "MidJourney", "CollabDiff", "pixart", "RDDM", "ddim", "VQGAN",
                       "whichfaceisreal"],
    "Face Editing": ["stargan", "starganv2", "styleclip", "e4e", "e4s"],
}
method_to_family = {m: fam for fam, ms in FAMILIES.items() for m in ms}
assert len(method_to_family) == 40, f"family taxonomy covers {len(method_to_family)}/40"

family_method_count = {fam: len(ms) for fam, ms in FAMILIES.items()}
family_fake_full = {fam: sum(per_method_full[m] for m in ms) for fam, ms in FAMILIES.items()}

# --------------------------------------------------------------------------
# 4. Local image frames: structure + identity-folder stats
# --------------------------------------------------------------------------
LOCAL_METHODS = {
    "facedancer": {"fake": 24710},
    "faceswap":   {"fake": 24634},
    "DiT":        {"fake": 31885},
    "ddim":       {"fake": 31885},
    "CollabDiff": {"real": 1000, "fake": 1000},
    "deepfacelab": {"real": 1547, "fake": 3122},
}
IMG_EXTS = (".png", ".jpg", ".jpeg")

def walk_img(d):
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                yield os.path.join(root, fn)

# identity folder stats for frame-based methods (frames grouped under <idA_idB>/<vidid>)
identity_stats = {}
for method in ["facedancer", "faceswap", "DiT", "ddim"]:
    base = os.path.join(ROOT, "data", method)
    frames = list(walk_img(base))
    ident_dirs = set()
    for f in frames:
        # identity folder = the leaf dir containing the frames
        ident_dirs.add(os.path.dirname(f))
    n_ident = len(ident_dirs)
    identity_stats[method] = {
        "n_frames": len(frames),
        "n_identity_dirs": n_ident,
        "frames_per_identity_mean": round(len(frames) / n_ident, 1) if n_ident else 0,
        "max_frames_per_identity": max((sum(1 for f in frames if os.path.dirname(f) == d) for d in ident_dirs), default=0),
    }

# --------------------------------------------------------------------------
# 5. Pixel statistics from a sampled subset
# --------------------------------------------------------------------------
PIXEL_SAMPLE = {
    "real/CollabDiff": ("data/CollabDiff/real", 300),
    "real/deepfacelab": ("data/deepfacelab/real", 300),
    "fake/CollabDiff": ("data/CollabDiff/fake", 300),
    "fake/deepfacelab": ("data/deepfacelab/fake", 300),
    "fake/facedancer": ("data/facedancer", 300),
    "fake/faceswap": ("data/faceswap", 300),
    "fake/DiT": ("data/DiT", 300),
    "fake/ddim": ("data/ddim", 300),
}

def laplacian_var(gray):
    """approx Laplacian variance (sharpness) via np.diff convolution."""
    lap = (np.roll(gray, 1, 0) + np.roll(gray, -1, 0) +
           np.roll(gray, 1, 1) + np.roll(gray, -1, 1) - 4 * gray)
    return float(np.var(lap))

pixel = {}
size_counts = collections.Counter()
hist_data = {}  # raw arrays for histogram plotting
for key, (rel, cap) in PIXEL_SAMPLE.items():
    d = os.path.join(ROOT, rel)
    files = list(walk_img(d))
    sel = rng.sample(files, min(cap, len(files)))
    brightness, sharpness, sizes = [], [], []
    for f in sel:
        try:
            im = Image.open(f).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        sizes.append((w, h))
        size_counts[(w, h)] += 1
        a = np.asarray(im.resize((64, 64)))  # downsample for speed
        gray = a.mean(axis=2)
        brightness.append(float(gray.mean()))
        sharpness.append(laplacian_var(gray))
    hist_data[key] = np.array(brightness)
    hist_data[key + "__sharp"] = np.array(sharpness)
    pixel[key] = {
        "n": len(brightness),
        "brightness_mean": round(np.mean(brightness), 2),
        "brightness_std": round(np.std(brightness), 2),
        "sharpness_mean": round(np.mean(sharpness), 1),
        "sharpness_median": round(np.median(sharpness), 1),
        "sizes": dict(collections.Counter(sizes)),
    }
    print(f"{key:>20}: n={len(brightness):>3}  bright={np.mean(brightness):.1f}±{np.std(brightness):.1f}  "
          f"sharp={np.mean(sharpness):.0f} med={np.median(sharpness):.0f}  sizes={dict(collections.Counter(sizes))}")

# consistency: per_method_full should sum to the full fake count
assert sum(per_method_full.values()) == n_fake_full, \
    f"per-method full {sum(per_method_full.values())} != fake {n_fake_full}"

# --------------------------------------------------------------------------
# 6. Gallery sample paths (6 methods fake + real) for the grid figures
# --------------------------------------------------------------------------
def sample_gallery(rel, k):
    d = os.path.join(ROOT, rel)
    files = list(walk_img(d))
    return [os.path.abspath(f) for f in rng.sample(files, min(k, len(files)))]

gallery = {
    "fake": {m: sample_gallery(f"data/{m}", 3) for m in ["facedancer", "faceswap", "DiT", "ddim", "CollabDiff", "deepfacelab"]},
    "real_CollabDiff": sample_gallery("data/CollabDiff/real", 4),
    "real_deepfacelab": sample_gallery("data/deepfacelab/real", 4),
    "fake_CollabDiff_pairs": sample_gallery("data/CollabDiff/fake", 4),
    "fake_deepfacelab_pairs": sample_gallery("data/deepfacelab/fake", 4),
}

# --------------------------------------------------------------------------
# 7. Compose summary
# --------------------------------------------------------------------------
summary = {
    "test_data_v3_full": {"total": len(full_labels), "real": n_real_full, "fake": n_fake_full,
                          "methods": int(len(set(full_methods)))},
    "split_stats": split_stats,
    "split_info_identity_disjoint": split_info["identity_disjoint_splits"],
    "test_full_total_info": split_info.get("test_full_total"),
    "total_methods_info": split_info.get("total_methods"),
    "per_method_full": dict(per_method_full),
    "per_method_train_split": dict(per_method_train),
    "methods_summary": methods_summary,
    "eval_domain": eval_json["per_domain"],
    "eval_identity": {
        "n_identity_keys": eval_json["n_identity_keys"],
        "n_train": eval_json["n_train"],
        "n_test": eval_json["n_test"],
        "paired_only": eval_json["paired_only"],
    },
    "family_method_count": family_method_count,
    "family_fake_full": family_fake_full,
    "method_to_family": method_to_family,
    "identity_stats_local": identity_stats,
    "pixel": {k: {**v, "sizes": {"_".join(map(str, s)): n for s, n in v["sizes"].items()}} for k, v in pixel.items()},
    "size_counts": {"_".join(map(str, k)): v for k, v in size_counts.items()},
    "gallery": gallery,
}
out = os.path.join(FIG, "summary.json")
json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
np.savez_compressed(os.path.join(FIG, "hist_data.npz"), **hist_data)
print(f"\nWrote {out} + hist_data.npz")

print("\n=== summary key numbers ===")
print(f"test_data_v3 full: {summary['test_data_v3_full']}")
print(f"identity-disjoint splits: {summary['split_info_identity_disjoint']}")
print(f"family fake full: {summary['family_fake_full']}")
print(f"identity stats local: {json.dumps(identity_stats, indent=1)}")

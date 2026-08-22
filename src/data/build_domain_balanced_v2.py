"""Build Domain-Balanced Dataset Split V2 for EXP-02.

Features:
- 30,000 Real (20,000 FF++ + 10,000 Celeb-DF Real)
- 30,000 Fake (10,000 Celeb-DF Fake + 20,000 DF40 Fake with oversampled Diffusion & GAN methods)
- Strict zero-leakage check against val_domain_balanced.csv and test_balanced.csv
"""
import glob
import os
import random
import pandas as pd
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

splits_dir = "data/splits"
val_csv = os.path.join(splits_dir, "val_domain_balanced.csv")
test_csv = os.path.join(splits_dir, "test_balanced.csv")

df_val = pd.read_csv(val_csv)
df_test = pd.read_csv(test_csv)

val_test_paths = set(df_val['path']).union(set(df_test['path']))
print(f"Total Val+Test paths to exclude: {len(val_test_paths):,}")

# 1. Real Samples Pool
ff_frames = sorted(glob.glob("/workspace/data/FaceForensics++/original_sequences/youtube/c23/frames/*/*.png"))
celeb_real_frames = sorted(glob.glob("/workspace/hoangtuan/deepfake-ViT/data/processed/celeb_df_extracted/*.png"))

ff_train_pool = [p for p in ff_frames if p not in val_test_paths]
celeb_real_train_pool = [p for p in celeb_real_frames if p not in val_test_paths]

print(f"Available FF++ Real (train): {len(ff_train_pool):,}")
print(f"Available Celeb Real (train): {len(celeb_real_train_pool):,}")

random.shuffle(ff_train_pool)
random.shuffle(celeb_real_train_pool)

selected_real_ff = ff_train_pool[:20000]
selected_real_celeb = celeb_real_train_pool[:10000]
selected_real = [(p, 0) for p in selected_real_ff + selected_real_celeb]
print(f"Selected Real: {len(selected_real):,} ({len(selected_real_ff)} FF++ + {len(selected_real_celeb)} Celeb-DF)")

# 2. Fake Samples Pool
celeb_train_fake = sorted(glob.glob("/workspace/hoangtuan/deepfake-ViT/data/processed/celeb_df_train_fake_extracted/*.png"))
celeb_fake_train_pool = [p for p in celeb_train_fake if p not in val_test_paths]
random.shuffle(celeb_fake_train_pool)
selected_fake_celeb = celeb_fake_train_pool[:10000]

# DF40 Fake Methods Pool
manifest_df = pd.read_csv("/workspace/data/DF40_train_manifest.csv")
manifest_fake = manifest_df[manifest_df['label'] == 1]
manifest_fake = manifest_fake[~manifest_fake['path'].isin(val_test_paths)]

# Group by method and sample
diffusion_methods = ['DiT', 'SiT', 'sd2.1', 'pixart', 'ddim', 'RDDM']
gan_methods = ['StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'VQGAN']
swap_methods = ['simswap', 'faceswap', 'mobileswap', 'inswap', 'blendface', 'facedancer', 'fsgan', 'uniface']
reenact_methods = ['sadtalker', 'wav2lip', 'fomm', 'pirender', 'tpsm', 'lia', 'MRAA', 'danet', 'mcnet', 'facevid2vid', 'e4s', 'one_shot_free', 'hyperreenact']

selected_df40_fake = []

# Target counts per category:
# Diffusion: 6 methods * 850 = 5,100
# GAN: 4 methods * 850 = 3,400
# Swap: 8 methods * 600 = 4,800
# Reenact: 13 methods * 515 = 6,700
# Total DF40 Fake = 20,000

for m in diffusion_methods:
    paths = manifest_fake[manifest_fake['method'] == m]['path'].tolist()
    random.shuffle(paths)
    selected_df40_fake.extend(paths[:850])

for m in gan_methods:
    paths = manifest_fake[manifest_fake['method'] == m]['path'].tolist()
    random.shuffle(paths)
    selected_df40_fake.extend(paths[:850])

for m in swap_methods:
    paths = manifest_fake[manifest_fake['method'] == m]['path'].tolist()
    random.shuffle(paths)
    n_sample = min(len(paths), 600)
    selected_df40_fake.extend(paths[:n_sample])

for m in reenact_methods:
    paths = manifest_fake[manifest_fake['method'] == m]['path'].tolist()
    random.shuffle(paths)
    n_sample = min(len(paths), 515)
    selected_df40_fake.extend(paths[:n_sample])

# Fill remaining to exactly 20,000 if needed
if len(selected_df40_fake) < 20000:
    diff = 20000 - len(selected_df40_fake)
    remaining_pool = [p for p in manifest_fake['path'].tolist() if p not in set(selected_df40_fake)]
    random.shuffle(remaining_pool)
    selected_df40_fake.extend(remaining_pool[:diff])
else:
    selected_df40_fake = selected_df40_fake[:20000]

selected_fake = [(p, 1) for p in selected_fake_celeb + selected_df40_fake]
print(f"Selected Fake: {len(selected_fake):,} ({len(selected_fake_celeb)} Celeb-DF + {len(selected_df40_fake)} DF40)")

# Combine & Shuffle
all_train = selected_real + selected_fake
random.shuffle(all_train)

df_train_v2 = pd.DataFrame(all_train, columns=['path', 'label'])
out_csv = os.path.join(splits_dir, "train_domain_balanced_v2.csv")
df_train_v2.to_csv(out_csv, index=False)
print(f"Saved {out_csv} with {len(df_train_v2):,} samples (Real: {(df_train_v2['label']==0).sum():,}, Fake: {(df_train_v2['label']==1).sum():,})")

# Verification
train_paths = set(df_train_v2['path'])
assert len(train_paths & set(df_val['path'])) == 0, "ERROR: Leakage with Val set!"
assert len(train_paths & set(df_test['path'])) == 0, "ERROR: Leakage with Test set!"
print("✅ ZERO DATA LEAKAGE VERIFIED successfully!")

import glob, os, random, pandas as pd, numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

base_dir = os.path.dirname(os.path.abspath(__file__))
splits_dir = os.path.join(base_dir, "data", "splits")
val_csv = os.path.join(splits_dir, "val_domain_balanced.csv")
test_csv = os.path.join(splits_dir, "test_balanced.csv")

df_val = pd.read_csv(val_csv)
df_test = pd.read_csv(test_csv)
val_test_paths = set(df_val['path']).union(set(df_test['path']))
print(f"Total Val+Test paths to exclude: {len(val_test_paths):,}")

ff_frames = sorted(glob.glob("/workspace/data/FaceForensics++/original_sequences/youtube/c23/frames/*/*.png"))
celeb_real_frames = sorted(glob.glob("/workspace/hoangtuan/deepfake-ViT/data/processed/celeb_df_extracted/*.png"))

ff_train_pool = [p for p in ff_frames if p not in val_test_paths]
celeb_real_train_pool = [p for p in celeb_real_frames if p not in val_test_paths]

random.shuffle(ff_train_pool)
random.shuffle(celeb_real_train_pool)

selected_real_ff = ff_train_pool[:20000]
selected_real_celeb = celeb_real_train_pool[:10000]
selected_real = [(p, 0) for p in selected_real_ff + selected_real_celeb]

celeb_train_fake = sorted(glob.glob("/workspace/hoangtuan/deepfake-ViT/data/processed/celeb_df_train_fake_extracted/*.png"))
celeb_fake_train_pool = [p for p in celeb_train_fake if p not in val_test_paths]
random.shuffle(celeb_fake_train_pool)
selected_fake_celeb = celeb_fake_train_pool[:10000]

manifest_df = pd.read_csv("/workspace/data/DF40_train_manifest.csv")
manifest_fake = manifest_df[manifest_df['label'] == 1]
manifest_fake = manifest_fake[~manifest_fake['path'].isin(val_test_paths)]

diffusion_methods = ['DiT', 'SiT', 'sd2.1', 'pixart', 'ddim', 'RDDM']
gan_methods = ['StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'VQGAN']
swap_methods = ['simswap', 'faceswap', 'mobileswap', 'inswap', 'blendface', 'facedancer', 'fsgan', 'uniface']
reenact_methods = ['sadtalker', 'wav2lip', 'fomm', 'pirender', 'tpsm', 'lia', 'MRAA', 'danet', 'mcnet', 'facevid2vid', 'e4s', 'one_shot_free', 'hyperreenact']

selected_df40_fake = []
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
    selected_df40_fake.extend(paths[:min(len(paths), 600)])

for m in reenact_methods:
    paths = manifest_fake[manifest_fake['method'] == m]['path'].tolist()
    random.shuffle(paths)
    selected_df40_fake.extend(paths[:min(len(paths), 515)])

if len(selected_df40_fake) < 20000:
    diff = 20000 - len(selected_df40_fake)
    remaining_pool = [p for p in manifest_fake['path'].tolist() if p not in set(selected_df40_fake)]
    random.shuffle(remaining_pool)
    selected_df40_fake.extend(remaining_pool[:diff])
else:
    selected_df40_fake = selected_df40_fake[:20000]

selected_fake = [(p, 1) for p in selected_fake_celeb + selected_df40_fake]

all_train = selected_real + selected_fake
random.shuffle(all_train)

df_train_v2 = pd.DataFrame(all_train, columns=['path', 'label'])
out_csv = os.path.join(splits_dir, "train_domain_balanced_v2.csv")
df_train_v2.to_csv(out_csv, index=False)
print(f"SUCCESS: Saved {out_csv} ({len(df_train_v2):,} samples)")

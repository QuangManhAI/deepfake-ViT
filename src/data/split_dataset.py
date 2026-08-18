"""Chia dataset DeepFakeFace (data/hug) thành train / val / test THEO NHÂN VẬT.

Lý do chia theo nhân vật (folder 00–99): cùng 1 người xuất hiện ở cả 4 category
(wiki=real, insight/inpainting/text2img=fake). Nếu chia ngẫu nhiên theo ảnh, model
sẽ học gộp 1 người ở cả train lẫn test → điểm bị đánh giá cao ảo.

Tỷ lệ: 100 nhân vật → 72 train / 8 val / 20 test (test = 20%).

Output (data/splits/*.csv): cột `path,label` (label: 0=real, 1=fake), path tương đối
so với thư mục project. Không nhân đôi ảnh — chỉ là danh sách đường dẫn.
"""
import argparse
import csv
import os
import random
from collections import defaultdict

# wiki = ảnh thật (real), 3 category còn lại = deepfake (fake)
CATEGORIES = {"wiki": 0, "insight": 1, "inpainting": 1, "text2img": 1}
EXTENSIONS = (".jpg", ".jpeg", ".png")


def discover_samples(data_dir: str):
    """Gom toàn bộ ảnh theo nhân vật. Trả về dict {identity: [(path, label), ...]}."""
    samples = defaultdict(list)
    for cat, label in CATEGORIES.items():
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.isdir(cat_dir):
            print(f"[Cảnh báo] Không có thư mục: {cat_dir}")
            continue
        for ident in sorted(os.listdir(cat_dir)):
            idir = os.path.join(cat_dir, ident)
            if not os.path.isdir(idir):
                continue
            for f in sorted(os.listdir(idir)):
                if f.lower().endswith(EXTENSIONS):
                    samples[ident].append((os.path.join(cat_dir, ident, f), label))
    return samples


def write_csv(out_path: str, rows):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Chia dataset theo nhân vật")
    parser.add_argument("--data-dir", default="data/hug")
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--test-identities", type=int, default=20, help="Số nhân vật làm test (20 = 20%)")
    parser.add_argument("--val-identities", type=int, default=8, help="Số nhân vật làm val (lấy từ phần train)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = discover_samples(args.data_dir)
    idents = sorted(samples.keys())
    print(f"Nhân vật: {len(idents)} | Tổng ảnh: {sum(len(v) for v in samples.values())}")

    # ---- Chia ngẫu nhiên theo nhân vật (có seed để tái lập) ----
    rng = random.Random(args.seed)
    shuffled = idents[:]
    rng.shuffle(shuffled)

    test_ids = set(shuffled[: args.test_identities])
    rest = shuffled[args.test_identities:]
    val_ids = set(rest[: args.val_identities])
    train_ids = set(rest[args.val_identities:])

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    print(f"\nSplit: train={len(train_ids)} nhân vật | val={len(val_ids)} | test={len(test_ids)}")

    # ---- Xuất CSV ----
    for name, id_set in splits.items():
        rows = []
        for ident in sorted(id_set):
            rows.extend(samples[ident])
        # path tương đối so với project (bỏ tiền tố "./")
        rows = [(p[2:] if p.startswith("./") else p, lbl) for p, lbl in rows]
        write_csv(os.path.join(args.out_dir, f"{name}.csv"), rows)

        n_real = sum(1 for _, lbl in rows if lbl == 0)
        n_fake = sum(1 for _, lbl in rows if lbl == 1)
        print(f"  {name:>5}: {len(rows):>7} ảnh | real={n_real:>6} | fake={n_fake:>6} (tỷ lệ 1:{n_fake / n_real:.1f})")

    # ---- Lưu thông tin split ----
    summary = {name: sorted(s) for name, s in splits.items()}
    import json
    with open(os.path.join(args.out_dir, "split_info.json"), "w") as f:
        json.dump({"seed": args.seed, "splits": summary}, f, indent=2)
    print(f"\nĐã lưu CSV + split_info.json vào {args.out_dir}/")


if __name__ == "__main__":
    main()

"""Tạo split CÂN BẰNG 1:1 từ các CSV split đầy đủ (data/splits/*.csv).

Cân bằng bằng cách chỉ giữ:
  - real = wiki (toàn bộ ảnh thật)
  - fake = 1 loại duy nhất (mặc định: insight — swap face)

Identity vẫn tách biệt giữa train/val/test (kế thừa từ split gốc theo nhân vật).

Output: data/splits/{train,val,test}_{fake_type}.csv
"""
import argparse
import csv
import os
import re

EXTENSIONS = (".jpg", ".jpeg", ".png")
REAL_CAT = "wiki"


def keep_row(path: str, fake_type: str) -> bool:
    """Giữ ảnh nếu thuộc real (wiki) hoặc đúng loại fake được chọn."""
    return f"/hug/{REAL_CAT}/" in path or f"/hug/{fake_type}/" in path


def main():
    parser = argparse.ArgumentParser(description="Tạo split cân bằng 1:1")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--fake-type", default="insight", choices=["insight", "inpainting", "text2img"])
    parser.add_argument("--out-dir", default="data/splits")
    args = parser.parse_args()

    for name in ("train", "val", "test"):
        src = os.path.join(args.splits_dir, f"{name}.csv")
        out = os.path.join(args.out_dir, f"{name}_{args.fake_type}.csv")

        kept, dropped = [], 0
        with open(src, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                path, label = row[0], int(row[1])
                if not path.lower().endswith(EXTENSIONS):
                    continue
                if keep_row(path, args.fake_type):
                    kept.append((path, label))
                else:
                    dropped += 1

        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(kept)

        n_real = sum(1 for _, lbl in kept if lbl == 0)
        n_fake = sum(1 for _, lbl in kept if lbl == 1)
        print(f"  {name:>5}: {len(kept):>6} ảnh (giữ) | real={n_real} fake={n_fake} | bỏ {dropped} loại fake khác")

    print(f"\nĐã tạo split cân bằng (real=wiki, fake={args.fake_type}) vào {args.out_dir}/")


if __name__ == "__main__":
    main()

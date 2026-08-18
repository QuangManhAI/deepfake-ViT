"""Build test_data_v2 (MỞ RỘNG 31 method): bộ test identity-unique.

Mỗi khuôn mặt xuất hiện DUY NHẤT 1 lần; phân chia theo IDENTITY key (cột `identity`
trong manifest), real+fake cùng 1 người luôn cùng 1 nhánh split.

REAL (identity-unique, 1 frame/người):
  - Celeb-DF: 178 identity (data/raw/real-root/Celeb-DF-v2)
  - FF++   : 999 identity (data/FaceForensics++)
  Tổng 1.177.

FAKE (1 frame / (method, identity)), từ 31 method có data:
  [A] 20 method cdf trên Air  (MRAA, blendface, danet, e4s, facedancer, faceswap,
       facevid2vid, fomm, fsgan, hyperreenact, inswap, lia, mcnet, one_shot_free,
       pirender, sadtalker, simswap, tpsm, uniface, wav2lip)
       identity = idN_M cuối (`id0_id16_0003` -> `id16_0003`) -> "cdc:..." nếu có real,
       ngược lại "oth:<method>:<dir>"
  [B] 18 method ff trên Air (dạng A_B, identity = id ĐẦU — đã verify) -> "ffc:A"
  [C] 3 method tổng hợp local (StyleGAN2/VQGAN/ddim, dir toàn số, mặt ảo) -> "efs:..."
  [D] 5 method df40_subset (CollabDiff/MidJourney/deepfacelab/heygen/whichfaceisreal)
       -> "efs:<method>:<id>" (mặt sinh, không ghép real)
  [E] 3 method FE (stargan/starganv2/styleclip — ảnh CelebA, 1 ảnh = 1 identity) -> "fe:..."

Frame matching: real chọn frame giữa; fake ưu tiên frame CÙNG index (nếu có), fallback
frame giữa. Sau copy: verify decode từng file (loại ảnh hỏng khỏi manifest).

Chạy:
  .venv/bin/python src/data/build_test_data_v2.py --dry-run   # chỉ đếm
  .venv/bin/python src/data/build_test_data_v2.py             # build + copy
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from PIL import Image

print = partial(print, flush=True)

OUT = "test_data_v2"
MANIFEST = os.path.join(OUT, "manifest.csv")
CELEB_REAL = "data/raw/real-root/Celeb-DF-v2"
FF_REAL = "data/FaceForensics++/original_sequences/youtube/c23/frames"
AIR = "/Volumes/quangmanh/Downloads/DF40"
DF40_SUB = "data/df40_subset"

# [A] 20 method có cdf trên Air
CDF_METHODS = ["MRAA", "blendface", "danet", "e4s", "facedancer", "faceswap",
               "facevid2vid", "fomm", "fsgan", "hyperreenact", "inswap", "lia",
               "mcnet", "one_shot_free", "pirender", "sadtalker", "simswap",
               "tpsm", "uniface", "wav2lip"]
# [B] 18 method ff dạng A_B (loại sadtalker/wav2lip — ff không dạng A_B)
FF_METHODS = ["MRAA", "blendface", "danet", "e4s", "facedancer", "faceswap",
              "facevid2vid", "fomm", "fsgan", "hyperreenact", "inswap", "lia",
              "mcnet", "one_shot_free", "pirender", "simswap", "tpsm", "uniface"]
# [C] tổng hợp local (dir toàn số, mặt ảo)
SYN_LOCAL = ["StyleGAN2", "VQGAN", "ddim"]
# [D] synthesis từ df40_subset (không có cdf/ff trên Air) — fake base dir (có layout khác nhau)
SYN_DF40 = {
    "CollabDiff": "data/df40_subset/CollabDiff/CollabDiff/fake",
    "MidJourney": "data/df40_subset/MidJourney/MidJourney/fake",
    "whichfaceisreal": "data/df40_subset/whichfaceisreal/whichfaceisreal/fake",
    "deepfacelab": "data/df40_subset/deepfacelab/deepfacelab/fake/frames",
    "heygen": "data/df40_subset/heygen/heygen/heygen_new/fake/frames",
}
# [E] FE: ảnh CelebA đơn lẻ
FE = ["stargan", "starganv2", "styleclip"]
# [F] 7 method cấu trúc mới trên Air: cdf/Fake_from_{Celeb,YouTube}-real (hoặc cdf/{Celeb,YouTube}-real)
#     + ff/<id ĐƠN> (140 dir, id = FF++). Frame name NGẪU NHIÊN (11287.png, sample-*.png, seed*.png)
#     -> phải listdir chọn frame (không probe được).
#     cdf_ident: "dir_id"    = identity lấy từ tên dir (idN_M / 5-số)
#                "frame_id"  = identity lấy từ tên FRAME ('Celeb-real_id13_0011_045.png')
#                "synthetic" = frame seed* = mặt tổng hợp -> efs (KHÔNG ghép real)
#     ff_kind:   "real" = ghép FF++ identity (ffc); "synthetic" = seed* -> efs
NEW_CDF_FF = {
    "DiT":        ("dir_id", "real"),
    "RDDM":       ("dir_id", "real"),
    "SiT":        ("dir_id", "real"),
    "StyleGAN3":  ("synthetic", "synthetic"),
    "StyleGANXL": ("synthetic", "synthetic"),
    "pixart":     ("frame_id", "real"),
    "sd2.1":      ("frame_id", "real"),
}
# [G] e4e: e4e/e4e/ff/<id>/ (1000 dir), frame 00001.jpg tuần tự -> probe được
E4E_FF = os.path.join(AIR, "e4e", "e4e", "ff")
# [H] mobileswap: zip trên Air -> unzip local, rồi xử lý như cdf/ff chuẩn
MS_ZIP_CDF = os.path.join(AIR, "mobileswap", "cdf", "frames.zip")
MS_ZIP_FF = os.path.join(AIR, "mobileswap", "ff", "frames.zip")
MS_STAGE = "data/raw/mobileswap_ext"

IDN_M = re.compile(r"id\d+_\d+")
FRAME_ID_RE = re.compile(r"(?:Celeb-real|YouTube-real)_(id\d+_\d+|\d+)_\d+\.png$")
SKIP = {".DS_Store"}
FAST = False   # --fast: chỉ đếm, không chạm từng dir (đỡ chậm trên SMB)


def mid_frame(d):
    if FAST:
        return "FAST.png"
    fs = sorted(f for f in os.listdir(d) if f not in SKIP and os.path.isfile(os.path.join(d, f)))
    return fs[len(fs) // 2] if fs else None


# Frame chuẩn trên Air cdf/ff (verified: mọi dir có `000.png`/`000000.png`/`00000000.png`).
# `isfile` probe ~2ms vs `os.listdir` 6-125ms trên SMB -> chọn bằng probe để đỡ chậm.
PROBE_NAMES = ["000.png", "000000.png", "00000000.png", "000001.png"]


def probe_frame(d, prefer=None):
    """Tìm 1 frame hợp lệ KHÔNG listdir (SMB chậm). prefer = frame index khớp real
    (matched frame, cùng source video); fallback probe `000*.png`; cuối cùng listdir."""
    if FAST:
        return "FAST.png"
    if prefer is not None and os.path.isfile(os.path.join(d, prefer)):
        return prefer
    for name in PROBE_NAMES:
        if os.path.isfile(os.path.join(d, name)):
            return name
    return mid_frame(d)


def celeb_ids():
    ids = set()
    for sub in ("YouTube-real", "Celeb-real"):
        base = os.path.join(CELEB_REAL, sub, "frames")
        if os.path.isdir(base):
            ids.update(os.listdir(base))
    return ids


def ff_ids():
    if os.path.isdir(FF_REAL):
        return set(os.listdir(FF_REAL))
    return set()


def celeb_identity(video):
    ms = IDN_M.findall(video)
    return ms[-1] if ms else None


def safe(name):
    return name.replace(":", "_")


def isdir(p):
    return True if FAST else os.path.isdir(p)


def main():
    global FAST
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="chỉ đếm nhanh (không chạm từng dir — dùng cho --dry-run)")
    ap.add_argument("--no-verify", action="store_true", help="bỏ bước verify decode (nhanh)")
    ap.add_argument("--scan-only", action="store_true",
                    help="chỉ liệt kê + chọn frame bằng probe thật, KHÔNG copy (đo tốc độ)")
    args = ap.parse_args()
    FAST = args.fast or args.dry_run

    cdc_real = celeb_ids()
    ffc_real = ff_ids()
    print(f"Real pool: Celeb-DF={len(cdc_real)} | FF++={len(ffc_real)}")

    items = []          # dict: method, video, src, path, identity, domain, label
    seen = set()
    counts = {"cdc": 0, "ffc": 0, "efs": 0, "oth": 0, "fe": 0}

    def add(method, video, src, identity, domain):
        key = (method, identity)
        if key in seen:
            return
        seen.add(key)
        # path có method để KHÔNG đè khi nhiều method cùng identity chọn cùng frame
        items.append({"method": method, "video": video, "src": src,
                      "path": f"{method}__{safe(identity)}__{os.path.basename(src)}",
                      "identity": f"{domain}:{identity}", "domain": domain,
                      "label": 0 if method == "real" else 1})
        counts[domain] += 1

    real_sel = {}

    # ---------- REAL ----------
    for sub in ("YouTube-real", "Celeb-real"):
        base = os.path.join(CELEB_REAL, sub, "frames")
        if not os.path.isdir(base):
            continue
        for vid in sorted(os.listdir(base)):
            d = os.path.join(base, vid)
            if not os.path.isdir(d):
                continue
            fr = mid_frame(d)
            if not fr:
                continue
            add("real", vid, os.path.join(d, fr), vid, "cdc")
            real_sel[("cdc", vid)] = fr
    for vid in sorted(os.listdir(FF_REAL)):
        d = os.path.join(FF_REAL, vid)
        if not os.path.isdir(d):
            continue
        fr = mid_frame(d)
        if not fr:
            continue
        add("real", vid, os.path.join(d, fr), vid, "ffc")
        real_sel[("ffc", vid)] = fr
    n_r = sum(1 for it in items if it["label"] == 0)
    print(f"Real: {counts['cdc']} Celeb-DF + {counts['ffc']} FF++ = {n_r}")

    # ---------- [A] cdf Air ----------
    for method in CDF_METHODS:
        base = os.path.join(AIR, method, "cdf", "frames")
        if not os.path.isdir(base):
            print(f"  ! {method}: thiếu cdf Air — bỏ")
            continue
        for vid in sorted(os.listdir(base)):
            if vid in SKIP:
                continue
            fdir = os.path.join(base, vid)
            if not isdir(fdir):
                continue
            ident = celeb_identity(vid)
            if ident is not None and ident in cdc_real:
                dom = "cdc"
            else:
                ident, dom = f"{method}:{vid}", "oth"
            prefer = real_sel.get(("cdc", ident)) if dom == "cdc" else None
            fr = probe_frame(fdir, prefer)
            if not fr:
                continue
            add(method, vid, os.path.join(fdir, fr), ident, dom)

    # ---------- [B] ff Air (A_B) ----------
    for method in FF_METHODS:
        base = os.path.join(AIR, method, "ff", "frames")
        if not os.path.isdir(base):
            continue
        for vid in sorted(os.listdir(base)):
            if not re.fullmatch(r"\d+_\d+", vid):
                continue
            ident = vid.split("_")[0]
            if ident not in ffc_real:
                continue
            fdir = os.path.join(base, vid)
            if not isdir(fdir):
                continue
            fr = probe_frame(fdir, real_sel.get(("ffc", ident)))
            if not fr:
                continue
            add(method, vid, os.path.join(fdir, fr), ident, "ffc")

    # ---------- [C] synthesis local (test_data) ----------
    for method in SYN_LOCAL:
        base = f"test_data/{method}/fake"
        if not os.path.isdir(base):
            print(f"  ! {method}: thiếu local — bỏ")
            continue
        for vid in sorted(os.listdir(base)):
            fdir = os.path.join(base, vid)
            if not isdir(fdir):
                continue
            fr = mid_frame(fdir)
            if not fr:
                continue
            add(method, vid, os.path.join(fdir, fr), f"{method}:{vid}", "efs")

    # ---------- [D] synthesis df40_subset ----------
    for method, base in SYN_DF40.items():
        if not os.path.isdir(base):
            print(f"  ! {method}: thiếu df40_subset fake — bỏ")
            continue
        for vid in sorted(os.listdir(base)):
            fdir = os.path.join(base, vid)
            if os.path.isfile(fdir):
                # layout phẳng: 1 file = 1 ảnh fake (MidJourney/whichfaceisreal)
                if not vid.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                add(method, vid, fdir, f"{method}:{vid}", "efs")
                continue
            if not isdir(fdir):
                continue
            fr = mid_frame(fdir)
            if not fr:
                continue
            add(method, vid, os.path.join(fdir, fr), f"{method}:{vid}", "efs")

    # ---------- [E] FE local (ảnh đơn lẻ) ----------
    for method in FE:
        base = f"test_data/{method}/fake"
        if not os.path.isdir(base):
            print(f"  ! {method}: thiếu local — bỏ")
            continue
        for f in sorted(os.listdir(base)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            vid = os.path.splitext(f)[0]
            add(method, vid, os.path.join(base, f), f"{method}:{vid}", "fe")

    # ---------- [F] 7 method cấu trúc mới (DiT/RDDM/SiT/StyleGAN3/StyleGANXL/pixart/sd2.1) ----------
    for method, (cdf_ident, ff_kind) in NEW_CDF_FF.items():
        base_dir = os.path.join(AIR, method, "cdf")
        if not os.path.isdir(base_dir):
            print(f"  ! {method}: thiếu cdf Air — bỏ")
            continue
        for sub in ("Fake_from_Celeb-real", "Celeb-real",
                    "Fake_from_Youtube-real", "YouTube-real"):
            base = os.path.join(base_dir, sub)
            if not isdir(base):
                continue
            for vid in sorted(os.listdir(base)):
                fdir = os.path.join(base, vid)
                if not isdir(fdir):
                    continue
                if cdf_ident == "synthetic":
                    # StyleGAN3/XL: seed* = mặt tổng hợp, KHÔNG ghép real
                    fr = mid_frame(fdir)
                    if not fr:
                        continue
                    add(method, f"{sub}:{vid}", os.path.join(fdir, fr),
                        f"{method}:{sub}:{vid}", "efs")
                elif cdf_ident == "dir_id":
                    ident = vid
                    dom = "cdc" if ident in cdc_real else "oth"
                    if dom == "oth":
                        ident = f"{method}:{vid}"
                    fr = mid_frame(fdir)
                    if not fr:
                        continue
                    add(method, vid, os.path.join(fdir, fr), ident, dom)
                else:
                    # frame_id (pixart/sd2.1): identity trích từ tên frame
                    fs = [f for f in os.listdir(fdir) if FRAME_ID_RE.match(f)]
                    if not fs:
                        fr = mid_frame(fdir)
                        if not fr:
                            continue
                        add(method, vid, os.path.join(fdir, fr), f"{method}:{vid}", "oth")
                        continue
                    fr = fs[0]
                    ident = FRAME_ID_RE.match(fr).group(1)
                    dom = "cdc" if ident in cdc_real else "oth"
                    if dom == "oth":
                        ident = f"{method}:{vid}"
                    add(method, vid, os.path.join(fdir, fr), ident, dom)
        # ff phần (id đơn = FF++ identity)
        base = os.path.join(AIR, method, "ff")
        if not isdir(base):
            continue
        for vid in sorted(os.listdir(base)):
            fdir = os.path.join(base, vid)
            if not isdir(fdir):
                continue
            if ff_kind == "synthetic":
                fr = mid_frame(fdir)
                if not fr:
                    continue
                add(method, f"ff:{vid}", os.path.join(fdir, fr), f"{method}:ff:{vid}", "efs")
            else:
                if vid not in ffc_real:
                    continue
                fr = mid_frame(fdir)
                if not fr:
                    continue
                add(method, f"ff:{vid}", os.path.join(fdir, fr), vid, "ffc")

    # ---------- [G] e4e: e4e/e4e/ff/<id>/, frame 00001.jpg — edit FF++ -> ffc paired ----------
    if isdir(E4E_FF):
        for vid in sorted(os.listdir(E4E_FF)):
            fdir = os.path.join(E4E_FF, vid)
            if not isdir(fdir):
                continue
            if vid not in ffc_real:
                continue
            fr = "00001.jpg" if os.path.isfile(os.path.join(fdir, "00001.jpg")) else mid_frame(fdir)
            if not fr:
                continue
            add("e4e", f"ff:{vid}", os.path.join(fdir, fr), vid, "ffc")
    else:
        print("  ! e4e: thiếu ff Air — bỏ")

    # ---------- [H] mobileswap: unzip Air zips -> local staging, rồi cdf/ff chuẩn ----------
    ms_cdf = ms_ff = None
    for split, zip_path in (("cdf", MS_ZIP_CDF), ("ff", MS_ZIP_FF)):
        marker = os.path.join(MS_STAGE, f"{split}_frames_dir.txt")
        if os.path.isfile(marker):
            with open(marker) as f:
                frames_dir = f.read().strip()
        elif os.path.isfile(zip_path):
            dest = os.path.join(MS_STAGE, split)
            print(f"  ...unzip mobileswap {split} ({os.path.getsize(zip_path)/1e9:.1f}GB) -> {dest}")
            os.makedirs(dest, exist_ok=True)
            subprocess.run(["unzip", "-q", "-o", zip_path, "-d", dest], check=True)
            fdirs = glob.glob(os.path.join(dest, "**", "frames"), recursive=True)
            if not fdirs:
                print(f"  ! mobileswap {split}: không tìm thấy frames/ trong zip — bỏ")
                continue
            frames_dir = fdirs[0]
            os.makedirs(MS_STAGE, exist_ok=True)
            with open(marker, "w") as f:
                f.write(frames_dir)
        else:
            print(f"  ! mobileswap {split}: thiếu zip — bỏ")
            continue
        if split == "cdf":
            ms_cdf = frames_dir
        else:
            ms_ff = frames_dir

    if ms_cdf and os.path.isdir(ms_cdf):
        for vid in sorted(os.listdir(ms_cdf)):
            fdir = os.path.join(ms_cdf, vid)
            if not os.path.isdir(fdir):
                continue
            ident = celeb_identity(vid)
            if ident is not None and ident in cdc_real:
                dom = "cdc"
            else:
                ident, dom = f"mobileswap:{vid}", "oth"
            fr = probe_frame(fdir, real_sel.get(("cdc", ident)) if dom == "cdc" else None)
            if not fr:
                continue
            add("mobileswap", vid, os.path.join(fdir, fr), ident, dom)
    if ms_ff and os.path.isdir(ms_ff):
        for vid in sorted(os.listdir(ms_ff)):
            if not re.fullmatch(r"\d+_\d+", vid):
                continue
            ident = vid.split("_")[0]
            if ident not in ffc_real:
                continue
            fdir = os.path.join(ms_ff, vid)
            if not os.path.isdir(fdir):
                continue
            fr = probe_frame(fdir, real_sel.get(("ffc", ident)))
            if not fr:
                continue
            add("mobileswap", vid, os.path.join(fdir, fr), ident, "ffc")

    n_real = sum(1 for it in items if it["label"] == 0)
    n_fake = len(items) - n_real
    n_methods = len(set(it["method"] for it in items if it["label"] == 1))
    print(f"\nTổng: {len(items):,} ảnh (real={n_real:,} fake={n_fake:,}) | "
          f"{n_methods} method | {counts}")

    grp = {}
    for it in items:
        grp.setdefault(it["identity"], {"real": 0, "fake": 0})
        grp[it["identity"]]["real" if it["label"] == 0 else "fake"] += 1
    n_pair = sum(1 for v in grp.values() if v["real"] and v["fake"])
    n_fake_pair = sum(v["fake"] for v in grp.values() if v["real"] and v["fake"])
    n_fake_unp = sum(v["fake"] for v in grp.values() if not v["real"])
    print(f"Identity keys: {len(grp):,} | paired: {n_pair:,} ({n_fake_pair:,} fake) | "
          f"fake không ghép real: {n_fake_unp:,}")

    if args.dry_run or args.scan_only:
        print("\n[dry-run] Không copy.")
        return

    # ---------- copy + verify + manifest ----------
    os.makedirs(OUT, exist_ok=True)
    bad_copy, bad_decode = 0, []

    def copy_one(it):
        """Copy 1 file + verify decode. Trả (status, it, err). Thread-safe."""
        dst = os.path.join(OUT, it["path"])
        try:
            shutil.copy2(it["src"], dst)
        except Exception as e:
            return ("copy", it, str(e))
        if not args.no_verify:
            try:
                Image.open(dst).load()
            except Exception:
                try:
                    os.remove(dst)
                except OSError:
                    pass
                return ("decode", it, None)
        return ("ok", it, None)

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(copy_one, it): it for it in items}
        for fut in as_completed(futs):
            status, it, err = fut.result()
            done += 1
            if status == "copy":
                bad_copy += 1
                it["_dead"] = True
                print(f"  ! copy lỗi {it['src']}: {err}")
            elif status == "decode":
                bad_decode.append(it["path"])
                it["_dead"] = True
            if done % 1000 == 0:
                print(f"  ...copy {done:,}/{len(items):,}")
    items = [it for it in items if not it.get("_dead")]
    print(f"Copy OK: {len(items) + bad_copy + len(bad_decode):,} | lỗi copy={bad_copy} | "
          f"decode hỏng={len(bad_decode)}")

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "video", "path", "identity",
                                          "domain", "label"])
        w.writeheader()
        for it in items:
            w.writerow({k: it[k] for k in w.fieldnames})
    n_real = sum(1 for it in items if it["label"] == 0)
    print(f"Saved: {MANIFEST} ({len(items):,} rows, real={n_real:,}) → {OUT}/")


if __name__ == "__main__":
    main()

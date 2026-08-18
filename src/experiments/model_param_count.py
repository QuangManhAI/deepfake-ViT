"""Đếm số tham số thật của các model video (CNN vs ViT) để chọn cặp matched-size.

Chạy:
    python3 experiments/model_param_count.py

Kết quả ghi trong experiments/model_param_count.log
"""
from torchvision.models.video import (
    swin3d_t, swin3d_s, swin3d_b, r2plus1d_18, r3d_18, mc3_18,
    mvit_v1_b, mvit_v2_s, s3d,
)
from pytorchvideo.models.hub import (
    x3d_xs, x3d_s, x3d_m, x3d_l, mvit_base_16x4, slow_r50,
    slowfast_r50, i3d_r50, csn_r101, r2plus1d_r50,
)


def count(name, model):
    n = sum(p.numel() for p in model.parameters())
    print(f"{name:22s} {n/1e6:8.2f}M")
    return n


def main():
    print("== pytorchvideo (Kinetics-400 pretrained) ==")
    count("X3D-XS", x3d_xs(pretrained=False))
    count("X3D-S", x3d_s(pretrained=False))
    count("X3D-M", x3d_m(pretrained=False))
    count("X3D-L", x3d_l(pretrained=False))
    count("MViT-B 16x4", mvit_base_16x4(pretrained=False))
    count("Slow-R50", slow_r50(pretrained=False))
    count("SlowFast-R50", slowfast_r50(pretrained=False))
    count("I3D-R50", i3d_r50(pretrained=False))
    count("CSN-R101", csn_r101(pretrained=False))
    count("R(2+1)D-R50", r2plus1d_r50(pretrained=False))

    print("== torchvision ==")
    count("Swin3D-T", swin3d_t(weights=None))
    count("Swin3D-S", swin3d_s(weights=None))
    count("Swin3D-B", swin3d_b(weights=None))
    count("R(2+1)D-18", r2plus1d_18(weights=None))
    count("R3D-18", r3d_18(weights=None))
    count("MC3-18", mc3_18(weights=None))
    count("S3D", s3d(weights=None))
    count("MViTv1-B (tv)", mvit_v1_b(weights=None))
    count("MViTv2-S (tv)", mvit_v2_s(weights=None))


if __name__ == "__main__":
    main()

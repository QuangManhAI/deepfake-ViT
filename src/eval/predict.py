"""Dự đoán nhãn cho 1 ảnh bằng DINOv3 backbone + linear probe.

Linear probe được train lại trên đặc trưng đã cache (features.npz) — cùng split 80/20
như script đánh giá, nên kết quả dự đoán là hợp lệ (ảnh không nằm trong tập train).
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.models.dinov3_vit import load_dinov3  # noqa: E402

IMG_SIZE = 256
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train_probe(features_path: str, seed: int = 42, test_size: float = 0.2):
    """Train linear probe trên train split của features đã cache."""
    data = np.load(features_path, allow_pickle=True)
    X, y = data["features"], data["labels"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    clf = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
    )
    clf.fit(X_train, y_train)
    return clf


def extract_feature(model, img_path: str, device: str) -> np.ndarray:
    img = TRANSFORM(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        feat = model(img)
    return feat.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description="Dự đoán nhãn 1 ảnh")
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh cần dự đoán")
    parser.add_argument("--model", default="experiments/checkpoints/weights/model.safetensors")
    parser.add_argument("--features", default="experiments/results/features.npz")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Train linear probe (trên train split — ảnh cần test nằm ngoài split này)
    clf = train_probe(args.features)
    model = load_dinov3(args.model, img_size=IMG_SIZE).to(device)

    feat = extract_feature(model, args.image, device)
    prob = clf.predict_proba(feat.reshape(1, -1))[0]
    p_fake = float(prob[1])  # class 1 = FAKE
    label = "FAKE" if p_fake >= 0.5 else "REAL"

    print(f"\nẢnh: {args.image}")
    print(f"  → Kết luận: **{label}** (xác suất fake = {p_fake:.1%})")
    print(f"    real: {prob[0]:.1%} | fake: {prob[1]:.1%}")


if __name__ == "__main__":
    main()

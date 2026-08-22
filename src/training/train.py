"""Fine-tune DINOv3 backbone + classification head trên dữ liệu cân bằng.

Thiết kế:
  - Backbone: DINOv3 ViT-S/16 (load từ model.safetensors) — được fine-tune (không đông cứng)
  - Head: Linear(384, 2) trên CLS token
  - Optimizer: AdamW 2 nhóm LR (backbone thấp 1e-5, head cao 1e-3)
  - Scheduler: CosineAnnealingLR
  - Checkpoint: full-state (model+optimizer+scheduler+RNG+history) theo
    LOGGING_CHECKPOINT_RULES.md — lưu `<run>_best.pt` + `<run>_last.pt` mỗi epoch,
    history JSONL, config JSON; `--resume`/`--force-resume` để tiếp tục chính xác.
  - Giữ một bản best-state legacy `dinov3_finetuned.pt` cho các eval script cũ.

Cách chạy:
  .venv/bin/python src/training/train.py --train-csv data/splits/train_insight.csv \
      --val-csv data/splits/val_insight.csv --test-csv data/splits/test_insight.csv
  .venv/bin/python src/training/train.py ... --resume            # tiếp tục từ _last.pt
  .venv/bin/python src/training/train.py ... --force-resume      # từ _best.pt
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.models.dinov3_vit import load_dinov3  # noqa: E402
from src.utils.run_logger import (  # noqa: E402
    RunLogger,
    append_history_jsonl,
    find_latest_run_dir,
    load_full_checkpoint,
    make_run_dir,
    save_full_checkpoint,
    write_config_json,
)
from src.utils.seeding import set_seed  # noqa: E402

IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ImageDataset(Dataset):
    def __init__(self, csv_path, transform=None, max_samples=None):
        self.rows = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    self.rows.append((row[0], int(row[1])))
        if max_samples and len(self.rows) > max_samples:
            real = [r for r in self.rows if r[1] == 0]
            fake = [r for r in self.rows if r[1] == 1]
            half = max_samples // 2
            self.rows = real[:half] + fake[:half]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class DinoViTClassifier(nn.Module):
    """Backbone DINOv3 + Linear head trên CLS token."""

    def __init__(self, backbone: nn.Module, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    total_loss, n_batches = 0.0, 0
    for x, y in loader:
        x, y_dev = x.to(device), y.to(device)
        logits = model(x)
        if criterion:
            loss = criterion(logits, y_dev)
            total_loss += loss.item()
            n_batches += 1
        probs = torch.softmax(logits, dim=1)
        all_y.extend(y.tolist())
        all_pred.extend(logits.argmax(1).tolist())
        all_prob.extend(probs[:, 1].tolist())
    y = np.array(all_y)
    pred = np.array(all_pred)
    prob = np.array(all_prob)
    val_loss = (total_loss / max(1, n_batches)) if criterion else None
    return {
        "loss": val_loss,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv3")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--model", default="experiments/checkpoints/weights/model.safetensors")
    parser.add_argument("--output-dir", default="experiments/results/checkpoints")
    parser.add_argument("--run-name", default="dinov3_finetuned")
    parser.add_argument("--report", default="experiments/results/finetune_report.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze the backbone entirely; train only the classification head")
    parser.add_argument("--max-train", type=int, default=0,
                        help="Cap training samples (0 = use all; balanced if possible)")
    parser.add_argument("--class-weight", action="store_true",
                        help="Use inverse-frequency class weights in the loss (helps imbalance)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early-stop after this many epochs without val-acc improvement (0 = off)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="mixed precision (bfloat16)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the run's _last.pt (exact state)")
    parser.add_argument("--force-resume", action="store_true",
                        help="resume from _best.pt with a fresh early-stopping budget")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    device_type = "cuda" if device == "cuda" else "cpu"
    print(f"Device: {device}", flush=True)

    set_seed(args.seed)

    # ---------- Run dir (LOGGING_CHECKPOINT_RULES layout) ----------
    resuming = args.resume or args.force_resume
    prev_run_dir = find_latest_run_dir(args.output_dir, args.run_name) if resuming else None
    if prev_run_dir:
        run_dir = prev_run_dir
        print(f"Resuming into existing run dir: {run_dir}", flush=True)
    else:
        run_dir = make_run_dir(args.output_dir, args.run_name)
    logger = RunLogger(run_dir, args.run_name)
    best_path = os.path.join(run_dir, "checkpoints", f"{args.run_name}_best.pt")
    last_path = os.path.join(run_dir, "checkpoints", f"{args.run_name}_last.pt")
    metrics_path = os.path.join(run_dir, "metrics", f"{args.run_name}_history.jsonl")
    config_path = os.path.join(run_dir, "metrics", f"{args.run_name}_config.json")
    legacy_best_path = os.path.join(args.output_dir, "dinov3_finetuned.pt")

    # ---------- Data ----------
    train_ds = ImageDataset(args.train_csv, TRAIN_TF, max_samples=args.max_train or None)
    val_ds = ImageDataset(args.val_csv, EVAL_TF)
    test_ds = ImageDataset(args.test_csv, EVAL_TF)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ---------- Model ----------
    backbone = load_dinov3(args.model, img_size=IMG_SIZE)
    model = DinoViTClassifier(backbone).to(device)
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone FROZEN — training only the classification head")
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {n_params:,} params ({n_params/1e6:.1f}M) | "
                f"trainable: {n_trainable:,} ({n_trainable/1e6:.1f}M)")

    # ---------- Optimizer / Scheduler ----------
    opt_groups = []
    if not args.freeze_backbone:
        opt_groups.append({"params": model.backbone.parameters(), "lr": args.lr_backbone})
    opt_groups.append({"params": model.head.parameters(), "lr": args.lr_head})
    optimizer = torch.optim.AdamW(opt_groups, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    if args.class_weight:
        import collections
        counts = collections.Counter(r[1] for r in train_ds.rows)
        n_total = sum(counts.values())
        weights = torch.tensor(
            [n_total / max(1, counts[c]) / 2 for c in range(2)],
            dtype=torch.float32, device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
        logger.info(f"Class weights: real={weights[0]:.4f} fake={weights[1]:.4f} "
                    f"(counts={dict(counts)})")

    config = {
        "run_name": args.run_name,
        "description": "DINOv3 ViT-S/16 fine-tune, real/fake binary, DF40 splits.",
        "outputs": ["checkpoints/<run>_best.pt", "checkpoints/<run>_last.pt",
                    "metrics/<run>_history.jsonl", "metrics/<run>_config.json"],
        "args": vars(args),
        "seed": args.seed,
        "model_params_M": round(n_params / 1e6, 2),
    }

    # ---------- Resume ----------
    history = []
    best_val_acc = 0.0
    best_metrics = {}
    start_epoch = 0
    global_step = 0
    if args.resume or args.force_resume:
        ckpt_path = best_path if args.force_resume else last_path
        if not os.path.exists(ckpt_path):
            logger.warn(f"Resume requested but {ckpt_path} missing — starting fresh")
        else:
            ckpt = load_full_checkpoint(ckpt_path, model, optimizer, scheduler, device)
            history = ckpt.get("history", [])
            best_metrics = ckpt.get("best_metrics", {})
            best_val_acc = float(best_metrics.get("accuracy", 0.0) or 0.0)
            global_step = int(ckpt.get("global_step", 0))
            if args.force_resume:
                start_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", 0)) or 0)
            else:
                start_epoch = int(ckpt.get("epoch", 0) or 0)
                if ckpt.get("early_stop_triggered"):
                    logger.error("Run previously early-stopped — use --force-resume to retrain from best.")
                    return
            logger.info(f"Resumed from {ckpt_path} at epoch {start_epoch + 1} (global_step={global_step})")

    write_config_json(config_path, config)
    logger.info(f"Run dir: {run_dir}")

    # ---------- Train ----------
    os.makedirs(args.output_dir, exist_ok=True)
    patience_counter = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        total_loss, n_batch = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    unit="batch", ncols=100, leave=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=args.amp and device != "mps"):
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
            global_step += 1
            if n_batch % 20 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, criterion=criterion)
        train_loss = total_loss / n_batch
        epoch_metrics = {"epoch": epoch + 1, "train_loss": train_loss, **val_metrics}
        history.append(epoch_metrics)
        dt = time.time() - t0
        logger.epoch_summary(epoch + 1, args.epochs,
                             {"loss": train_loss, "val_loss": val_metrics["loss"],
                              "val_acc": val_metrics["accuracy"], "val_f1": val_metrics["f1"]},
                             dt)

        # every-epoch full-state checkpoint (resume source)
        save_full_checkpoint(last_path, model, optimizer, scheduler, epoch + 1,
                             global_step, {**best_metrics, "early_stop_triggered": False},
                             history, config, args.seed)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = {"accuracy": best_val_acc, **val_metrics, "best_epoch": epoch + 1}
            save_full_checkpoint(best_path, model, optimizer, scheduler, epoch + 1,
                                 global_step, best_metrics, history, config, args.seed)
            # legacy best-state copy for the old eval scripts (state_dict only)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1,
                        "val_metrics": val_metrics}, legacy_best_path)
            logger.info(f"Saved best checkpoint (val_acc={best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if args.patience and patience_counter >= args.patience:
                logger.info(f"Early stopping: no val-acc improvement for {patience_counter} "
                            f"epochs (patience={args.patience}). Stopping at epoch {epoch + 1}.")
                best_metrics["early_stop_triggered"] = True
                save_full_checkpoint(last_path, model, optimizer, scheduler, epoch + 1,
                                     global_step, best_metrics, history, config, args.seed)
                break

        append_history_jsonl(metrics_path, epoch_metrics)

    # ---------- Test ----------
    logger.info("Load checkpoint tốt nhất và đánh giá TEST...")
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, criterion=criterion)

    print("\n================= KẾT QUẢ TEST (sau fine-tune) =================")
    if test_metrics["loss"] is not None:
        print(f"  loss     : {test_metrics['loss']:.4f}")
    print(f"  accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  precision: {test_metrics['precision']:.4f}")
    print(f"  recall   : {test_metrics['recall']:.4f}")
    print(f"  f1       : {test_metrics['f1']:.4f}")
    print(f"  roc_auc  : {test_metrics['roc_auc']:.4f}")
    cm = test_metrics["confusion_matrix"]
    print(f"  CM [TN FP; FN TP]: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    print("===================================================================")

    report = {"best_val_acc": best_val_acc, "test": test_metrics,
              "history": history, "args": vars(args), "run_dir": run_dir}
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Đã lưu: {best_path} + {last_path} + {args.report}")


if __name__ == "__main__":
    main()

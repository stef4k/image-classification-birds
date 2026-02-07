import argparse
import json
import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from birds_ml.arcface_margin import ArcMarginProduct
from birds_ml.config import Config
from birds_ml.data import load_trainval_from_folders, load_val_with_given_mapping
from birds_ml.embedder import build_backbone
from birds_ml.features import SampleDataset
from birds_ml.utils import ensure_dir, set_seed


class SquarePad:
    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        scale = self.target_size / max_wh
        new_w, new_h = int(w * scale), int(h * scale)
        img = TF.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)

        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        pad_left = delta_w // 2
        pad_right = delta_w - pad_left
        pad_top = delta_h // 2
        pad_bottom = delta_h - pad_top
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=128, padding_mode="constant")


def build_transforms(img_size: int, mean, std):
    train_tfm = transforms.Compose([
        SquarePad(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tfm = transforms.Compose([
        SquarePad(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tfm, val_tfm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="eva02_large_448")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--use_arcface", action="store_true")
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Use only train_images for CV. By default train_images + val_images are combined.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional name for this CV run. If omitted, a deterministic name is generated.",
    )
    return parser.parse_args()


def make_head(input_dim: int, num_classes: int, use_arcface: bool, device: torch.device):
    if use_arcface:
        return ArcMarginProduct(input_dim, num_classes, s=30.0, m=0.50).to(device)

    head = nn.Linear(input_dim, num_classes).to(device)
    nn.init.constant_(head.bias, 0)
    nn.init.normal_(head.weight, std=0.01)
    return head


def eval_fold(backbone_model, head, val_dl, use_arcface: bool, amp_enabled: bool, device: torch.device):
    head.eval()

    all_true = []
    all_pred = []
    all_conf = []
    all_paths = []

    with torch.no_grad():
        for imgs, labels, paths in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast(enabled=amp_enabled):
                feats = backbone_model(imgs)

                if use_arcface:
                    norm_feats = F.normalize(feats)
                    norm_weights = F.normalize(head.weight)
                    logits = F.linear(norm_feats, norm_weights) * head.s
                else:
                    logits = head(feats)

            probs = torch.softmax(logits, dim=1)
            conf, predicted = torch.max(probs, dim=1)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(predicted.cpu().numpy().tolist())
            all_conf.extend(conf.cpu().numpy().tolist())
            all_paths.extend(list(paths))

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    y_conf = np.asarray(all_conf, dtype=np.float32)
    acc = float((y_true == y_pred).mean()) if y_true.size > 0 else 0.0
    return acc, y_true, y_pred, y_conf, all_paths


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    cfg = Config()
    cfg = replace(cfg, use_crops=args.use_crops)

    ensure_dir(cfg.outputs_dir)
    set_seed(cfg.seed)

    device_name = cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    device = torch.device(device_name)
    amp_enabled = device.type == "cuda"

    backbone_model, model_config = build_backbone(args.backbone)
    backbone_model.to(device)
    backbone_model.eval()
    for p in backbone_model.parameters():
        p.requires_grad = False

    train_tfm, val_tfm = build_transforms(args.img_size, model_config["mean"], model_config["std"])

    train_samples, class_to_idx = load_trainval_from_folders(cfg.train_dir)

    all_samples = list(train_samples)
    if not args.train_only:
        val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)
        all_samples.extend(val_samples)

    if not all_samples:
        raise ValueError("No samples found for cross-validation.")

    labels = np.asarray([s.label for s in all_samples], dtype=np.int64)
    num_classes = len(class_to_idx)

    class_counts = np.bincount(labels, minlength=num_classes)
    min_class_count = int(class_counts[class_counts > 0].min())
    if min_class_count < 2:
        raise ValueError(
            "At least one class has fewer than 2 samples, so cross-validation is not possible."
        )

    folds = args.folds
    if folds > min_class_count:
        print(
            f"Requested {args.folds} folds, but smallest class has {min_class_count} samples. "
            f"Using {min_class_count} folds instead."
        )
        folds = min_class_count

    head_prefix = "arcface" if args.use_arcface else "linear"
    crop_suffix = "_cropped" if cfg.use_crops else ""
    run_name = args.run_name or f"cv_{head_prefix}_{args.backbone}{crop_suffix}_{args.img_size}_e{args.epochs}_f{folds}"
    run_dir = ensure_dir(cfg.outputs_dir / "cv_runs" / run_name)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.seed)

    fold_accs = []
    class_acc_by_fold = defaultdict(list)
    class_precision_by_fold = defaultdict(list)
    class_recall_by_fold = defaultdict(list)
    all_oof_rows = []

    print(
        f"Running {folds}-fold CV | Backbone: {args.backbone} | "
        f"Head: {'ArcFace' if args.use_arcface else 'Linear'} | "
        f"Samples: {len(all_samples)} | Classes: {num_classes}"
    )
    print(f"Saving CV artifacts to: {run_dir}")

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        fold_train = [all_samples[i] for i in tr_idx]
        fold_val = [all_samples[i] for i in va_idx]

        train_dl = DataLoader(
            SampleDataset(fold_train, train_tfm),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_dl = DataLoader(
            SampleDataset(fold_val, val_tfm),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        head = make_head(backbone_model.num_features, num_classes, args.use_arcface, device)
        optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler = GradScaler(enabled=amp_enabled)

        best_acc = -1.0
        best_y_true = None
        best_y_pred = None
        best_y_conf = None
        best_paths = None

        for epoch in range(args.epochs):
            head.train()
            running_loss = 0.0

            for imgs, yb, _ in train_dl:
                imgs, yb = imgs.to(device), yb.to(device)
                optimizer.zero_grad()

                with autocast(enabled=amp_enabled):
                    with torch.no_grad():
                        feats = backbone_model(imgs)

                    if args.use_arcface:
                        logits = head(feats, yb)
                    else:
                        logits = head(feats)

                    loss = criterion(logits, yb)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

            scheduler.step()

            val_acc, y_true_fold, y_pred_fold, y_conf_fold, y_paths_fold = eval_fold(
                backbone_model=backbone_model,
                head=head,
                val_dl=val_dl,
                use_arcface=args.use_arcface,
                amp_enabled=amp_enabled,
                device=device,
            )

            avg_loss = running_loss / max(1, len(train_dl))
            print(
                f"Fold {fold_idx}/{folds} | Epoch {epoch + 1}/{args.epochs} "
                f"| Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_acc >= best_acc:
                best_acc = val_acc
                best_y_true = y_true_fold
                best_y_pred = y_pred_fold
                best_y_conf = y_conf_fold
                best_paths = y_paths_fold

        fold_accs.append(best_acc)

        precision, recall, _, _ = precision_recall_fscore_support(
            best_y_true,
            best_y_pred,
            labels=list(range(num_classes)),
            zero_division=0,
        )

        for cls_idx in range(num_classes):
            cls_mask = best_y_true == cls_idx
            cls_acc = float((best_y_pred[cls_mask] == cls_idx).mean()) if np.any(cls_mask) else math.nan
            class_acc_by_fold[cls_idx].append(cls_acc)
            class_precision_by_fold[cls_idx].append(float(precision[cls_idx]))
            class_recall_by_fold[cls_idx].append(float(recall[cls_idx]))

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        fold_rows = []
        for i, src_path in enumerate(best_paths):
            gt_idx = int(best_y_true[i])
            pred_idx = int(best_y_pred[i])
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "source_path": str(src_path),
                    "ground_truth_idx": gt_idx,
                    "predicted_idx": pred_idx,
                    "ground_truth": idx_to_class[gt_idx],
                    "predicted": idx_to_class[pred_idx],
                    "confidence": float(best_y_conf[i]),
                    "correct": bool(gt_idx == pred_idx),
                }
            )

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(run_dir / f"fold_{fold_idx}_predictions.csv", index=False)
        all_oof_rows.extend(fold_rows)

        print(f"Fold {fold_idx} best accuracy: {best_acc:.4f}")

    print("\n=== Cross-validation summary ===")
    print(
        f"Fold accuracy mean: {float(np.mean(fold_accs)):.4f} | "
        f"std: {float(np.std(fold_accs)):.4f}"
    )
    print(
        "Per-species avg metrics are computed as the mean of fold metrics. "
        "(Species accuracy = correctly predicted images / images of that species in the fold.)"
    )

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    rows = []
    for cls_idx in range(num_classes):
        rows.append(
            {
                "species": idx_to_class[cls_idx],
                "avg_accuracy": float(np.nanmean(class_acc_by_fold[cls_idx])),
                "avg_precision": float(np.nanmean(class_precision_by_fold[cls_idx])),
                "avg_recall": float(np.nanmean(class_recall_by_fold[cls_idx])),
                "n_images_total": int(class_counts[cls_idx]),
            }
        )

    report_df = pd.DataFrame(rows).sort_values("avg_accuracy", ascending=True)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(report_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    oof_df = pd.DataFrame(all_oof_rows)
    oof_df.to_csv(run_dir / "oof_predictions.csv", index=False)
    report_df.to_csv(run_dir / "per_species_metrics.csv", index=False)

    run_meta = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "backbone": args.backbone,
        "img_size": args.img_size,
        "epochs": args.epochs,
        "folds": folds,
        "use_crops": bool(args.use_crops),
        "use_arcface": bool(args.use_arcface),
        "train_only": bool(args.train_only),
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
        "fold_acc_mean": float(np.mean(fold_accs)),
        "fold_acc_std": float(np.std(fold_accs)),
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    print(f"\nSaved OOF predictions: {run_dir / 'oof_predictions.csv'}")
    print(f"Saved per-species metrics: {run_dir / 'per_species_metrics.csv'}")
    print(f"Saved run metadata: {run_dir / 'meta.json'}")


if __name__ == "__main__":
    main()

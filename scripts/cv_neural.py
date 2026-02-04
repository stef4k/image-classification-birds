import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from birds_ml.config import Config
from birds_ml.data import load_trainval_from_folders, load_val_with_given_mapping, Sample
from birds_ml.embedder import build_backbone
from birds_ml.features import SampleDataset
from birds_ml.utils import ensure_dir, set_seed


class SquarePad:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # resize so longest edge = target_size
        w, h = img.size
        max_wh = max(w, h)
        scale = self.target_size / max_wh
        new_w, new_h = int(w * scale), int(h * scale)
        img = TF.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)

        # pad to make it square
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        pad_left = delta_w // 2
        pad_right = delta_w - pad_left
        pad_top = delta_h // 2
        pad_bottom = delta_h - pad_top

        # fill with gray (128)
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=128, padding_mode="constant")

def _iter_blocks(backbone_model):
    if hasattr(backbone_model, "blocks"):
        return list(backbone_model.blocks)
    if hasattr(backbone_model, "stages"):
        blocks = []
        for stage in backbone_model.stages:
            if hasattr(stage, "blocks"):
                blocks.extend(list(stage.blocks))
            else:
                blocks.append(stage)
        return blocks
    if hasattr(backbone_model, "layers"):
        return list(backbone_model.layers)
    if hasattr(backbone_model, "features"):
        return list(backbone_model.features)
    return list(backbone_model.children())

def _unfreeze_last_blocks(backbone_model, n_blocks: int):
    if n_blocks <= 0:
        return
    blocks = _iter_blocks(backbone_model)
    if not blocks:
        return
    for block in blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

def _build_class_weights(class_to_idx, crow_a, crow_b, crow_weight, device):
    weights = torch.ones(len(class_to_idx), device=device)
    for cname in (crow_a, crow_b):
        if cname in class_to_idx:
            weights[class_to_idx[cname]] = crow_weight
    return weights

def _eval(backbone_model, head, val_dl, device):
    backbone_model.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, _ in val_dl:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                feats = backbone_model(imgs)
                preds = head(feats)
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0


def _merge_train_val(cfg: Config) -> (List[Sample], Dict[str, int]):
    train_samples, class_to_idx = load_trainval_from_folders(cfg.train_dir)
    val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)
    samples = train_samples + val_samples
    return samples, class_to_idx


def main():
    cfg = Config()
    ensure_dir(cfg.outputs_dir)

    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_so150m2_384")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--finetune_epochs", type=int, default=8)
    ap.add_argument("--finetune_blocks", type=int, default=2)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--backbone_lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--save_folds", action="store_true")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--use_crops", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crow_a", type=str, default="American_Crow")
    ap.add_argument("--crow_b", type=str, default="Fish_Crow")
    ap.add_argument("--crow_weight", type=float, default=2.0)
    args = ap.parse_args()

    cfg = replace(cfg, use_crops=args.use_crops)
    set_seed(args.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    num_workers = cfg.num_workers if args.num_workers is None else args.num_workers

    crop_suffix = "_cropped" if args.use_crops else ""
    model_name = f"timm_finetune_{args.backbone}{crop_suffix}_{args.img_size}"

    # data
    samples, class_to_idx = _merge_train_val(cfg)
    labels = np.array([int(s.label) for s in samples], dtype=np.int64)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # model
    backbone_model, model_config = build_backbone(args.backbone)
    backbone_model.to(device)
    backbone_model.eval()
    for p in backbone_model.parameters():
        p.requires_grad = False

    # transforms (match train_neural.py)
    train_tfm = transforms.Compose([
        SquarePad(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_config["mean"], std=model_config["std"]),
    ])
    val_tfm = transforms.Compose([
        SquarePad(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_config["mean"], std=model_config["std"]),
    ])

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_metrics = []
    per_class_acc = {name: [] for name in class_to_idx.keys()}
    crow_folds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        train_ds = SampleDataset([samples[i] for i in train_idx], train_tfm)
        val_ds = SampleDataset([samples[i] for i in val_idx], val_tfm)

        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        head = nn.Linear(backbone_model.num_features, len(class_to_idx)).to(device)
        nn.init.constant_(head.bias, 0)
        nn.init.normal_(head.weight, std=0.01)

        class_weights = _build_class_weights(
            class_to_idx, args.crow_a, args.crow_b, args.crow_weight, device
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        best_acc = 0.0
        best_state = None

        # Phase A: head only
        optimizer = optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        patience_left = args.patience

        for epoch in range(args.epochs):
            head.train()
            for imgs, lbls, _ in train_dl:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    with torch.no_grad():
                        feats = backbone_model(imgs)
                    preds = head(feats)
                    loss = criterion(preds, lbls)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            acc = _eval(backbone_model, head, val_dl, device)
            if acc >= best_acc:
                best_acc = acc
                best_state = {
                    "backbone": {k: v.detach().cpu() for k, v in backbone_model.state_dict().items()},
                    "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                }
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        # Phase B: unfreeze last blocks
        if args.finetune_epochs > 0 and args.finetune_blocks > 0:
            _unfreeze_last_blocks(backbone_model, args.finetune_blocks)
            backbone_model.train()
            head.train()

            optimizer = optim.AdamW(
                [
                    {"params": head.parameters(), "lr": args.head_lr},
                    {
                        "params": [p for p in backbone_model.parameters() if p.requires_grad],
                        "lr": args.backbone_lr,
                    },
                ],
                weight_decay=args.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
            patience_left = args.patience

            for epoch in range(args.finetune_epochs):
                for imgs, lbls, _ in train_dl:
                    imgs = imgs.to(device, non_blocking=True)
                    lbls = lbls.to(device, non_blocking=True)
                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        feats = backbone_model(imgs)
                        preds = head(feats)
                        loss = criterion(preds, lbls)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()

                acc = _eval(backbone_model, head, val_dl, device)
                if acc >= best_acc:
                    best_acc = acc
                    best_state = {
                        "backbone": {k: v.detach().cpu() for k, v in backbone_model.state_dict().items()},
                        "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    }
                    patience_left = args.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        if best_state is not None:
            backbone_model.load_state_dict(best_state["backbone"])
            head.load_state_dict(best_state["head"])

        # validation
        head.eval()
        y_true, y_pred, y_conf = [], [], []
        class_correct = {name: 0 for name in class_to_idx.keys()}
        class_total = {name: 0 for name in class_to_idx.keys()}

        with torch.no_grad():
            for imgs, lbls, _ in val_dl:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    feats = backbone_model(imgs)
                    logits = head(feats)
                probs = torch.softmax(logits, dim=1)
                conf, preds = torch.max(probs, dim=1)

                y_true.extend(lbls.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_conf.extend(conf.cpu().tolist())

                for t, p in zip(lbls.cpu().tolist(), preds.cpu().tolist()):
                    cname = idx_to_class[int(t)]
                    class_total[cname] += 1
                    if t == p:
                        class_correct[cname] += 1

        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        fold_metrics.append({"fold": fold, "accuracy": acc, "macro_f1": macro_f1})

        for cname in class_to_idx.keys():
            if class_total[cname] > 0:
                per_class_acc[cname].append(class_correct[cname] / class_total[cname])

        # Crow-pair confusion + confidence distributions
        crow_a = args.crow_a
        crow_b = args.crow_b
        if crow_a in class_to_idx and crow_b in class_to_idx:
            a_idx = class_to_idx[crow_a]
            b_idx = class_to_idx[crow_b]

            y_true_np = np.array(y_true)
            y_pred_np = np.array(y_pred)
            y_conf_np = np.array(y_conf, dtype=np.float32)

            a_mask = y_true_np == a_idx
            b_mask = y_true_np == b_idx

            a_to_b = float(np.mean(y_pred_np[a_mask] == b_idx)) if np.any(a_mask) else 0.0
            b_to_a = float(np.mean(y_pred_np[b_mask] == a_idx)) if np.any(b_mask) else 0.0

            a_mis_conf = y_conf_np[a_mask & (y_pred_np == b_idx)].tolist()
            b_mis_conf = y_conf_np[b_mask & (y_pred_np == a_idx)].tolist()

            crow_folds.append({
                "fold": fold,
                "crow_a": crow_a,
                "crow_b": crow_b,
                "crow_a_to_b_rate": a_to_b,
                "crow_b_to_a_rate": b_to_a,
                "crow_a_to_b_conf": a_mis_conf,
                "crow_b_to_a_conf": b_mis_conf,
            })
        else:
            crow_folds.append({
                "fold": fold,
                "crow_a": crow_a,
                "crow_b": crow_b,
                "crow_a_to_b_rate": None,
                "crow_b_to_a_rate": None,
                "crow_a_to_b_conf": [],
                "crow_b_to_a_conf": [],
            })

        if args.save_folds:
            ckpt_path = cfg.outputs_dir / f"cv_fold{fold}_{model_name}.pth"
            torch.save(
                {
                    "backbone": backbone_model.state_dict(),
                    "head": head.state_dict(),
                    "config": model_config,
                },
                ckpt_path,
            )

        print(f"Fold {fold}/{args.folds} | Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f}")

    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["macro_f1"] for m in fold_metrics]

    summary = {
        "backbone": args.backbone,
        "img_size": args.img_size,
        "use_crops": bool(args.use_crops),
        "folds": args.folds,
        "epochs": args.epochs,
        "finetune_epochs": args.finetune_epochs,
        "finetune_blocks": args.finetune_blocks,
        "head_lr": args.head_lr,
        "backbone_lr": args.backbone_lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "folds_metrics": fold_metrics,
        "crow_pair": {
            "crow_a": args.crow_a,
            "crow_b": args.crow_b,
            "folds": crow_folds,
        },
    }

    out_json = cfg.outputs_dir / f"cv_neural_{args.backbone}{crop_suffix}_{args.img_size}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    per_class_rows = []
    for cname, vals in per_class_acc.items():
        if vals:
            per_class_rows.append({
                "class": cname,
                "mean_accuracy": float(np.mean(vals)),
                "std_accuracy": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            })

    out_csv = cfg.outputs_dir / f"cv_neural_{args.backbone}{crop_suffix}_{args.img_size}_per_class.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        f.write("class,mean_accuracy,std_accuracy\n")
        for r in per_class_rows:
            f.write(f"{r['class']},{r['mean_accuracy']:.6f},{r['std_accuracy']:.6f}\n")

    # Crow-pair summary CSV (rates + basic confidence stats per fold + aggregated)
    crow_csv = cfg.outputs_dir / f"cv_neural_{args.backbone}{crop_suffix}_{args.img_size}_crow_pair.csv"
    with crow_csv.open("w", newline="", encoding="utf-8") as f:
        f.write(
            "fold,crow_a,crow_b,crow_a_to_b_rate,crow_b_to_a_rate,"
            "crow_a_to_b_conf_mean,crow_a_to_b_conf_std,crow_a_to_b_conf_n,"
            "crow_b_to_a_conf_mean,crow_b_to_a_conf_std,crow_b_to_a_conf_n\n"
        )

        def _stats(xs):
            if not xs:
                return 0.0, 0.0, 0
            arr = np.array(xs, dtype=np.float32)
            return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, int(len(arr))

        # per-fold rows
        for row in crow_folds:
            a_mean, a_std, a_n = _stats(row["crow_a_to_b_conf"])
            b_mean, b_std, b_n = _stats(row["crow_b_to_a_conf"])
            f.write(
                f"{row['fold']},{row['crow_a']},{row['crow_b']},"
                f"{row['crow_a_to_b_rate']},{row['crow_b_to_a_rate']},"
                f"{a_mean:.6f},{a_std:.6f},{a_n},"
                f"{b_mean:.6f},{b_std:.6f},{b_n}\n"
            )

        # aggregated over folds
        all_a_conf = [c for row in crow_folds for c in row["crow_a_to_b_conf"]]
        all_b_conf = [c for row in crow_folds for c in row["crow_b_to_a_conf"]]
        a_mean, a_std, a_n = _stats(all_a_conf)
        b_mean, b_std, b_n = _stats(all_b_conf)
        a_rates = [row["crow_a_to_b_rate"] for row in crow_folds if row["crow_a_to_b_rate"] is not None]
        b_rates = [row["crow_b_to_a_rate"] for row in crow_folds if row["crow_b_to_a_rate"] is not None]
        a_rate = float(np.mean(a_rates)) if a_rates else None
        b_rate = float(np.mean(b_rates)) if b_rates else None
        f.write(
            f"all,{args.crow_a},{args.crow_b},{a_rate},{b_rate},"
            f"{a_mean:.6f},{a_std:.6f},{a_n},"
            f"{b_mean:.6f},{b_std:.6f},{b_n}\n"
        )

    print("CV done.")
    print(f"Summary: {out_json}")
    print(f"Per-class: {out_csv}")
    print(f"Crow pair: {crow_csv}")


if __name__ == "__main__":
    main()

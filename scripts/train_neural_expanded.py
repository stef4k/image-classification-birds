import argparse
import json
import os
import time
from dataclasses import replace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb

from birds_ml.config import Config
from birds_ml.data import load_trainval_from_folders, load_val_with_given_mapping
from birds_ml.features import SampleDataset
from birds_ml.embedder import build_backbone
from birds_ml.head import CustomHead
from birds_ml.utils import set_seed, ensure_dir


def build_scheduler(optimizer, name: str, epochs: int, step_size: int, gamma: float, min_lr: float):
    name = (name or "cosine").lower()
    if name == "none":
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"Unknown scheduler: {name}")


def main():
    cfg = Config()
    ensure_dir(cfg.outputs_dir)

    parser = argparse.ArgumentParser()

    # Core
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Override cfg.seed for reproducibility")
    parser.add_argument("--device", default=None, help="Override cfg.device, e.g. cuda or cpu")

    # Data loading
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=None, help="Override cfg.num_workers")

    # Optimization
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--betas", type=str, default="0.9,0.999", help="AdamW betas as 'b1,b2'")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW eps")

    # Scheduler
    parser.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--step_size", type=int, default=10, help="StepLR step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma")
    parser.add_argument("--min_lr", type=float, default=0.0, help="Cosine eta_min")

    # Head / regularization
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=None, help="Override default dropout rule")

    # Loss
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # Augmentation + params
    parser.add_argument("--augment", action="store_true", help="Enable augmentation pipeline")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--resize_train", type=int, default=256)
    parser.add_argument("--crop_scale_min", type=float, default=0.7)
    parser.add_argument("--crop_scale_max", type=float, default=1.0)
    parser.add_argument("--rotation_deg", type=float, default=30.0)
    parser.add_argument("--hflip_p", type=float, default=0.5)

    # W&B toggles (optional)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default=None, help="Override WANDB_PROJECT")
    parser.add_argument("--wandb_entity", default=None, help="Override WANDB_ENTITY")
    parser.add_argument("--wandb_group", default=None, help="Override WANDB_GROUP")
    parser.add_argument("--wandb_name", default=None, help="Override WANDB_NAME")
    parser.add_argument("--wandb_mode", default=None, choices=["online", "offline", "disabled"],
                        help="Override WANDB_MODE (online/offline/disabled)")

    args = parser.parse_args()

    # Apply cfg overrides
    cfg = replace(cfg, use_crops=args.use_crops)
    if args.device is not None:
        cfg = replace(cfg, device=args.device)
    if args.num_workers is not None:
        cfg = replace(cfg, num_workers=args.num_workers)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # -----------------------
    # W&B init (if enabled)
    # -----------------------
    use_wandb = bool(args.wandb) and (args.wandb_mode != "disabled") and (os.environ.get("WANDB_MODE") != "disabled")
    run = None
    if use_wandb:
        wandb_mode = (
            args.wandb_mode
            or os.environ.get("WANDB_MODE", "online")
        )
        wandb_project = (
            args.wandb_project
            or os.environ.get("WANDB_PROJECT", "birds-neural")
        )
        wandb_entity = (
            args.wandb_entity
            or os.environ.get("WANDB_ENTITY", None)
        )
        wandb_group = (
            args.wandb_group
            or os.environ.get("WANDB_GROUP", None)
        )
        wandb_name = (
            args.wandb_name
            or os.environ.get("WANDB_NAME", None)
        )

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            name=wandb_name,
            config=vars(args),
            mode=wandb_mode,   # online | offline
            reinit=True,
        )

    # Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    standard_tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if args.augment:
        train_tfm = transforms.Compose([
            transforms.Resize((args.resize_train, args.resize_train)),
            transforms.RandomResizedCrop(
                args.img_size,
                scale=(args.crop_scale_min, args.crop_scale_max),
            ),
            transforms.RandomHorizontalFlip(p=args.hflip_p),
            transforms.RandomRotation(args.rotation_deg),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tfm = standard_tfm

    print(
        "Training Neural Head | "
        f"Backbone={args.backbone} | Crops={cfg.use_crops} | Augment={args.augment} | "
        f"Device={device} | Seed={cfg.seed}"
    )

    # Data
    print(f"Loading TRAIN data from: {cfg.train_dir}")
    train_samples, class_to_idx = load_trainval_from_folders(cfg.train_dir)

    print(f"Loading VAL data from: {cfg.val_dir}")
    val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)

    print(f"Dataset Size -> Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_ds = SampleDataset(train_samples, train_tfm)
    val_ds = SampleDataset(val_samples, standard_tfm)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # Model
    backbone_model, _ = build_backbone(args.backbone)
    backbone_model.to(device)
    backbone_model.eval()  # freeze

    input_dim = 2048 if args.backbone == "resnet50" else 1280

    # default dropout rule unless overridden
    if args.dropout is not None:
        dropout = float(args.dropout)
    else:
        dropout = 0.7 if args.augment else 0.5

    head = CustomHead(input_dim, args.hidden_dim, len(class_to_idx), dropout_prob=dropout).to(device)

    # Optimizer
    opt_name = args.optimizer.lower()
    if opt_name == "adamw":
        b1, b2 = (float(x) for x in args.betas.split(","))
        optimizer = optim.AdamW(
            head.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(b1, b2),
            eps=args.eps,
        )
    else:
        optimizer = optim.SGD(
            head.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

    scheduler = build_scheduler(
        optimizer=optimizer,
        name=args.scheduler,
        epochs=args.epochs,
        step_size=args.step_size,
        gamma=args.gamma,
        min_lr=args.min_lr,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Naming (keep it readable but unique)
    crop_suffix = "_cropped" if cfg.use_crops else ""
    aug_suffix = "_aug" if args.augment else ""
    model_name = (
        f"neural_{args.backbone}{crop_suffix}{aug_suffix}"
        f"_bs{args.batch_size}_lr{args.lr:g}_wd{args.weight_decay:g}"
        f"_{args.optimizer}_hd{args.hidden_dim}_do{dropout:g}"
        f"_ls{args.label_smoothing:g}_sch{args.scheduler}_seed{cfg.seed}"
    )

    # Push derived fields to W&B config once we know dataset sizes, etc.
    if run is not None:
        wandb.config.update(
            {
                "model_name": model_name,
                "seed_effective": cfg.seed,
                "device_effective": str(device),
                "num_workers_effective": cfg.num_workers,
                "dropout_effective": dropout,
                "n_classes": len(class_to_idx),
                "train_size": len(train_samples),
                "val_size": len(val_samples),
            },
            allow_val_change=True,
        )

    # Training
    best_acc = 0.0
    best_epoch = -1
    t0 = time.time()

    for epoch in range(args.epochs):
        head.train()
        train_loss = 0.0

        for imgs, labels, _ in train_dl:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.no_grad():
                feats = backbone_model(imgs)

            preds = head(feats)
            loss = criterion(preds, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        # Validation
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_dl:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                feats = backbone_model(imgs)
                preds = head(feats)
                predicted = preds.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        avg_loss = train_loss / max(len(train_dl), 1)
        cur_lr = float(optimizer.param_groups[0]["lr"])

        print(f"Epoch {epoch+1}/{args.epochs} | lr={cur_lr:.3e} | Loss={avg_loss:.4f} | Val Acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save(head.state_dict(), cfg.outputs_dir / f"{model_name}.pth")

        # W&B logging
        if run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_acc": acc,
                    "lr": cur_lr,
                    "best_val_acc_so_far": best_acc,
                }
            )

    # Metadata
    meta = {
        "kind": "neural",
        "model_name": model_name,
        "backbone": args.backbone,
        "use_crops": cfg.use_crops,
        "augment": args.augment,
        "seed": cfg.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": cfg.num_workers,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "betas": args.betas,
        "eps": args.eps,
        "scheduler": args.scheduler,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "min_lr": args.min_lr,
        "hidden_dim": args.hidden_dim,
        "dropout": dropout,
        "label_smoothing": args.label_smoothing,
        "img_size": args.img_size,
        "resize_train": args.resize_train,
        "crop_scale_min": args.crop_scale_min,
        "crop_scale_max": args.crop_scale_max,
        "rotation_deg": args.rotation_deg,
        "hflip_p": args.hflip_p,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "train_time_sec": time.time() - t0,
    }
    (cfg.outputs_dir / f"meta_{model_name}.json").write_text(json.dumps(meta, indent=2))

    if run is not None:
        wandb.summary["best_val_acc"] = best_acc
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["train_time_sec"] = meta["train_time_sec"]
        wandb.finish()

    print(f"Finished. Best Val Acc: {best_acc:.4f} @ epoch {best_epoch}")
    print(f"Saved model: outputs/{model_name}.pth")


if __name__ == "__main__":
    main()

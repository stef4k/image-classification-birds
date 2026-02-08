import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from birds_ml.arcface_margin import ArcMarginProduct
from birds_ml.config import Config
from birds_ml.embedder import build_backbone
from birds_ml.utils import ensure_dir, set_seed


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROW_CLASSES = ["American_Crow", "Fish_Crow"]


@dataclass(frozen=True)
class WeightedSample:
    path: Path
    label: int
    weight: float
    source: str


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


class WeightedSampleDataset(Dataset):
    def __init__(self, samples: List[WeightedSample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.transform(img)
        return x, int(s.label), float(s.weight), str(s.path)


def list_images_recursive(root: Path) -> List[Path]:
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return sorted(paths)


def normalize_class_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def resolve_class_dir(root: Path, class_name: str) -> Optional[Path]:
    if not root.exists() or not root.is_dir():
        return None

    exact = root / class_name
    if exact.exists() and exact.is_dir():
        return exact

    target = normalize_class_name(class_name)
    for p in root.iterdir():
        if p.is_dir() and normalize_class_name(p.name) == target:
            return p
    return None


def count_crow_like_files(root: Optional[Path]) -> int:
    if root is None or not root.exists() or not root.is_dir():
        return 0
    total = 0
    for cname in CROW_CLASSES:
        cdir = resolve_class_dir(root, cname)
        if cdir is not None:
            total += len(list_images_recursive(cdir))
    return total


def choose_pseudo_dir(cfg: Config, args) -> Optional[Path]:
    if args.pseudo_dir is not None:
        return args.pseudo_dir

    candidates = []
    if args.use_crops:
        candidates.append(cfg.data_dir / "pseudo_labels_cropped")
    candidates.append(cfg.data_dir / "pseudo_labels")

    best = None
    best_count = -1
    for cand in candidates:
        c = count_crow_like_files(cand)
        if c > best_count:
            best = cand
            best_count = c
    return best


def load_confidence_lookup(
    conf_csv: Optional[Path],
) -> Dict[str, tuple[str, float]]:
    if conf_csv is None or not conf_csv.exists():
        return {}

    df = pd.read_csv(conf_csv)
    required = {"path", "predicted_label", "confidence"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{conf_csv} must contain columns: {sorted(required)}. Found: {list(df.columns)}"
        )
    lookup = {}
    for _, row in df.iterrows():
        predicted = str(row["predicted_label"])
        conf = float(row["confidence"])
        keys = [str(row["path"])]
        if "relative_path" in row and pd.notna(row["relative_path"]):
            keys.append(str(row["relative_path"]))
        for k in list(keys):
            keys.append(Path(k).name)
        for key in keys:
            lookup[key] = (predicted, conf)
    return lookup


def load_real_samples(split_dir: Path, class_to_idx: Dict[str, int], weight: float) -> List[WeightedSample]:
    samples: List[WeightedSample] = []
    for cname, idx in class_to_idx.items():
        cdir = resolve_class_dir(split_dir, cname)
        if cdir is None:
            continue
        for p in list_images_recursive(cdir):
            samples.append(WeightedSample(path=p, label=idx, weight=weight, source="real"))
    return samples


def load_pseudo_samples(
    pseudo_dir: Optional[Path],
    class_to_idx: Dict[str, int],
    conf_lookup: Dict[str, tuple[str, float]],
    pseudo_min_conf: float,
    pseudo_weight: float,
    pseudo_ignore_conf_filter: bool,
) -> List[WeightedSample]:
    if pseudo_dir is None or not pseudo_dir.exists():
        return []

    samples: List[WeightedSample] = []
    for cname, idx in class_to_idx.items():
        cdir = resolve_class_dir(pseudo_dir, cname)
        if cdir is None:
            continue

        for p in list_images_recursive(cdir):
            if pseudo_ignore_conf_filter:
                sample_weight = pseudo_weight
            else:
                conf_row = conf_lookup.get(p.name)
                if conf_row is None:
                    conf_row = conf_lookup.get(str(p))
                if conf_row is None:
                    rel = p.relative_to(pseudo_dir)
                    conf_row = conf_lookup.get(str(rel))
                if conf_row is not None:
                    predicted_label, confidence = conf_row
                    if normalize_class_name(predicted_label) != normalize_class_name(cname) or confidence < pseudo_min_conf:
                        continue
                    sample_weight = pseudo_weight * confidence
                else:
                    sample_weight = pseudo_weight

            samples.append(
                WeightedSample(
                    path=p,
                    label=idx,
                    weight=float(sample_weight),
                    source="pseudo",
                )
            )
    return samples


def build_transforms(img_size: int, mean, std, aug_mode: str):
    common = [
        SquarePad(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    if aug_mode == "none":
        train_tfm = transforms.Compose(common)
    elif aug_mode == "strong":
        train_tfm = transforms.Compose(
            [
                SquarePad(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=8),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        train_tfm = transforms.Compose(
            [
                SquarePad(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    val_tfm = transforms.Compose(common)
    return train_tfm, val_tfm


def forward_eval_logits(backbone_model, head, imgs: torch.Tensor, head_type: str):
    feats = backbone_model(imgs)
    if head_type == "arcface":
        norm_feats = F.normalize(feats)
        norm_weights = F.normalize(head.weight)
        logits = F.linear(norm_feats, norm_weights) * head.s
        return logits
    return head(feats)


def evaluate_loader(
    backbone_model,
    head,
    dl,
    amp_enabled: bool,
    device: torch.device,
    head_type: str,
    val_tta_hflip: bool,
):
    correct = 0
    total = 0
    per_class_correct = [0, 0]
    per_class_total = [0, 0]

    with torch.no_grad():
        for imgs, labels, _, _ in dl:
            imgs = imgs.to(device)
            labels = labels.to(device)
            with autocast(enabled=amp_enabled):
                logits = forward_eval_logits(backbone_model, head, imgs, head_type=head_type)
                if val_tta_hflip:
                    imgs_flip = torch.flip(imgs, [3])
                    logits_flip = forward_eval_logits(backbone_model, head, imgs_flip, head_type=head_type)
                    logits = (logits + logits_flip) / 2.0

            pred = torch.argmax(logits, dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))

            for i in range(len(CROW_CLASSES)):
                cls_mask = labels == i
                per_class_total[i] += int(cls_mask.sum().item())
                per_class_correct[i] += int(((pred == labels) & cls_mask).sum().item())

    acc = correct / max(total, 1)
    per_class_acc = [per_class_correct[i] / max(per_class_total[i], 1) for i in range(len(CROW_CLASSES))]
    balanced_acc = float(sum(per_class_acc) / len(per_class_acc))
    return acc, balanced_acc, per_class_acc


def compute_per_example_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    loss_type: str,
    label_smoothing: float,
    focal_gamma: float,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    if loss_type == "focal":
        p = torch.softmax(logits, dim=1).gather(1, labels.view(-1, 1)).squeeze(1)
        p = p.clamp(1e-6, 1.0 - 1e-6)
        focal_factor = (1.0 - p).pow(focal_gamma)
        return focal_factor * ce
    return ce


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="vit_so150m2_384")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--finetune_backbone", action="store_true")
    parser.add_argument("--head_type", choices=["linear", "arcface"], default="linear")
    parser.add_argument("--arcface_s", type=float, default=30.0)
    parser.add_argument("--arcface_m", type=float, default=0.50)
    parser.add_argument("--loss_type", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--aug_mode", choices=["none", "light", "strong"], default="light")
    parser.add_argument("--val_tta_hflip", action="store_true")
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--real_weight", type=float, default=1.0)
    parser.add_argument("--pseudo_weight", type=float, default=0.35)
    parser.add_argument("--pseudo_min_conf", type=float, default=0.95)
    parser.add_argument(
        "--pseudo_ignore_conf_filter",
        action="store_true",
        help="Use pseudo labels from folder names directly and ignore confidence CSV filtering.",
    )
    parser.add_argument(
        "--train_dir",
        type=Path,
        default=None,
        help="Optional override for training root (class folders). Default: data/train_images(_cropped).",
    )
    parser.add_argument(
        "--val_dir",
        type=Path,
        default=None,
        help="Optional override for validation root (class folders). Default: data/val_images(_cropped).",
    )
    parser.add_argument(
        "--secondary_val_dir",
        type=Path,
        default=None,
        help="Optional second validation root. Default: data/val_images(_cropped).",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["acc", "balanced_acc"],
        default="balanced_acc",
        help="Metric used to keep best checkpoint.",
    )
    parser.add_argument(
        "--pseudo_dir",
        type=Path,
        default=None,
        help="Default: data/pseudo_labels or data/pseudo_labels_cropped if available with --use_crops.",
    )
    parser.add_argument(
        "--pseudo_conf_csv",
        type=Path,
        default=Path("outputs/extra_images_cv_ensemble_predictions_with_conf.csv"),
    )
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    cfg = replace(Config(), use_crops=args.use_crops)
    ensure_dir(cfg.outputs_dir)
    set_seed(cfg.seed)

    device_name = cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    device = torch.device(device_name)
    amp_enabled = device.type == "cuda"

    class_to_idx = {name: i for i, name in enumerate(CROW_CLASSES)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    pseudo_dir = choose_pseudo_dir(cfg, args)

    conf_lookup = load_confidence_lookup(args.pseudo_conf_csv)

    train_dir = args.train_dir or cfg.train_dir
    val_dir = args.val_dir or cfg.val_dir
    secondary_val_dir = args.secondary_val_dir or cfg.val_dir

    train_samples = load_real_samples(train_dir, class_to_idx, weight=args.real_weight)
    val_samples = load_real_samples(val_dir, class_to_idx, weight=1.0)
    secondary_val_samples: List[WeightedSample] = []
    if secondary_val_dir.resolve() != val_dir.resolve():
        secondary_val_samples = load_real_samples(secondary_val_dir, class_to_idx, weight=1.0)
    pseudo_samples = load_pseudo_samples(
        pseudo_dir=pseudo_dir,
        class_to_idx=class_to_idx,
        conf_lookup=conf_lookup,
        pseudo_min_conf=args.pseudo_min_conf,
        pseudo_weight=args.pseudo_weight,
        pseudo_ignore_conf_filter=bool(args.pseudo_ignore_conf_filter),
    )
    train_samples.extend(pseudo_samples)

    if not train_samples:
        raise ValueError("No training samples found for crow separator.")
    if not val_samples:
        raise ValueError("No validation samples found for crow separator.")

    train_counts = [sum(1 for s in train_samples if s.label == i) for i in range(len(CROW_CLASSES))]
    val_counts = [sum(1 for s in val_samples if s.label == i) for i in range(len(CROW_CLASSES))]
    secondary_val_counts = [sum(1 for s in secondary_val_samples if s.label == i) for i in range(len(CROW_CLASSES))]
    pseudo_counts = [sum(1 for s in pseudo_samples if s.label == i) for i in range(len(CROW_CLASSES))]
    if any(c == 0 for c in val_counts):
        raise ValueError(f"Validation directory must contain both crow classes. Counts: {dict(zip(CROW_CLASSES, val_counts))}")
    if secondary_val_samples and any(c == 0 for c in secondary_val_counts):
        raise ValueError(
            f"Secondary validation directory must contain both crow classes. Counts: {dict(zip(CROW_CLASSES, secondary_val_counts))}"
        )

    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir}")
    print(f"Pseudo dir: {pseudo_dir}")
    if secondary_val_samples:
        print(f"Secondary val dir: {secondary_val_dir}")
    print(f"Train counts: {dict(zip(CROW_CLASSES, train_counts))}")
    print(f"Val counts:   {dict(zip(CROW_CLASSES, val_counts))}")
    if secondary_val_samples:
        print(f"Secondary val counts: {dict(zip(CROW_CLASSES, secondary_val_counts))}")
    print(f"Pseudo kept:  {dict(zip(CROW_CLASSES, pseudo_counts))}")

    backbone_model, model_config = build_backbone(args.backbone)
    backbone_model.to(device)

    if args.finetune_backbone:
        backbone_model.train()
        for p in backbone_model.parameters():
            p.requires_grad = True
        print("Backbone mode: fine-tuning enabled")
    else:
        backbone_model.eval()
        for p in backbone_model.parameters():
            p.requires_grad = False
        print("Backbone mode: frozen")

    train_tfm, val_tfm = build_transforms(
        args.img_size,
        model_config["mean"],
        model_config["std"],
        aug_mode=args.aug_mode,
    )
    train_dl = DataLoader(
        WeightedSampleDataset(train_samples, train_tfm),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_dl = DataLoader(
        WeightedSampleDataset(val_samples, val_tfm),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    secondary_val_dl = None
    if secondary_val_samples:
        secondary_val_dl = DataLoader(
            WeightedSampleDataset(secondary_val_samples, val_tfm),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    if args.head_type == "arcface":
        head = ArcMarginProduct(
            backbone_model.num_features,
            len(CROW_CLASSES),
            s=args.arcface_s,
            m=args.arcface_m,
        ).to(device)
    else:
        head = nn.Linear(backbone_model.num_features, len(CROW_CLASSES)).to(device)
        nn.init.constant_(head.bias, 0.0)
        nn.init.normal_(head.weight, std=0.01)

    train_labels = torch.tensor([s.label for s in train_samples], dtype=torch.long)
    label_counts = torch.bincount(train_labels, minlength=len(CROW_CLASSES)).float()
    class_weights = label_counts.sum() / (len(CROW_CLASSES) * label_counts.clamp_min(1.0))
    class_weights = class_weights.to(device)

    if args.finetune_backbone:
        optimizer = optim.AdamW(
            [
                {"params": backbone_model.parameters(), "lr": args.lr_backbone},
                {"params": head.parameters(), "lr": args.lr_head},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=amp_enabled)

    crop_suffix = "_cropped" if cfg.use_crops else ""
    run_name = args.run_name or f"crow_separator_{args.head_type}_{args.backbone}{crop_suffix}_{args.img_size}"
    ckpt_path = cfg.outputs_dir / f"{run_name}.pth"
    meta_path = cfg.outputs_dir / f"meta_{run_name}.json"

    best_selection_metric = -1.0
    best_acc = -1.0
    best_balanced_acc = -1.0
    best_per_class_acc = [0.0, 0.0]
    best_secondary_acc = -1.0
    best_secondary_balanced_acc = -1.0
    best_secondary_per_class_acc = [0.0, 0.0]
    for epoch in range(args.epochs):
        head.train()
        if args.finetune_backbone:
            backbone_model.train()
        else:
            backbone_model.eval()

        running_loss = 0.0
        for imgs, labels, weights, _ in train_dl:
            imgs = imgs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()

            with autocast(enabled=amp_enabled):
                if args.finetune_backbone:
                    feats = backbone_model(imgs)
                else:
                    with torch.no_grad():
                        feats = backbone_model(imgs)

                if args.head_type == "arcface":
                    logits = head(feats, labels)
                else:
                    logits = head(feats)

                per_example = compute_per_example_loss(
                    logits=logits,
                    labels=labels,
                    class_weights=class_weights,
                    loss_type=args.loss_type,
                    label_smoothing=args.label_smoothing,
                    focal_gamma=args.focal_gamma,
                )
                loss = (per_example * weights).sum() / weights.sum().clamp_min(1e-6)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

        scheduler.step()

        head.eval()
        backbone_model.eval()
        acc, balanced_acc, per_class_acc = evaluate_loader(
            backbone_model=backbone_model,
            head=head,
            dl=val_dl,
            amp_enabled=amp_enabled,
            device=device,
            head_type=args.head_type,
            val_tta_hflip=bool(args.val_tta_hflip),
        )
        secondary_acc = None
        secondary_balanced_acc = None
        secondary_per_class_acc = None
        if secondary_val_dl is not None:
            secondary_acc, secondary_balanced_acc, secondary_per_class_acc = evaluate_loader(
                backbone_model=backbone_model,
                head=head,
                dl=secondary_val_dl,
                amp_enabled=amp_enabled,
                device=device,
                head_type=args.head_type,
                val_tta_hflip=bool(args.val_tta_hflip),
            )
        selection_metric = acc if args.selection_metric == "acc" else balanced_acc
        avg_loss = running_loss / max(1, len(train_dl))
        msg = (
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | "
            f"Balanced: {balanced_acc:.4f} | "
            f"American: {per_class_acc[0]:.4f} | Fish: {per_class_acc[1]:.4f}"
        )
        if secondary_acc is not None and secondary_balanced_acc is not None and secondary_per_class_acc is not None:
            msg += (
                f" || Secondary Acc: {secondary_acc:.4f} | "
                f"Secondary Balanced: {secondary_balanced_acc:.4f} | "
                f"Secondary American: {secondary_per_class_acc[0]:.4f} | "
                f"Secondary Fish: {secondary_per_class_acc[1]:.4f}"
            )
        print(msg)

        if selection_metric >= best_selection_metric:
            best_selection_metric = selection_metric
            best_acc = acc
            best_balanced_acc = balanced_acc
            best_per_class_acc = [float(per_class_acc[0]), float(per_class_acc[1])]
            if secondary_acc is not None and secondary_balanced_acc is not None and secondary_per_class_acc is not None:
                best_secondary_acc = float(secondary_acc)
                best_secondary_balanced_acc = float(secondary_balanced_acc)
                best_secondary_per_class_acc = [
                    float(secondary_per_class_acc[0]),
                    float(secondary_per_class_acc[1]),
                ]
            torch.save(
                {
                    "backbone": backbone_model.state_dict(),
                    "head": head.state_dict(),
                    "config": model_config,
                    "class_to_idx": class_to_idx,
                    "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
                    "head_type": args.head_type,
                    "arcface_s": args.arcface_s,
                    "arcface_m": args.arcface_m,
                },
                ckpt_path,
            )

            meta = {
                "kind": "crow_separator_binary",
                "backbone": args.backbone,
                "img_size": args.img_size,
                "use_crops": bool(args.use_crops),
                "finetune_backbone": bool(args.finetune_backbone),
                "head_type": args.head_type,
                "arcface_s": args.arcface_s,
                "arcface_m": args.arcface_m,
                "loss_type": args.loss_type,
                "focal_gamma": args.focal_gamma,
                "aug_mode": args.aug_mode,
                "val_tta_hflip": bool(args.val_tta_hflip),
                "label_smoothing": args.label_smoothing,
                "weight_decay": args.weight_decay,
                "class_to_idx": class_to_idx,
                "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
                "train_counts": dict(zip(CROW_CLASSES, train_counts)),
                "val_counts": dict(zip(CROW_CLASSES, val_counts)),
                "train_dir": str(train_dir),
                "val_dir": str(val_dir),
                "secondary_val_dir": str(secondary_val_dir) if secondary_val_samples else None,
                "secondary_val_counts": dict(zip(CROW_CLASSES, secondary_val_counts)) if secondary_val_samples else None,
                "pseudo_counts_used": dict(zip(CROW_CLASSES, pseudo_counts)),
                "pseudo_dir": str(pseudo_dir),
                "pseudo_conf_csv": str(args.pseudo_conf_csv),
                "pseudo_min_conf": args.pseudo_min_conf,
                "real_weight": args.real_weight,
                "pseudo_weight": args.pseudo_weight,
                "selection_metric": args.selection_metric,
                "best_selection_metric": best_selection_metric,
                "best_val_acc": best_acc,
                "best_balanced_acc": best_balanced_acc,
                "best_val_acc_american_crow": best_per_class_acc[0],
                "best_val_acc_fish_crow": best_per_class_acc[1],
                "best_secondary_val_acc": best_secondary_acc if secondary_val_samples else None,
                "best_secondary_balanced_acc": best_secondary_balanced_acc if secondary_val_samples else None,
                "best_secondary_val_acc_american_crow": best_secondary_per_class_acc[0] if secondary_val_samples else None,
                "best_secondary_val_acc_fish_crow": best_secondary_per_class_acc[1] if secondary_val_samples else None,
            }
            meta_path.write_text(json.dumps(meta, indent=2))

    summary = (
        f"Finished. Best Val Acc: {best_acc:.4f} | "
        f"Best Balanced Acc: {best_balanced_acc:.4f} | "
        f"Selection metric ({args.selection_metric}): {best_selection_metric:.4f}"
    )
    if secondary_val_samples:
        summary += (
            f" | Best Secondary Val Acc: {best_secondary_acc:.4f}"
            f" | Best Secondary Balanced: {best_secondary_balanced_acc:.4f}"
        )
    print(summary)
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved metadata:   {meta_path}")


if __name__ == "__main__":
    main()

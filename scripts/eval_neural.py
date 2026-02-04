import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

from birds_ml.config import Config
from birds_ml.data import load_val_with_given_mapping
from birds_ml.features import SampleDataset
from birds_ml.embedder import build_backbone
from torchvision.transforms import functional as TF


class SquarePad:
    def __init__(self, target_size):
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


def main():
    cfg = Config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_so150m2_384")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--val_dir", default=str(cfg.data_dir / "f_images_cropped"))
    ap.add_argument("--use_crops", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--ckpt_glob", default=None, help="Glob pattern under outputs/ for CV fold checkpoints.")
    ap.add_argument("--no_tta", action="store_true", help="Disable horizontal-flip TTA.")
    args = ap.parse_args()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    crop_suffix = "_cropped" if args.use_crops else ""
    meta_candidates = [
        cfg.outputs_dir / f"meta_timm_finetune_{args.backbone}{crop_suffix}_{args.img_size}.json",
        cfg.outputs_dir / f"meta_frozen_timm_{args.backbone}{crop_suffix}_{args.img_size}.json",
        cfg.outputs_dir / f"meta_timm_{args.backbone}{crop_suffix}_{args.img_size}.json",
    ]
    meta_path = next((p for p in meta_candidates if p.exists()), None)
    if meta_path is None:
        raise FileNotFoundError(
            f"Meta not found for backbone={args.backbone} img_size={args.img_size}. "
            "Expected meta_timm_finetune_*, meta_frozen_timm_*, or meta_timm_*."
        )

    meta = json.loads(meta_path.read_text())
    class_to_idx = meta["class_to_idx"]
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}

    def _resolve_checkpoints():
        if args.ckpt_glob:
            return sorted(cfg.outputs_dir.glob(args.ckpt_glob))
        patterns = [
            f"cv_fold*_timm_finetune_{args.backbone}{crop_suffix}_{args.img_size}.pth",
            f"cv_fold*_frozen_timm_{args.backbone}{crop_suffix}_{args.img_size}.pth",
            f"cv_fold*_timm_{args.backbone}{crop_suffix}_{args.img_size}.pth",
        ]
        for pattern in patterns:
            matches = sorted(cfg.outputs_dir.glob(pattern))
            if matches:
                return matches
        single_candidates = [
            cfg.outputs_dir / f"timm_finetune_{args.backbone}{crop_suffix}_{args.img_size}.pth",
            cfg.outputs_dir / f"frozen_timm_{args.backbone}{crop_suffix}_{args.img_size}.pth",
            cfg.outputs_dir / f"timm_{args.backbone}{crop_suffix}_{args.img_size}.pth",
        ]
        for p in single_candidates:
            if p.exists():
                return [p]
        return []

    ckpt_paths = _resolve_checkpoints()
    if not ckpt_paths:
        raise FileNotFoundError("No checkpoints found for the given backbone/img_size.")

    models = []
    saved_config = None
    for ckpt_path in ckpt_paths:
        checkpoint = torch.load(ckpt_path, map_location=device)
        backbone_model, _ = build_backbone(args.backbone)
        backbone_model.load_state_dict(checkpoint["backbone"])
        backbone_model.to(device).eval()

        input_dim = backbone_model.num_features
        head = nn.Linear(input_dim, len(idx_to_class)).to(device)
        head.load_state_dict(checkpoint["head"])
        head.to(device).eval()

        if saved_config is None:
            saved_config = checkpoint.get(
                "config", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
            )
        models.append((backbone_model, head))

    if saved_config is None:
        saved_config = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    val_tfm = transforms.Compose([
        SquarePad(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=saved_config["mean"], std=saved_config["std"]),
    ])

    val_dir = Path(args.val_dir)
    val_samples = load_val_with_given_mapping(val_dir, class_to_idx)
    if not val_samples:
        raise ValueError(f"No validation images found under: {val_dir}")

    dl = DataLoader(
        SampleDataset(val_samples, val_tfm),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels, _ in dl:
            imgs = imgs.to(device)
            labels = labels.to(device)

            probs_sum = None
            for backbone_model, head in models:
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    feats = backbone_model(imgs)
                    logits = head(feats)
                    probs = torch.softmax(logits, dim=1)

                    if not args.no_tta:
                        imgs_flip = torch.flip(imgs, dims=[3])
                        feats_flip = backbone_model(imgs_flip)
                        logits_flip = head(feats_flip)
                        probs_flip = torch.softmax(logits_flip, dim=1)
                        probs = 0.5 * (probs + probs_flip)

                if probs_sum is None:
                    probs_sum = probs
                else:
                    probs_sum += probs

            probs_avg = probs_sum / len(models)
            preds = torch.argmax(probs_avg, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(idx_to_class))))
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion matrix:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(cm_df)


if __name__ == "__main__":
    main()

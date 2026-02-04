import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from birds_ml.config import Config
from birds_ml.data import load_val_with_given_mapping
from birds_ml.embedder import build_backbone
from birds_ml.features import SampleDataset
from birds_ml.utils import ensure_dir


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


def main():
    cfg = Config()
    ensure_dir(cfg.outputs_dir)

    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_so150m2_384")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--use_crops", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None, help="Path to .pth checkpoint")
    ap.add_argument("--meta", type=str, default=None, help="Path to meta json")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path")
    ap.add_argument("--only_mistakes", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=None)
    args = ap.parse_args()

    cfg = replace(cfg, use_crops=args.use_crops)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    num_workers = cfg.num_workers if args.num_workers is None else args.num_workers

    crop_suffix = "_cropped" if cfg.use_crops else ""
    model_name = f"frozen_timm_{args.backbone}{crop_suffix}_{args.img_size}"

    meta_path = Path(args.meta) if args.meta else (cfg.outputs_dir / f"meta_{model_name}.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    class_to_idx = meta["class_to_idx"]
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}

    ckpt_path = Path(args.ckpt) if args.ckpt else (cfg.outputs_dir / f"{model_name}.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build model
    backbone_model, model_config = build_backbone(args.backbone)
    head = nn.Linear(backbone_model.num_features, len(class_to_idx))

    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone_model.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])

    backbone_model.to(device).eval()
    head.to(device).eval()

    # transforms (match train_neural.py)
    val_tfm = transforms.Compose([
        SquarePad(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_config["mean"], std=model_config["std"]),
    ])

    # data
    val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)
    val_dl = DataLoader(
        SampleDataset(val_samples, val_tfm),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    rows = []
    with torch.no_grad():
        for imgs, labels, paths in val_dl:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats = backbone_model(imgs)
            logits = head(feats)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

            for p, t, c, path in zip(pred.cpu().tolist(), labels.cpu().tolist(), conf.cpu().tolist(), paths):
                pred_name = idx_to_class[int(p)]
                true_name = idx_to_class[int(t)]
                correct = pred_name == true_name
                if args.only_mistakes and correct:
                    continue
                rows.append({
                    "path": path,
                    "true_label": true_name,
                    "pred_label": pred_name,
                    "confidence": float(c),
                    "correct": bool(correct),
                })

    out_path = Path(args.out) if args.out else (cfg.outputs_dir / f"val_mistakes_{model_name}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "true_label", "pred_label", "confidence", "correct"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV: {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

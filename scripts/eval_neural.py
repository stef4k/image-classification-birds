import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision.transforms import functional as TF

from birds_ml.config import Config
from birds_ml.data import load_val_with_given_mapping
from birds_ml.features import SampleDataset
from birds_ml.embedder import build_backbone

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # No init needed here as we load from state_dict

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
    ap.add_argument("model_name_override", nargs="?", default=None, help="Optional: Specific model name to load (e.g. arcface_final_...)")
    ap.add_argument("--no_tta", action="store_true", help="Disable horizontal-flip TTA.")
    args = ap.parse_args()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    crop_suffix = "_cropped" if args.use_crops else ""
    
    # We look for ArcFace first, then Linear, then Legacy
    arcface_name = f"arcface_final_{args.backbone}{crop_suffix}_{args.img_size}"
    linear_name = f"linear_{args.backbone}{crop_suffix}_{args.img_size}"
    legacy_name = f"frozen_timm_{args.backbone}{crop_suffix}_{args.img_size}"
    
    meta_candidates = [
        cfg.outputs_dir / f"meta_{arcface_name}.json",
        cfg.outputs_dir / f"meta_{linear_name}.json",
        cfg.outputs_dir / f"meta_{legacy_name}.json",
        # Fallback patterns
        cfg.outputs_dir / f"meta_timm_finetune_{args.backbone}{crop_suffix}_{args.img_size}.json",
        cfg.outputs_dir / f"meta_timm_{args.backbone}{crop_suffix}_{args.img_size}.json",
    ]
    
    # If user provided a specific name, prioritize that
    if args.model_name_override:
        # Strip extension if user accidentally added .pth
        clean_name = args.model_name_override.replace(".pth", "").replace(".json", "")
        # Remove 'meta_' prefix if user added it
        if clean_name.startswith("meta_"):
             clean_name = clean_name[5:]
             
        meta_candidates.insert(0, cfg.outputs_dir / f"meta_{clean_name}.json")

    meta_path = next((p for p in meta_candidates if p.exists()), None)
    if meta_path is None:
        raise FileNotFoundError(
            f"Meta not found for backbone={args.backbone} img_size={args.img_size}. "
            "Tried checking standard naming conventions."
        )

    print(f"Loading Metadata: {meta_path.name}")
    meta = json.loads(meta_path.read_text())
    class_to_idx = meta["class_to_idx"]
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    
    # Auto-detect Architecture
    is_arcface = "arcface" in meta.get("kind", "")
    print(f"Architecture: {'ArcFace' if is_arcface else 'Linear'}")

    # The meta file usually corresponds to a .pth file of the same name (minus 'meta_')
    # e.g., meta_arcface_final_...json -> arcface_final_...pth
    model_name = meta_path.stem.replace("meta_", "")
    ckpt_path = cfg.outputs_dir / f"{model_name}.pth"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # model
    print(f"Loading Weights: {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    backbone_model, _ = build_backbone(args.backbone)
    backbone_model.load_state_dict(checkpoint["backbone"])
    backbone_model.to(device).eval()

    input_dim = backbone_model.num_features
    
    if is_arcface:
        head = ArcMarginProduct(input_dim, len(idx_to_class), s=30.0, m=0.50).to(device)
    else:
        head = nn.Linear(input_dim, len(idx_to_class)).to(device)
        
    head.load_state_dict(checkpoint["head"])
    head.to(device).eval()

    saved_config = checkpoint.get(
        "config", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    )

    # data
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

    # inference + evaluation
    y_true = []
    y_pred = []
    
    print(f"Evaluating on {len(val_samples)} images...")
    
    with torch.no_grad():
        for imgs, labels, _ in dl:
            imgs = imgs.to(device)
            labels = labels.to(device) # keep labels for tracking, not used in forward
            
            # image
            feats1 = backbone_model(imgs)
            if is_arcface:
                norm_feats1 = F.normalize(feats1)
                norm_weights = F.normalize(head.weight)
                logits1 = F.linear(norm_feats1, norm_weights) * head.s
            else:
                logits1 = head(feats1)
            
            final_probs = torch.softmax(logits1, dim=1)

            # TTA
            if not args.no_tta:
                imgs_flip = torch.flip(imgs, dims=[3])
                feats2 = backbone_model(imgs_flip)
                
                if is_arcface:
                    norm_feats2 = F.normalize(feats2)
                    logits2 = F.linear(norm_feats2, norm_weights) * head.s
                else:
                    logits2 = head(feats2)
                
                probs2 = torch.softmax(logits2, dim=1)
                final_probs = 0.5 * (final_probs + probs2)

            preds = torch.argmax(final_probs, dim=1)

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
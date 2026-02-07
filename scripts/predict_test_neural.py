import argparse
import json
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from dataclasses import replace
from tqdm import tqdm

from birds_ml.config import Config
from birds_ml.data import load_test_recursive
from birds_ml.features import SampleDataset
from birds_ml.embedder import build_backbone
# from birds_ml.head import CustomHead
from birds_ml.utils import ensure_dir

KAGGLE_NAME_TO_IDX = {
    "Groove_billed_Ani": 0, "Red_winged_Blackbird": 1, "Rusty_Blackbird": 2, 
    "Gray_Catbird": 3, "Brandt_Cormorant": 4, "Eastern_Towhee": 5, 
    "Indigo_Bunting": 6, "Brewer_Blackbird": 7, "Painted_Bunting": 8, 
    "Bobolink": 9, "Lazuli_Bunting": 10, "Yellow_headed_Blackbird": 11, 
    "American_Crow": 12, "Fish_Crow": 13, "Brown_Creeper": 14, 
    "Yellow_billed_Cuckoo": 15, "Yellow_breasted_Chat": 16, 
    "Black_billed_Cuckoo": 17, "Gray_crowned_Rosy_Finch": 18, 
    "Bronzed_Cowbird": 19,
}

import torch.nn.functional as F
from torchvision.transforms import functional as TF

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
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=128, padding_mode='constant')

def main():
    cfg = Config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--test_dir", default=None, help="Optional test directory override (e.g. data/extra_images).")
    parser.add_argument(
        "--out_with_conf",
        default=None,
        help="Optional detailed CSV with predicted label and confidence per image.",
    )
    parser.add_argument("--ckpt_glob", default=None, help="Glob pattern under outputs/ for CV fold checkpoints.")
    parser.add_argument("--no_tta", action="store_true", help="Disable horizontal-flip TTA.")
    args = parser.parse_args()
    
    cfg = replace(cfg, use_crops=args.use_crops)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    crop_suffix = "_cropped" if cfg.use_crops else ""

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
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    print(f"Using meta: {meta_path.name}")

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

    print(f"Loading {len(ckpt_paths)} checkpoint(s).")

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
            transforms.Normalize(mean=saved_config['mean'], std=saved_config['std'])
        ])

    # data
    test_dir = Path(args.test_dir) if args.test_dir else cfg.test_dir
    test_samples = sorted(load_test_recursive(test_dir), key=lambda s: s.path.name.lower())
    test_dir_abs = test_dir.resolve()
    
    dl = DataLoader(
        SampleDataset(test_samples, val_tfm), 
        batch_size=32, 
        shuffle=False, 
        num_workers=0
    )

    all_preds = []
    all_conf = []
    all_files = []
    all_relpaths = []
    
    print(f"Running Inference on {len(test_samples)} images from: {test_dir}")
    
    with torch.no_grad():
        for imgs, _, paths in tqdm(dl):
            imgs = imgs.to(device)
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
            conf, preds = torch.max(probs_avg, dim=1)
            preds = preds.cpu().numpy()
            conf = conf.cpu().numpy()

            all_preds.extend(preds)
            all_conf.extend(conf.tolist())
            batch_paths = [Path(p) for p in paths]
            all_files.extend([p.name for p in batch_paths])
            for p in batch_paths:
                try:
                    rel = p.resolve().relative_to(test_dir_abs)
                    all_relpaths.append(str(rel))
                except Exception:
                    all_relpaths.append(p.name)

    kaggle_ids = []
    for p in all_preds:
        class_name = idx_to_class[p]
        if class_name not in KAGGLE_NAME_TO_IDX:
            print(f"WARNING: Class '{class_name}' not found in Kaggle mapping! Defaulting to -1.")
            kaggle_ids.append(-1)
        else:
            kaggle_ids.append(KAGGLE_NAME_TO_IDX[class_name])
    
    df = pd.DataFrame({"path": all_files, "class_idx": kaggle_ids})
    df.to_csv(cfg.outputs_dir / args.out, index=False)
    print(f"Saved {args.out}")

    detailed_rows = []
    for fname, relpath, pred_idx, conf_val, kaggle_idx in zip(
        all_files, all_relpaths, all_preds, all_conf, kaggle_ids
    ):
        class_name = idx_to_class[int(pred_idx)]
        detailed_rows.append(
            {
                "path": fname,
                "relative_path": relpath,
                "predicted_label": class_name,
                "predicted_idx_internal": int(pred_idx),
                "class_idx": int(kaggle_idx),
                "confidence": float(conf_val),
            }
        )

    detailed_out = args.out_with_conf
    if detailed_out is None:
        out_stem = Path(args.out).stem
        detailed_out = f"{out_stem}_with_conf.csv"
    detailed_path = cfg.outputs_dir / detailed_out
    pd.DataFrame(detailed_rows).to_csv(detailed_path, index=False)
    print(f"Saved {detailed_path.name}")

if __name__ == "__main__":
    main()

import argparse
import json
import math
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from dataclasses import replace
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from birds_ml.config import Config
from birds_ml.data import load_test_recursive
from birds_ml.features import SampleDataset
from birds_ml.embedder import build_backbone
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

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # no init needed here as we load from state_dict

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
        return TF.pad(img, (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2)), fill=128, padding_mode='constant')

def main():
    cfg = Config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--use_arcface", action="store_true", help="Use ArcFace head logic")
    args = parser.parse_args()
    
    cfg = replace(cfg, use_crops=args.use_crops)
    device = torch.device(cfg.device)
    
    crop_suffix = "_cropped" if cfg.use_crops else ""
    head_prefix = "arcface" if args.use_arcface else "linear"
    
    model_name = f"{head_prefix}_{args.backbone}{crop_suffix}_{args.img_size}"
    print(f"--> Looking for specific model: {model_name}")
    meta_path = cfg.outputs_dir / f"meta_{model_name}.json"
    weights_path = cfg.outputs_dir / f"{model_name}.pth"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}\nDid you train with --use_arcface={args.use_arcface}?")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    meta = json.loads(meta_path.read_text())
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    checkpoint = torch.load(weights_path, map_location=device)
    
    # backbone
    backbone_model, _ = build_backbone(args.backbone)
    backbone_model.load_state_dict(checkpoint['backbone'])
    backbone_model.to(device).eval()
    
    # head
    input_dim = backbone_model.num_features
    
    if args.use_arcface:
        head = ArcMarginProduct(input_dim, len(idx_to_class), s=30.0, m=0.50).to(device)
    else:
        head = nn.Linear(input_dim, len(idx_to_class)).to(device)
        
    head.load_state_dict(checkpoint['head'])
    head.to(device).eval()

    saved_config = checkpoint.get('config', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    
    val_tfm = transforms.Compose([
            SquarePad(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=saved_config['mean'], std=saved_config['std'])
        ])

    # data
    test_samples = sorted(load_test_recursive(cfg.test_dir), key=lambda s: s.path.name.lower())
    
    dl = DataLoader(
        SampleDataset(test_samples, val_tfm), 
        batch_size=32, 
        shuffle=False, 
        num_workers=0
    )

    all_preds = []
    all_files = []
    
    print(f"Running Inference on {len(test_samples)} images...")
    
    with torch.no_grad():
        for imgs, _, paths in tqdm(dl):
            imgs = imgs.to(device)
            
            feats1 = backbone_model(imgs)
            if args.use_arcface:
                norm_feats1 = F.normalize(feats1)
                norm_weights = F.normalize(head.weight)
                logits1 = F.linear(norm_feats1, norm_weights) * head.s
            else:
                logits1 = head(feats1)
            
            imgs_flip = torch.flip(imgs, [3])
            feats2 = backbone_model(imgs_flip)
            if args.use_arcface:
                norm_feats2 = F.normalize(feats2)
                logits2 = F.linear(norm_feats2, norm_weights) * head.s
            else:
                logits2 = head(feats2)
            
            # average
            avg_logits = (logits1 + logits2) / 2.0
            preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_files.extend([Path(p).name for p in paths])

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

if __name__ == "__main__":
    main()
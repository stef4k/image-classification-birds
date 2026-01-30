import argparse
import json
import torch
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
from birds_ml.head import CustomHead
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

def main():
    cfg = Config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()
    
    cfg = replace(cfg, use_crops=args.use_crops)
    device = torch.device(cfg.device)
    
    crop_suffix = "_cropped" if cfg.use_crops else ""
    model_name = f"frozen_timm_{args.backbone}{crop_suffix}_{args.img_size}"
    
    meta_path = cfg.outputs_dir / f"meta_{model_name}.json"
    if not meta_path.exists():
        fallback_name = f"timm_{args.backbone}{crop_suffix}_{args.img_size}"
        fallback_path = cfg.outputs_dir / f"meta_{fallback_name}.json"
        
        if fallback_path.exists():
            print(f"Note: Found model under fallback name: {fallback_name}")
            model_name = fallback_name
            meta_path = fallback_path
        else:
            raise FileNotFoundError(f"Meta not found: {meta_path}. \nDid you train with --backbone {args.backbone} --img_size {args.img_size}?")
    
    meta = json.loads(meta_path.read_text())
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    
    print(f"Loading Model: {model_name}")

    # checkpoint
    weights_path = cfg.outputs_dir / f"{model_name}.pth"
    checkpoint = torch.load(weights_path, map_location=device)
    
    # backbone
    backbone_model, _ = build_backbone(args.backbone)
    backbone_model.load_state_dict(checkpoint['backbone'])
    backbone_model.to(device).eval()
    
    # head
    input_dim = backbone_model.num_features
    head = CustomHead(input_dim, 512, len(idx_to_class), dropout_prob=0.0)
    head.load_state_dict(checkpoint['head'])
    head.to(device).eval()

    saved_config = checkpoint.get('config', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    
    val_tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
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
            feats = backbone_model(imgs)
            logits = head(feats)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
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
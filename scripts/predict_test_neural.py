import argparse
import json
import torch
import pandas as pd
import numpy as np
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
    parser.add_argument("--out", default="submission_neural.csv")
    args = parser.parse_args()
    
    cfg = replace(cfg, use_crops=args.use_crops)
    device = torch.device(cfg.device)
    
    crop_suffix = "_cropped" if cfg.use_crops else ""
    model_name = f"neural_{args.backbone}{crop_suffix}"
    
    # load metadata
    meta_path = cfg.outputs_dir / f"meta_{model_name}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}. Run train_neural.py first.")
    
    meta = json.loads(meta_path.read_text())
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    num_classes = len(idx_to_class)

    print(f"Loading Model: {model_name} | Classes: {num_classes}")

    # load models
    # Backbone (Frozen)
    backbone_model, _ = build_backbone(args.backbone)
    backbone_model.to(device)
    backbone_model.eval()
    
    # head (Trained)
    input_dim = 2048 if args.backbone == "resnet50" else 1280
    head = CustomHead(input_dim, 512, num_classes, dropout_prob=0.0) # no dropout during inference
    
    weights_path = cfg.outputs_dir / f"{model_name}.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
        
    head.load_state_dict(torch.load(weights_path, map_location=device))
    head.to(device)
    head.eval()

    print(f"Predicting on: {cfg.test_dir}")
    test_samples = load_test_recursive(cfg.test_dir)
    if len(test_samples) == 0:
        raise ValueError(f"No test images found in {cfg.test_dir}")
        
    test_samples = sorted(test_samples, key=lambda s: s.path.name.lower())
    
    # standard inference transform (Resize -> Normalize)
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ds = SampleDataset(test_samples, val_tfm)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=cfg.num_workers)

    # inference
    all_preds = []
    all_files = []
    
    print("Running Inference...")
    with torch.no_grad():
        for imgs, _, paths in tqdm(dl):
            imgs = imgs.to(device)
            
            # forward pass
            feats = backbone_model(imgs)
            logits = head(feats)
            
            # predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_files.extend([Path(p).name for p in paths])

    kaggle_ids = []
    unmapped = []
    
    for p in all_preds:
        cname = idx_to_class[p]
        if cname in KAGGLE_NAME_TO_IDX:
            kaggle_ids.append(KAGGLE_NAME_TO_IDX[cname])
        else:
            kaggle_ids.append(-1)
            unmapped.append(cname)

    if unmapped:
        print(f"Warning: {len(set(unmapped))} classes could not be mapped to Kaggle IDs: {list(set(unmapped))[:5]}")

    # save submission
    df = pd.DataFrame({"path": all_files, "class_idx": kaggle_ids})
    path = cfg.outputs_dir / args.out
    df.to_csv(path, index=False)
    print(f"Saved submission to {path}")

if __name__ == "__main__":
    main()
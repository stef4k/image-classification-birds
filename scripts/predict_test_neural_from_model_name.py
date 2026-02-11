import argparse
import json
from pathlib import Path
from dataclasses import replace

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from birds_ml.config import Config
from birds_ml.data import load_test_recursive
from birds_ml.features import SampleDataset
from birds_ml.embedder import build_backbone
from birds_ml.head import CustomHead

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
    parser.add_argument("--model_name", required=True, help="Exact model_name string saved by training.")
    parser.add_argument("--out", default="submission_neural.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(cfg.device)

    # --- Load meta ---
    meta_path = cfg.outputs_dir / f"meta_{args.model_name}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text())

    backbone = meta["backbone"]
    use_crops = bool(meta.get("use_crops", False))
    hidden_dim = int(meta.get("hidden_dim", 512))
    img_size = int(meta.get("img_size", 224))

    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    num_classes = len(idx_to_class)

    cfg = replace(cfg, use_crops=use_crops)

    print(f"Using model_name={args.model_name}")
    print(f"backbone={backbone} use_crops={use_crops} hidden_dim={hidden_dim} img_size={img_size} num_classes={num_classes}")
    print(f"device={device}")

    # --- Build models ---
    backbone_model, _ = build_backbone(backbone)
    backbone_model.to(device).eval()

    input_dim = 2048 if backbone == "resnet50" else 1280
    head = CustomHead(input_dim, hidden_dim, num_classes, dropout_prob=0.0)

    weights_path = cfg.outputs_dir / f"{args.model_name}.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    head.load_state_dict(torch.load(weights_path, map_location=device))
    head.to(device).eval()

    # --- Test set ---
    print(f"Loading test images from: {cfg.test_dir}")
    test_samples = load_test_recursive(cfg.test_dir)
    if len(test_samples) == 0:
        raise ValueError(f"No test images found in {cfg.test_dir}")

    test_samples = sorted(test_samples, key=lambda s: s.path.name.lower())

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = SampleDataset(test_samples, tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    all_files = []
    all_preds = []

    print("Running inference...")
    with torch.no_grad():
        for imgs, _, paths in tqdm(dl):
            imgs = imgs.to(device, non_blocking=True)
            feats = backbone_model(imgs)
            logits = head(feats)
            preds = logits.argmax(dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_files.extend([Path(p).name for p in paths])

    # --- Map to Kaggle ids ---
    kaggle_ids = []
    unmapped = set()

    for p in all_preds:
        cname = idx_to_class[int(p)]
        if cname in KAGGLE_NAME_TO_IDX:
            kaggle_ids.append(KAGGLE_NAME_TO_IDX[cname])
        else:
            kaggle_ids.append(-1)
            unmapped.add(cname)

    if unmapped:
        print(f"WARNING: unmapped class names (showing up to 10): {list(unmapped)[:10]}")

    out_path = cfg.outputs_dir / args.out
    pd.DataFrame({"path": all_files, "class_idx": kaggle_ids}).to_csv(out_path, index=False)
    print(f"Saved submission: {out_path}")

if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataclasses import replace
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
    parser.add_argument(
        "--model_name",
        required=True,
        help="Exact model_name used in training (the long one saved in meta_*.json).",
    )
    parser.add_argument("--out", default="submission_neural.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(cfg.device)

    # --- load metadata (single source of truth) ---
    meta_path = cfg.outputs_dir / f"meta_{args.model_name}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text())

    backbone = meta["backbone"]
    use_crops = bool(meta.get("use_crops", False))
    augment = bool(meta.get("augment", False))
    hidden_dim = int(meta.get("hidden_dim", 512))
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    num_classes = len(idx_to_class)

    # Make sure cfg uses crops consistently with training
    cfg = replace(cfg, use_crops=use_crops)

    print(
        f"Loading model_name={args.model_name}\n"
        f"  backbone={backbone} use_crops={use_crops} augment={augment}\n"
        f"  hidden_dim={hidden_dim} num_classes={num_classes} device={device}"
    )

    # --- build models ---
    backbone_model, _ = build_backbone(backbone)
    backbone_model.to(device).eval()

    input_dim = 2048 if backbone == "resnet50" else 1280
    head = CustomHead(input_dim, hidden_dim, num_classes, dropout_prob=0.0)  # dropout off at inference
    weights_path = cfg.outputs_dir / f"{args.model_name}.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    head.load_state_dict(torch.load(weights_path, map_location=device))
    head.to(device).eval()

    # --- test data ---
    print(f"Predicting on: {cfg.test_dir}")
    test_samples = load_test_recursive(cfg.test_dir)
    if len(test_samples) == 0:
        raise ValueError(f"No test images found in {cfg.test_dir}")
    test_samples = sorted(test_samples, key=lambda s: s.path.name.lower())

    # IMPORTANT: inference transform should be deterministic, not augmented
    img_size = int(meta.get("img_size", 224))
    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = SampleDataset(test_samples, val_tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers)

    all_preds = []
    all_files = []

    print("Running inference...")
    with torch.no_grad():
        for imgs, _, paths in tqdm(dl):
            imgs = imgs.to(device, non_blocking=True)
            feats = backbone_model(imgs)
            logits = head(feats)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_files.extend([Path(p).name for p in paths])

    # map internal class names -> kaggle indices
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
        print(f"WARNING: {len(unmapped)} classes unmapped to Kaggle IDs. Example: {list(unmapped)[:10]}")

    df = pd.DataFrame({"path": all_files, "class_idx": kaggle_ids})
    out_path = cfg.outputs_dir / args.out
    df.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()

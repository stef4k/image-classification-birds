import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from birds_ml.config import Config
from birds_ml.arcface_margin import ArcMarginProduct
from birds_ml.data import Sample, load_test_recursive
from birds_ml.embedder import build_backbone
from birds_ml.features import SampleDataset


AMERICAN_CROW_KAGGLE_IDX = 12
FISH_CROW_KAGGLE_IDX = 13
SEPARATOR_CLASS_TO_KAGGLE_IDX = {
    "American_Crow": AMERICAN_CROW_KAGGLE_IDX,
    "Fish_Crow": FISH_CROW_KAGGLE_IDX,
}


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
        return TF.pad(
            img,
            (
                delta_w // 2,
                delta_h // 2,
                delta_w - (delta_w // 2),
                delta_h - (delta_h // 2),
            ),
            fill=128,
            padding_mode="constant",
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_in", type=Path, required=True)
    parser.add_argument("--submission_out", type=Path, required=True)
    parser.add_argument("--separator_backbone", default="vit_so150m2_384")
    parser.add_argument("--separator_img_size", type=int, default=384)
    parser.add_argument("--separator_model_name", default=None)
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = replace(Config(), use_crops=args.use_crops)

    if not args.submission_in.exists():
        raise FileNotFoundError(f"Submission file not found: {args.submission_in}")

    crop_suffix = "_cropped" if cfg.use_crops else ""
    model_name = args.separator_model_name or f"crow_separator_{args.separator_backbone}{crop_suffix}_{args.separator_img_size}"
    ckpt_path = cfg.outputs_dir / f"{model_name}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Crow separator checkpoint not found: {ckpt_path}")

    device_name = cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    device = torch.device(device_name)

    checkpoint = torch.load(ckpt_path, map_location=device)
    idx_to_class = {
        int(k): v for k, v in checkpoint.get("idx_to_class", {"0": "American_Crow", "1": "Fish_Crow"}).items()
    }
    head_type = checkpoint.get("head_type", "linear")
    if head_type not in {"linear", "arcface"}:
        raise ValueError(f"Unsupported head_type in checkpoint: {head_type}")
    if set(idx_to_class.values()) != set(SEPARATOR_CLASS_TO_KAGGLE_IDX.keys()):
        raise ValueError(f"Unexpected separator classes in checkpoint: {idx_to_class}")

    backbone_model, _ = build_backbone(args.separator_backbone)
    backbone_model.load_state_dict(checkpoint["backbone"])
    backbone_model.to(device).eval()

    if head_type == "arcface":
        head = ArcMarginProduct(
            backbone_model.num_features,
            len(idx_to_class),
            s=float(checkpoint.get("arcface_s", 30.0)),
            m=float(checkpoint.get("arcface_m", 0.50)),
        ).to(device)
    else:
        head = nn.Linear(backbone_model.num_features, len(idx_to_class)).to(device)
    head.load_state_dict(checkpoint["head"])
    head.to(device).eval()
    print(f"Loaded separator head type: {head_type}")

    saved_config = checkpoint.get(
        "config",
        {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    )
    tfm = transforms.Compose(
        [
            SquarePad(args.separator_img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=saved_config["mean"], std=saved_config["std"]),
        ]
    )

    sub = pd.read_csv(args.submission_in)
    required_cols = {"path", "class_idx"}
    if not required_cols.issubset(sub.columns):
        raise ValueError(f"{args.submission_in} must contain columns: {sorted(required_cols)}")

    sub["class_idx"] = sub["class_idx"].astype(int)
    mask = sub["class_idx"].isin([AMERICAN_CROW_KAGGLE_IDX, FISH_CROW_KAGGLE_IDX])
    target_rows = sub[mask].copy()
    if target_rows.empty:
        sub.to_csv(args.submission_out, index=False)
        print("No American/Fish crow rows found in submission. File copied unchanged.")
        print(f"Saved: {args.submission_out}")
        return

    test_samples_all = load_test_recursive(cfg.test_dir)
    path_map = {}
    for s in test_samples_all:
        name = s.path.name
        if name in path_map:
            raise ValueError(f"Duplicate test filename found, cannot map uniquely: {name}")
        path_map[name] = s.path

    rerank_samples = []
    missing = []
    for name in target_rows["path"].astype(str).tolist():
        if name not in path_map:
            missing.append(name)
            continue
        rerank_samples.append(Sample(path=path_map[name], label=None))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} submission paths not found under {cfg.test_dir}. Examples: {preview}")

    dl = DataLoader(
        SampleDataset(rerank_samples, tfm),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    pred_by_name = {}

    def infer_logits(feats):
        if head_type == "arcface":
            norm_feats = F.normalize(feats)
            norm_weights = F.normalize(head.weight)
            return F.linear(norm_feats, norm_weights) * head.s
        return head(feats)

    with torch.no_grad():
        for imgs, _, paths in tqdm(dl, desc="Crow separator inference"):
            imgs = imgs.to(device)

            feats1 = backbone_model(imgs)
            logits1 = infer_logits(feats1)

            imgs_flip = torch.flip(imgs, [3])
            feats2 = backbone_model(imgs_flip)
            logits2 = infer_logits(feats2)

            avg_logits = (logits1 + logits2) / 2.0
            pred = torch.argmax(F.softmax(avg_logits, dim=1), dim=1).cpu().tolist()

            for i, p in enumerate(paths):
                filename = Path(p).name
                class_name = idx_to_class[int(pred[i])]
                pred_by_name[filename] = SEPARATOR_CLASS_TO_KAGGLE_IDX[class_name]

    before = sub.loc[mask, "class_idx"].astype(int)
    sub.loc[mask, "class_idx"] = sub.loc[mask, "path"].map(pred_by_name).astype(int)
    after = sub.loc[mask, "class_idx"].astype(int)

    changed = int((before.values != after.values).sum())
    total = int(mask.sum())
    print(f"Reranked rows: {total} | Changed by separator: {changed}")
    print(
        "Final crow counts | "
        f"American_Crow: {(sub['class_idx'] == AMERICAN_CROW_KAGGLE_IDX).sum()} | "
        f"Fish_Crow: {(sub['class_idx'] == FISH_CROW_KAGGLE_IDX).sum()}"
    )

    args.submission_out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.submission_out, index=False)
    print(f"Saved reranked submission: {args.submission_out}")


if __name__ == "__main__":
    main()

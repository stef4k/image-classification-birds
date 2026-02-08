import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from birds_ml.arcface_margin import ArcMarginProduct
from birds_ml.config import Config
from birds_ml.data import Sample, load_test_recursive


AMERICAN_CROW_KAGGLE_IDX = 12
FISH_CROW_KAGGLE_IDX = 13
SEPARATOR_CLASS_ORDER = ["American_Crow", "Fish_Crow"]
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


class SampleTensorDataset(Dataset):
    def __init__(self, samples: List[Sample], tfm):
        self.samples = samples
        self.tfm = tfm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self.tfm(s.path)
        return img, str(s.path)


def _pil_loader_transform(img_size: int):
    from PIL import Image

    pad = SquarePad(img_size)

    def _load(path: Path):
        img = Image.open(path).convert("RGB")
        img = pad(img)
        return transforms.functional.to_tensor(img)

    return _load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_in", type=Path, required=True)
    parser.add_argument("--submission_out", type=Path, required=True)
    parser.add_argument("--separator_backbone", default="eva02_large_448")
    parser.add_argument("--separator_img_size", type=int, default=448)
    parser.add_argument("--separator_model_names", nargs="+", required=True)
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cuda:0, cpu)")
    return parser.parse_args()


def _resolve_idx_to_class(checkpoint) -> Dict[int, str]:
    raw = checkpoint.get("idx_to_class", {"0": "American_Crow", "1": "Fish_Crow"})
    return {int(k): str(v) for k, v in raw.items()}


def _infer_logits(head, head_type: str, feats: torch.Tensor) -> torch.Tensor:
    if head_type == "arcface":
        norm_feats = F.normalize(feats)
        norm_weights = F.normalize(head.weight)
        return F.linear(norm_feats, norm_weights) * float(head.s)
    return head(feats)


def _load_separator_model(
    cfg: Config,
    args,
    model_name: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module, str, Dict[int, str], Dict[str, List[float]]]:
    from birds_ml.embedder import build_backbone

    ckpt_path = cfg.outputs_dir / f"{model_name}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Crow separator checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    idx_to_class = _resolve_idx_to_class(checkpoint)
    head_type = str(checkpoint.get("head_type", "linear"))
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

    saved_config = checkpoint.get(
        "config",
        {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    )
    if "mean" not in saved_config or "std" not in saved_config:
        raise ValueError(f"Checkpoint config missing mean/std: {ckpt_path}")

    return backbone_model, head, head_type, idx_to_class, saved_config


def main():
    args = parse_args()
    cfg = replace(Config(), use_crops=args.use_crops)

    if not args.submission_in.exists():
        raise FileNotFoundError(f"Submission file not found: {args.submission_in}")

    if len(args.separator_model_names) < 2:
        raise ValueError("--separator_model_names must include at least 2 models for ensembling")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device_name = cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
        device = torch.device(device_name)

    sub = pd.read_csv(args.submission_in)
    required_cols = {"path", "class_idx"}
    if not required_cols.issubset(sub.columns):
        raise ValueError(f"{args.submission_in} must contain columns: {sorted(required_cols)}")

    sub["class_idx"] = sub["class_idx"].astype(int)
    mask = sub["class_idx"].isin([AMERICAN_CROW_KAGGLE_IDX, FISH_CROW_KAGGLE_IDX])
    target_rows = sub[mask].copy()
    if target_rows.empty:
        args.submission_out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(args.submission_out, index=False)
        print("No American/Fish crow rows found in submission. File copied unchanged.")
        print(f"Saved: {args.submission_out}")
        return

    test_samples_all = load_test_recursive(cfg.test_dir)
    path_map: Dict[str, Path] = {}
    for s in test_samples_all:
        name = s.path.name
        if name in path_map:
            raise ValueError(f"Duplicate test filename found, cannot map uniquely: {name}")
        path_map[name] = s.path

    rerank_samples: List[Sample] = []
    missing = []
    for name in target_rows["path"].astype(str).tolist():
        if name not in path_map:
            missing.append(name)
            continue
        rerank_samples.append(Sample(path=path_map[name], label=None))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} submission paths not found under {cfg.test_dir}. Examples: {preview}")

    to_tensor = _pil_loader_transform(args.separator_img_size)
    dl = DataLoader(
        SampleTensorDataset(rerank_samples, to_tensor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    probs_sum_by_name: Dict[str, torch.Tensor] = {}
    for s in rerank_samples:
        probs_sum_by_name[s.path.name] = torch.zeros(2, dtype=torch.float32)

    for model_i, model_name in enumerate(args.separator_model_names, start=1):
        print(f"\n[{model_i}/{len(args.separator_model_names)}] Loading separator model: {model_name}")
        backbone_model, head, head_type, idx_to_class, saved_config = _load_separator_model(
            cfg=cfg,
            args=args,
            model_name=model_name,
            device=device,
        )
        print(f"Loaded separator head type: {head_type}")

        idx_by_class = {v: k for k, v in idx_to_class.items()}
        if set(idx_by_class.keys()) != set(SEPARATOR_CLASS_ORDER):
            raise ValueError(f"Separator class mapping unexpected: {idx_to_class}")

        mean = torch.tensor(saved_config["mean"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(saved_config["std"], device=device, dtype=torch.float32).view(1, 3, 1, 1)

        with torch.no_grad():
            for imgs, paths in tqdm(dl, desc=f"Inference ({model_name})"):
                imgs = imgs.to(device)
                imgs = (imgs - mean) / std

                feats1 = backbone_model(imgs)
                logits1 = _infer_logits(head, head_type, feats1)

                imgs_flip = torch.flip(imgs, [3])
                feats2 = backbone_model(imgs_flip)
                logits2 = _infer_logits(head, head_type, feats2)

                avg_logits = (logits1 + logits2) / 2.0
                probs = F.softmax(avg_logits, dim=1)

                probs_canon = torch.stack(
                    [
                        probs[:, idx_by_class["American_Crow"]],
                        probs[:, idx_by_class["Fish_Crow"]],
                    ],
                    dim=1,
                ).detach().cpu()

                for i, p in enumerate(paths):
                    filename = Path(p).name
                    if filename not in probs_sum_by_name:
                        raise KeyError(f"Internal error: unexpected filename in dataloader: {filename}")
                    probs_sum_by_name[filename] += probs_canon[i]

        del backbone_model
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pred_by_name: Dict[str, int] = {}
    denom = float(len(args.separator_model_names))
    for filename, psum in probs_sum_by_name.items():
        pavg = psum / denom
        pred_idx = int(torch.argmax(pavg).item())
        pred_class = SEPARATOR_CLASS_ORDER[pred_idx]
        pred_by_name[filename] = SEPARATOR_CLASS_TO_KAGGLE_IDX[pred_class]

    before = sub.loc[mask, "class_idx"].astype(int)
    sub.loc[mask, "class_idx"] = sub.loc[mask, "path"].map(pred_by_name).astype(int)
    after = sub.loc[mask, "class_idx"].astype(int)

    changed = int((before.values != after.values).sum())
    total = int(mask.sum())
    print(f"\nReranked rows: {total} | Changed by ensemble: {changed}")
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

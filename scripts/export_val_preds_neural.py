import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from birds_ml.config import Config
from birds_ml.data import load_val_with_given_mapping
from birds_ml.features import extract_embeddings
from birds_ml.head import CustomHead
from birds_ml.utils import ensure_dir


def find_best_neural_meta(outputs_dir: Path) -> Path:
    metas = sorted(outputs_dir.glob("meta_neural_*.json"))
    if not metas:
        raise FileNotFoundError(f"No meta_neural_*.json found in {outputs_dir}")

    best_path = None
    best_acc = -1.0
    for mp in metas:
        m = json.loads(mp.read_text())
        acc = float(m.get("best_acc", -1.0))
        if acc > best_acc:
            best_acc = acc
            best_path = mp
    return best_path


def main():
    cfg = Config()
    ensure_dir(cfg.outputs_dir)
    ensure_dir(cfg.cache_dir)

    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=None, help="meta_neural_*.json path (relative to outputs/ or absolute). If omitted, picks best_acc")
    ap.add_argument("--weights", default=None, help="Path to .pth weights (absolute or relative). If omitted, uses outputs/{model_name}.pth")
    ap.add_argument("--use_crops", action="store_true")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default=None, help="Override cfg.device (cpu or cuda)")
    ap.add_argument("--out", default=None, help="Output CSV path (default: outputs/val_preds_<model_name>.csv)")
    args = ap.parse_args()

    cfg = replace(cfg, use_crops=args.use_crops)
    if args.device is not None:
        cfg = replace(cfg, device=args.device)

    crop_suffix = "_cropped" if cfg.use_crops else ""

    # ---- pick meta ----
    if args.meta:
        meta_path = Path(args.meta)
        if not meta_path.is_absolute():
            meta_path = cfg.outputs_dir / meta_path
    else:
        meta_path = find_best_neural_meta(cfg.outputs_dir)

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    class_to_idx = meta["class_to_idx"]
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    backbone = meta.get("backbone", "efficientnet_b0")
    hidden_dim = int(meta.get("hidden_dim", 512))
    dropout = float(meta.get("dropout", 0.5))

    model_name = meta.get("model_name", None)
    if model_name is None:
        model_name = meta_path.stem.replace("meta_", "")  # fallback

    # ---- weights ----
    if args.weights:
        head_path = Path(args.weights)
        if not head_path.is_absolute():
            head_path = Path.cwd() / head_path
    else:
        head_path = cfg.outputs_dir / f"{model_name}.pth"

    if not head_path.exists():
        raise FileNotFoundError(f"Head weights not found: {head_path}")

    # ---- val samples ----
    val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)

    # ---- embeddings ----
    cache_path = cfg.cache_dir / f"emb_{backbone}_val{crop_suffix}.npz"
    if cache_path.exists() and not args.no_cache:
        z = np.load(cache_path, allow_pickle=True)
        Xv, yv = z["X"], z["y"].astype(int)
        print(f"Loaded cache: {cache_path} X={Xv.shape}")
    else:
        Xv, yv, _ = extract_embeddings(
            val_samples, backbone, cfg.batch_size, cfg.num_workers, cfg.device
        )
        yv = yv.astype(int)
        np.savez_compressed(cache_path, X=Xv, y=yv)
        print(f"Saved cache: {cache_path} X={Xv.shape}")

    # ---- head + predict ----
    device = torch.device(cfg.device)
    input_dim = 2048 if backbone == "resnet50" else 1280

    head = CustomHead(input_dim, hidden_dim, len(class_to_idx), dropout_prob=dropout).to(device)
    head.load_state_dict(torch.load(head_path, map_location=device))
    head.eval()

    with torch.no_grad():
        feats = torch.from_numpy(Xv).float().to(device)
        logits = head(feats)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    pred_idx = probs.argmax(axis=1).astype(int)
    conf = probs.max(axis=1).astype(float)
    hard = (1.0 - conf).astype(float)

    k = max(1, int(args.topk))
    topk_idx = np.argsort(-probs, axis=1)[:, :k]
    topk_names = [[idx_to_class[int(i)] for i in row] for row in topk_idx]

    pred_name = [idx_to_class[int(i)] for i in pred_idx]
    true_name = [idx_to_class[int(i)] for i in yv]
    correct = (pred_idx == yv)

    # ---- output CSV ----
    if args.out:
        out_csv = Path(args.out)
        if not out_csv.is_absolute():
            out_csv = Path.cwd() / out_csv
    else:
        out_csv = cfg.outputs_dir / f"val_preds_{model_name}{crop_suffix}.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "true", "pred", "correct", "confidence", "hardness", "topk"])
        for i, s in enumerate(val_samples):
            w.writerow([
                str(s.path),
                true_name[i],
                pred_name[i],
                int(correct[i]),
                float(conf[i]),
                float(hard[i]),
                "|".join(topk_names[i]),
            ])

    print(f"\nSaved predictions CSV:\n  {out_csv}")
    print(f"Meta used: {meta_path.name}")
    print(f"Weights used: {head_path}")


if __name__ == "__main__":
    main()

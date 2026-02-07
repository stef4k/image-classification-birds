import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import replace

from birds_ml.config import Config
from birds_ml.data import load_test_recursive
from birds_ml.features import extract_embeddings
from birds_ml.model import load_model
from birds_ml.utils import ensure_dir

KAGGLE_NAME_TO_IDX = {
    "Groove_billed_Ani": 0,
    "Red_winged_Blackbird": 1,
    "Rusty_Blackbird": 2,
    "Gray_Catbird": 3,
    "Brandt_Cormorant": 4,
    "Eastern_Towhee": 5,
    "Indigo_Bunting": 6,
    "Brewer_Blackbird": 7,
    "Painted_Bunting": 8,
    "Bobolink": 9,
    "Lazuli_Bunting": 10,
    "Yellow_headed_Blackbird": 11,
    "American_Crow": 12,
    "Fish_Crow": 13,
    "Brown_Creeper": 14,
    "Yellow_billed_Cuckoo": 15,
    "Yellow_breasted_Chat": 16,
    "Black_billed_Cuckoo": 17,
    "Gray_crowned_Rosy_Finch": 18,
    "Bronzed_Cowbird": 19,
}

def main():
    cfg = Config()
    ensure_dir(cfg.outputs_dir)
    ensure_dir(cfg.cache_dir)

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["svm", "logreg"], default=None)
    ap.add_argument("--backbone", choices=["resnet50", "efficientnet_b0"], default=None)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--out", type=str, default="submission.csv")
    ap.add_argument("--use_crops", action="store_true", help="Use cropped datasets")
    args = ap.parse_args()
    cfg = replace(cfg, use_crops=args.use_crops)

    crop_suffix = "_cropped" if cfg.use_crops else ""

    # Pick meta: specific run if provided, else latest
    if args.kind and args.backbone:
        meta_filename = f"meta_{args.kind}_{args.backbone}{crop_suffix}.json"
        meta_path = cfg.outputs_dir / meta_filename
    else:
        meta_path = cfg.outputs_dir / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    kind = meta["kind"]
    backbone = meta["backbone"]
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}

    model_name = f"{kind}_{backbone}{crop_suffix}"
    model_path = cfg.outputs_dir / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(str(model_path))

    # Load test images
    print(f"Predicting on: {cfg.test_dir}")
    test_samples = load_test_recursive(cfg.test_dir)
    if len(test_samples) == 0:
        raise ValueError(f"No test images found under: {cfg.test_dir}")

    test_samples = sorted(test_samples, key=lambda s: s.path.name.lower())

    # Load/extract embeddings for the SAME backbone as the model
    cache_path = cfg.cache_dir / f"emb_{backbone}_test{crop_suffix}.npz"
    if cache_path.exists() and not args.no_cache:
        z = np.load(cache_path, allow_pickle=True)
        Xt = z["X"]
        files = z["files"].tolist()
        print(f"Loaded cache: {cache_path}  X={Xt.shape}")
    else:
        Xt, _, paths = extract_embeddings(
            test_samples, backbone, cfg.batch_size, cfg.num_workers, cfg.device
        )
        files = [Path(p).name for p in paths]
        np.savez_compressed(cache_path, X=Xt, files=np.array(files, dtype=object))
        print(f"Saved cache: {cache_path}  X={Xt.shape}")

    # Predict
    pred_model_idx = model.predict(Xt).astype(int)

    # Map -> Kaggle integer class_idx
    pred_class_idx = []
    unmapped = []
    for i in pred_model_idx:
        class_name = idx_to_class[i]
        if class_name not in KAGGLE_NAME_TO_IDX:
            unmapped.append(class_name)
            pred_class_idx.append(-1)
        else:
            pred_class_idx.append(KAGGLE_NAME_TO_IDX[class_name])

    if unmapped:
        unmapped = sorted(set(unmapped))
        raise ValueError(
            "Some predicted class names are not in the provided mapping. "
            f"Unmapped examples: {unmapped[:10]}"
        )

    out_path = cfg.outputs_dir / args.out
    df = pd.DataFrame({"path": files, "class_idx": pred_class_idx})
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  rows={len(df)}  kind={kind} backbone={backbone}")

if __name__ == "__main__":
    main()

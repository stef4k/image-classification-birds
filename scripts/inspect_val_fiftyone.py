import argparse
from dataclasses import replace
import json
import numpy as np
import fiftyone as fo

from birds_ml.config import Config
from birds_ml.data import load_val_with_given_mapping
from birds_ml.features import extract_embeddings
from birds_ml.model import load_model
from birds_ml.utils import ensure_dir


def main():
    cfg = Config()
    ensure_dir(cfg.outputs_dir)
    ensure_dir(cfg.cache_dir)

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["svm", "logreg"], default=None)
    ap.add_argument("--backbone", choices=["resnet50", "efficientnet_b0"], default=None)
    ap.add_argument("--no_cache", action="store_true")
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
    class_to_idx = meta["class_to_idx"]
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    kind = meta["kind"]
    backbone = meta["backbone"]

    # Load the matching trained model
    model_name = f"{kind}_{backbone}{crop_suffix}"
    model_path = cfg.outputs_dir / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load_model(str(model_path))

    # Load val samples with consistent mapping
    val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)

    # Load/extract embeddings for the SAME backbone as the model
    cache_path = cfg.cache_dir / f"emb_{backbone}_val{crop_suffix}.npz"
    if cache_path.exists() and not args.no_cache:
        z = np.load(cache_path, allow_pickle=True)
        Xv, yv = z["X"], z["y"]
        print(f"Loaded cache: {cache_path}  X={Xv.shape}")
    else:
        Xv, yv, _ = extract_embeddings(
            val_samples, backbone, cfg.batch_size, cfg.num_workers, cfg.device
        )
        np.savez_compressed(cache_path, X=Xv, y=yv)
        print(f"Saved cache: {cache_path}  X={Xv.shape}")

    # Predict labels
    pred_idx = model.predict(Xv).astype(int)
    pred_name = [idx_to_class[i] for i in pred_idx]
    true_name = [idx_to_class[int(y)] for y in yv]
    correct = [p == t for p, t in zip(pred_name, true_name)]

    # Confidence + hardness = 1 - confidence (only if predict_proba exists, e.g. LogReg)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(Xv)
        conf = probs.max(axis=1).astype(float)
        hard = (1.0 - conf).astype(float)
    else:
        conf = [None] * len(val_samples)
        hard = [None] * len(val_samples)

    # Build FiftyOne dataset
    ds_name = f"birds_val_{kind}_{backbone}{crop_suffix}"
    if fo.dataset_exists(ds_name):
        fo.delete_dataset(ds_name)
    dataset = fo.Dataset(ds_name)

    for i, s in enumerate(val_samples):
        sample = fo.Sample(filepath=str(s.path))
        sample["ground_truth"] = fo.Classification(label=true_name[i])
        sample["prediction"] = fo.Classification(label=pred_name[i])
        sample["correct"] = bool(correct[i])
        if conf[i] is not None:
            sample["confidence"] = float(conf[i])
            sample["hardness"] = float(hard[i])
        dataset.add_sample(sample)

    dataset.persistent = True

    print(f"Dataset created: {dataset.name} with {len(dataset)} samples")
    print("In the App: filter `correct == false` and sort by `hardness` desc (if available).")
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()

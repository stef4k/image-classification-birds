import argparse
from dataclasses import replace
import json
import os
import numpy as np

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
    ap.add_argument(
        "--use_crops",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use cropped validation images (default: true if available)",
    )
    args = ap.parse_args()

    # Prefer cropped validation images by default when available, since they're
    # easier to visually inspect in FiftyOne
    if args.use_crops is None:
        use_crops = (cfg.data_dir / "val_images_cropped").exists()
    else:
        use_crops = bool(args.use_crops)

    cfg = replace(cfg, use_crops=use_crops)
    data_crop_suffix = "_cropped" if cfg.use_crops else ""

    # FiftyOne uses an embedded MongoDB by default. On Windows, the default
    # global DB log rotation can fail if another process holds the log file,
    # which then manifests as a broken App UI. Use a project-local DB dir by
    # default so this script is self-contained and avoids global lock issues.
    if "FIFTYONE_DATABASE_URI" not in os.environ and "FIFTYONE_DATABASE_DIR" not in os.environ:
        fo_db_dir = cfg.outputs_dir / "fiftyone_db"
        ensure_dir(fo_db_dir)
        os.environ["FIFTYONE_DATABASE_DIR"] = str(fo_db_dir)

    import fiftyone as fo

    # Pick meta: specific run if provided, else latest
    if args.kind and args.backbone:
        # Try to keep meta+model consistent. Prefer the suffix that matches the
        # requested (or default) validation images, but fall back if needed.
        candidates = [
            cfg.outputs_dir / f"meta_{args.kind}_{args.backbone}{data_crop_suffix}.json",
            cfg.outputs_dir / f"meta_{args.kind}_{args.backbone}.json",
            cfg.outputs_dir / f"meta_{args.kind}_{args.backbone}_cropped.json",
        ]
        meta_path = next((p for p in candidates if p.exists()), candidates[0])
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
    model_candidates = [
        cfg.outputs_dir / f"{kind}_{backbone}{data_crop_suffix}.joblib",
        cfg.outputs_dir / f"{kind}_{backbone}.joblib",
        cfg.outputs_dir / f"{kind}_{backbone}_cropped.joblib",
    ]
    model_path = next((p for p in model_candidates if p.exists()), model_candidates[0])
    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Tried: " + ", ".join(str(p) for p in model_candidates)
        )
    model = load_model(str(model_path))

    # Load val samples with consistent mapping
    val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)

    # Load/extract embeddings for the SAME backbone as the model
    cache_path = cfg.cache_dir / f"emb_{backbone}_val{data_crop_suffix}.npz"
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

    # Confidence + hardness = 1 - confidence
    #
    # - LogReg exposes calibrated probabilities via `predict_proba`
    # - Many SVMs expose margins via `decision_function`
    # - Fallback: populate fields so they appear in the App
    n = len(val_samples)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(Xv)
        conf = probs.max(axis=1).astype(float)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(Xv)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            # Binary: map absolute margin -> (0.5, 1.0) via sigmoid
            conf = (1.0 / (1.0 + np.exp(-np.abs(scores)))).astype(float)
        else:
            # Multiclass: softmax(scores) and take max prob (not calibrated)
            scores = scores - scores.max(axis=1, keepdims=True)
            exps = np.exp(scores)
            probs = exps / exps.sum(axis=1, keepdims=True)
            conf = probs.max(axis=1).astype(float)
    else:
        conf = np.zeros(n, dtype=float)

    hard = (1.0 - conf).astype(float)

    # Build FiftyOne dataset
    ds_name = f"birds_val_{kind}_{backbone}{data_crop_suffix}"
    if fo.dataset_exists(ds_name):
        fo.delete_dataset(ds_name)
    dataset = fo.Dataset(ds_name)

    def _safe_float(x):
        try:
            x = float(x)
        except Exception:
            return None
        return None if np.isnan(x) else x

    for i, s in enumerate(val_samples):
        sample = fo.Sample(filepath=str(s.path))
        sample["ground_truth"] = fo.Classification(label=true_name[i])
        pred_conf = _safe_float(conf[i])
        sample["prediction"] = fo.Classification(label=pred_name[i], confidence=pred_conf)
        sample["correct"] = bool(correct[i])
        sample["confidence"] = _safe_float(conf[i])
        sample["hardness"] = _safe_float(hard[i])
        dataset.add_sample(sample)

    dataset.persistent = True

    print(f"Dataset created: {dataset.name} with {len(dataset)} samples")
    print("In the App: filter `correct == false` and sort by `hardness` desc (if available).")
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()

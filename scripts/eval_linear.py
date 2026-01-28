import argparse
import json
import numpy as np

from birds_ml.config import Config
from birds_ml.data import load_val_with_given_mapping
from birds_ml.features import extract_embeddings
from birds_ml.model import load_model
from birds_ml.metrics import compute_metrics, report


def main():
    cfg = Config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["svm", "logreg"], default=None)
    ap.add_argument("--backbone", choices=["resnet50", "efficientnet_b0"], default=None)
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    # Choose meta file: specific run if provided, else latest
    if args.kind and args.backbone:
        meta_path = cfg.outputs_dir / f"meta_{args.kind}_{args.backbone}.json"
    else:
        meta_path = cfg.outputs_dir / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    class_to_idx = meta["class_to_idx"]
    backbone = meta["backbone"]
    kind = meta["kind"]

    model_path = cfg.outputs_dir / f"{kind}_{backbone}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(str(model_path))

    val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)

    cache_path = cfg.cache_dir / f"emb_{backbone}_val.npz"
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

    pred = model.predict(Xv)
    m = compute_metrics(yv, pred)
    print(f"\n[VAL] kind={kind} backbone={backbone} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}\n")
    print(report(yv, pred))


if __name__ == "__main__":
    main()

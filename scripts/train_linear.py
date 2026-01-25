import argparse
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

from birds_ml.config import Config
from birds_ml.utils import set_seed, ensure_dir
from birds_ml.data import load_trainval_from_folders
from birds_ml.features import extract_embeddings
from birds_ml.model import build_model, save_model, LinearCfg
from birds_ml.metrics import compute_metrics


def main():
    cfg = Config()
    set_seed(cfg.seed)
    ensure_dir(cfg.outputs_dir)
    ensure_dir(cfg.cache_dir)

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["svm", "logreg"], default="svm")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--backbone", choices=["resnet50", "efficientnet_b0"], default=cfg.backbone)
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    # override backbone from CLI
    backbone = args.backbone

    train_samples, class_to_idx = load_trainval_from_folders(cfg.train_dir)

    cache_path = cfg.cache_dir / f"emb_{backbone}_train.npz"
    if cache_path.exists() and not args.no_cache:
        z = np.load(cache_path, allow_pickle=True)
        X, y = z["X"], z["y"]
        print(f"Loaded cache: {cache_path}  X={X.shape}")
    else:
        X, y, _ = extract_embeddings(
            train_samples, backbone, cfg.batch_size, cfg.num_workers, cfg.device
        )
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"Saved cache: {cache_path}  X={X.shape}")

    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    fold_metrics = []
    for i, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = build_model(LinearCfg(kind=args.kind, C=args.C))
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        m = compute_metrics(y[te], pred)
        fold_metrics.append(m)
        print(f"[Fold {i}] acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    std = {k: float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    print(f"\n[CV avg] acc={avg['accuracy']:.4f} macro_f1={avg['macro_f1']:.4f}")
    print(f"[CV std] acc={std['accuracy']:.4f} macro_f1={std['macro_f1']:.4f}")

    final = build_model(LinearCfg(kind=args.kind, C=args.C))
    final.fit(X, y)

    model_path = cfg.outputs_dir / f"{args.kind}_{backbone}.joblib"
    save_model(final, str(model_path))

    meta = {
        "kind": args.kind,
        "C": args.C,
        "backbone": backbone,
        "cv_folds": cfg.cv_folds,
        "cv_avg": avg,
        "cv_std": std,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
    }

    # save run-specific meta (doesn't overwrite other runs)
    meta_run_path = cfg.outputs_dir / f"meta_{args.kind}_{backbone}.json"
    meta_run_path.write_text(json.dumps(meta, indent=2))

    # also keep a "latest" meta.json for scripts that assume it
    (cfg.outputs_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved model: {model_path}")
    print(f"Saved meta:  {meta_run_path}")
    print(f"Updated latest meta: {cfg.outputs_dir / 'meta.json'}")


if __name__ == "__main__":
    main()

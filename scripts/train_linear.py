import argparse
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from dataclasses import replace

import wandb

from birds_ml.config import Config
from birds_ml.utils import set_seed, ensure_dir
from birds_ml.data import load_trainval_from_folders
from birds_ml.features import extract_embeddings, build_embedding_transforms
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
    ap.add_argument("--use_crops", action="store_true", help="Use cropped datasets")
    ap.add_argument("--augment", action="store_true", help="Enable augmentation pipeline")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--resize_train", type=int, default=256)
    ap.add_argument("--crop_scale_min", type=float, default=0.7)
    ap.add_argument("--crop_scale_max", type=float, default=1.0)
    ap.add_argument("--rotation_deg", type=float, default=30.0)
    ap.add_argument("--hflip_p", type=float, default=0.5)
    ap.add_argument("--svm_loss", choices=["hinge", "squared_hinge"], default="squared_hinge")
    ap.add_argument("--svm_dual", type=int, choices=[0, 1], default=1)
    ap.add_argument("--svm_max_iter", type=int, default=5000)
    ap.add_argument("--svm_tol", type=float, default=1e-4)
    ap.add_argument("--logreg_penalty", choices=["l2", "l1", "elasticnet"], default="l2")
    ap.add_argument("--logreg_l1_ratio", type=float, default=0.5)
    ap.add_argument("--logreg_solver", choices=["saga", "lbfgs", "sag", "newton-cg"], default="saga")
    ap.add_argument("--logreg_max_iter", type=int, default=1000)
    # W&B toggles (optional)
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", default=None, help="Override WANDB_PROJECT")
    ap.add_argument("--wandb_entity", default=None, help="Override WANDB_ENTITY")
    ap.add_argument("--wandb_group", default=None, help="Override WANDB_GROUP")
    ap.add_argument("--wandb_name", default=None, help="Override WANDB_NAME")
    ap.add_argument("--wandb_mode", default=None, choices=["online", "offline", "disabled"],
                    help="Override WANDB_MODE (online/offline/disabled)")
    args = ap.parse_args()
    cfg = replace(cfg, use_crops=args.use_crops)

    # -----------------------
    # W&B init (if enabled)
    # -----------------------
    use_wandb = bool(args.wandb) and (args.wandb_mode != "disabled") and (os.environ.get("WANDB_MODE") != "disabled")
    run = None
    if use_wandb:
        wandb_mode = (
            args.wandb_mode
            or os.environ.get("WANDB_MODE", "online")
        )
        wandb_project = (
            args.wandb_project
            or os.environ.get("WANDB_PROJECT", "birds-linear")
        )
        wandb_entity = (
            args.wandb_entity
            or os.environ.get("WANDB_ENTITY", None)
        )
        wandb_group = (
            args.wandb_group
            or os.environ.get("WANDB_GROUP", None)
        )
        wandb_name = (
            args.wandb_name
            or os.environ.get("WANDB_NAME", None)
        )

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            name=wandb_name,
            config=vars(args),
            mode=wandb_mode,   # online | offline
            reinit=True,
        )

    print(f"Config: use_crops={cfg.use_crops}")
    print(f"Training Data: {cfg.train_dir}")

    # override backbone from CLI
    backbone = args.backbone

    train_samples, class_to_idx = load_trainval_from_folders(cfg.train_dir)

    crop_suffix = "_cropped" if cfg.use_crops else ""
    aug_suffix = "_aug" if args.augment else "_noaug"
    tfm_suffix = (
        f"_img{args.img_size}"
        f"_rs{args.resize_train}"
        f"_cs{args.crop_scale_min:g}-{args.crop_scale_max:g}"
        f"_rot{args.rotation_deg:g}"
        f"_hf{args.hflip_p:g}"
    )
    cache_path = cfg.cache_dir / f"emb_{backbone}_train{crop_suffix}{aug_suffix}{tfm_suffix}.npz"

    if cache_path.exists() and not args.no_cache:
        z = np.load(cache_path, allow_pickle=True)
        X, y = z["X"], z["y"]
        print(f"Loaded cache: {cache_path}  X={X.shape}")
    else:
        standard_tfm, augment_tfm = build_embedding_transforms(
            img_size=args.img_size,
            resize_train=args.resize_train,
            crop_scale_min=args.crop_scale_min,
            crop_scale_max=args.crop_scale_max,
            rotation_deg=args.rotation_deg,
            hflip_p=args.hflip_p,
        )
        train_tfm = augment_tfm if args.augment else standard_tfm
        X, y, _ = extract_embeddings(
            train_samples, backbone, cfg.batch_size, cfg.num_workers, cfg.device, transform=train_tfm
        )
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"Saved cache: {cache_path}  X={X.shape}")

    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    fold_metrics = []
    for i, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = build_model(LinearCfg(
            kind=args.kind,
            C=args.C,
            svm_loss=args.svm_loss,
            svm_dual=bool(args.svm_dual),
            svm_max_iter=args.svm_max_iter,
            svm_tol=args.svm_tol,
            logreg_penalty=args.logreg_penalty,
            logreg_l1_ratio=args.logreg_l1_ratio,
            logreg_max_iter=args.logreg_max_iter,
            logreg_solver=args.logreg_solver,
        ))
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        m = compute_metrics(y[te], pred)
        fold_metrics.append(m)
        print(f"[Fold {i}] acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    std = {k: float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    print(f"\n[CV avg] acc={avg['accuracy']:.4f} macro_f1={avg['macro_f1']:.4f}")
    print(f"[CV std] acc={std['accuracy']:.4f} macro_f1={std['macro_f1']:.4f}")
    if run is not None:
        wandb.log(
            {
                "cv_avg_accuracy": avg["accuracy"],
                "cv_avg_macro_f1": avg["macro_f1"],
                "cv_std_accuracy": std["accuracy"],
                "cv_std_macro_f1": std["macro_f1"],
            }
        )

    final = build_model(LinearCfg(
        kind=args.kind,
        C=args.C,
        svm_loss=args.svm_loss,
        svm_dual=bool(args.svm_dual),
        svm_max_iter=args.svm_max_iter,
        svm_tol=args.svm_tol,
        logreg_penalty=args.logreg_penalty,
        logreg_l1_ratio=args.logreg_l1_ratio,
        logreg_max_iter=args.logreg_max_iter,
        logreg_solver=args.logreg_solver,
    ))
    final.fit(X, y)

    aug_suffix = "_aug" if args.augment else ""
    if args.kind == "svm":
        model_name = (
            f"{args.kind}_{backbone}{crop_suffix}{aug_suffix}"
            f"_C{args.C:g}_loss{args.svm_loss}_dual{int(args.svm_dual)}_tol{args.svm_tol:g}"
        )
    else:
        model_name = (
            f"{args.kind}_{backbone}{crop_suffix}{aug_suffix}"
            f"_C{args.C:g}_pen{args.logreg_penalty}_solv{args.logreg_solver}_l1r{args.logreg_l1_ratio:g}"
        )
    model_path = cfg.outputs_dir / f"{model_name}.joblib"
    save_model(final, str(model_path))

    meta = {
        "kind": args.kind,
        "C": args.C,
        "svm_loss": args.svm_loss,
        "svm_dual": bool(args.svm_dual),
        "svm_max_iter": args.svm_max_iter,
        "svm_tol": args.svm_tol,
        "logreg_penalty": args.logreg_penalty,
        "logreg_l1_ratio": args.logreg_l1_ratio,
        "logreg_max_iter": args.logreg_max_iter,
        "logreg_solver": args.logreg_solver,
        "backbone": backbone,
        "use_crops": cfg.use_crops,
        "augment": args.augment,
        "img_size": args.img_size,
        "resize_train": args.resize_train,
        "crop_scale_min": args.crop_scale_min,
        "crop_scale_max": args.crop_scale_max,
        "rotation_deg": args.rotation_deg,
        "hflip_p": args.hflip_p,
        "cv_folds": cfg.cv_folds,
        "cv_avg": avg,
        "cv_std": std,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
        "model_name": model_name,
    }

    # save run-specific meta (doesn't overwrite other runs)
    meta_run_path = cfg.outputs_dir / f"meta_{model_name}.json"
    meta_run_path.write_text(json.dumps(meta, indent=2))

    # also keep a "latest" meta.json for scripts that assume it
    (cfg.outputs_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    # keep a stable name per kind/backbone/crops as a moving pointer
    meta_latest_path = cfg.outputs_dir / f"meta_{args.kind}_{backbone}{crop_suffix}.json"
    meta_latest_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved model: {model_path}")
    print(f"Saved meta:  {meta_run_path}")
    print(f"Updated latest meta: {cfg.outputs_dir / 'meta.json'}")
    print(f"Updated kind/backbone meta: {meta_latest_path}")
    if run is not None:
        wandb.summary["cv_avg_accuracy"] = avg["accuracy"]
        wandb.summary["cv_avg_macro_f1"] = avg["macro_f1"]
        wandb.summary["cv_std_accuracy"] = std["accuracy"]
        wandb.summary["cv_std_macro_f1"] = std["macro_f1"]
        wandb.finish()


if __name__ == "__main__":
    main()

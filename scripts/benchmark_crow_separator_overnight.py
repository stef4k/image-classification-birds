import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="eva02_large_448")
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--use_crops", action="store_true")
    ap.add_argument("--val_dir", type=Path, default=Path("data/f_images_cropped"))
    ap.add_argument("--secondary_val_dir", type=Path, default=Path("data/val_images_cropped"))
    ap.add_argument("--pseudo_dir", type=Path, default=None)
    ap.add_argument(
        "--pseudo_conf_csv",
        type=Path,
        default=Path("outputs/extra_images_cv_ensemble_predictions_with_conf.csv"),
    )
    ap.add_argument(
        "--results_csv",
        type=Path,
        default=Path("outputs/crow_separator_overnight_benchmark.csv"),
    )
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument(
        "--pseudo_profile_weight",
        type=float,
        default=0.2,
        help="Pseudo-label weight used by the pseudo profile.",
    )
    ap.add_argument(
        "--pseudo_profile_min_conf",
        type=float,
        default=0.90,
        help="Pseudo-label minimum confidence used by the pseudo profile.",
    )
    ap.add_argument(
        "--pseudo_profile_name",
        default="pseudo_low",
        help="Name of the pseudo profile in the benchmark output.",
    )
    ap.add_argument(
        "--profile_mode",
        choices=["both", "pseudo_only", "train_only"],
        default="both",
        help="Which pseudo profile set to run.",
    )
    ap.add_argument(
        "--pseudo_ignore_conf_filter",
        action="store_true",
        help="Use pseudo labels from folder names directly and ignore confidence CSV filtering.",
    )
    return ap.parse_args()


def run_cmd(cmd: List[str]):
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pseudo_profiles(args) -> List[Dict]:
    train_only = {
        "name": "train_only",
        "pseudo_weight": 0.0,
        "pseudo_min_conf": 1.0,
    }
    pseudo = {
        "name": args.pseudo_profile_name,
        "pseudo_weight": args.pseudo_profile_weight,
        "pseudo_min_conf": args.pseudo_profile_min_conf,
    }
    if args.profile_mode == "train_only":
        return [train_only]
    if args.profile_mode == "pseudo_only":
        return [pseudo]
    return [train_only, pseudo]


def selection_metrics() -> List[str]:
    return ["balanced_acc", "acc"]


def neural_configs() -> List[Dict]:
    return [
        {
            "name": "linear_frozen_light",
            "head_type": "linear",
            "finetune_backbone": False,
            "loss_type": "ce",
            "aug_mode": "light",
            "label_smoothing": 0.0,
            "val_tta_hflip": False,
        },
        {
            "name": "linear_finetune_focal_strong_tta",
            "head_type": "linear",
            "finetune_backbone": True,
            "loss_type": "focal",
            "focal_gamma": 2.0,
            "aug_mode": "strong",
            "label_smoothing": 0.0,
            "val_tta_hflip": True,
            "lr_backbone": 1e-5,
        },
        {
            "name": "arcface_frozen_light",
            "head_type": "arcface",
            "arcface_m": 0.50,
            "arcface_s": 30.0,
            "finetune_backbone": False,
            "loss_type": "ce",
            "aug_mode": "light",
            "label_smoothing": 0.0,
            "val_tta_hflip": False,
        },
        {
            "name": "arcface_finetune_light_tta",
            "head_type": "arcface",
            "arcface_m": 0.35,
            "arcface_s": 45.0,
            "finetune_backbone": True,
            "loss_type": "ce",
            "aug_mode": "light",
            "label_smoothing": 0.02,
            "val_tta_hflip": True,
            "lr_backbone": 1e-5,
        },
    ]


def embedding_configs() -> List[Dict]:
    return [
        {
            "name": "knn_cos_k5",
            "method": "knn",
            "knn_k": 5,
            "knn_metric": "cosine",
            "knn_weights": "distance",
            "knn_weight_mode": "replicate",
            "knn_rep_scale": 4.0,
            "pca_dim": 0,
        },
        {
            "name": "knn_cos_k11_pca64",
            "method": "knn",
            "knn_k": 11,
            "knn_metric": "cosine",
            "knn_weights": "distance",
            "knn_weight_mode": "replicate",
            "knn_rep_scale": 4.0,
            "pca_dim": 64,
        },
        {
            "name": "rbf_svm_c3_scale",
            "method": "rbf_svm",
            "svm_c": 3.0,
            "svm_gamma": "scale",
            "pca_dim": 64,
        },
        {
            "name": "logreg_c2_pca64",
            "method": "logreg",
            "logreg_c": 2.0,
            "pca_dim": 64,
        },
    ]


def run_neural(cfg, args, profile, sel_metric):
    run_name = (
        f"overnight_neural_{cfg['name']}_{args.backbone}_{args.img_size}_"
        f"{profile['name']}_{sel_metric}"
    )
    cmd = [
        sys.executable,
        "scripts/train_crow_separator.py",
        "--backbone",
        args.backbone,
        "--img_size",
        str(args.img_size),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--head_type",
        cfg["head_type"],
        "--loss_type",
        cfg["loss_type"],
        "--aug_mode",
        cfg["aug_mode"],
        "--label_smoothing",
        str(cfg.get("label_smoothing", 0.0)),
        "--lr_head",
        str(cfg.get("lr_head", 1e-3)),
        "--lr_backbone",
        str(cfg.get("lr_backbone", 2e-5)),
        "--real_weight",
        "1.0",
        "--pseudo_weight",
        str(profile["pseudo_weight"]),
        "--pseudo_min_conf",
        str(profile["pseudo_min_conf"]),
        "--val_dir",
        str(args.val_dir),
        "--secondary_val_dir",
        str(args.secondary_val_dir),
        "--selection_metric",
        sel_metric,
        "--run_name",
        run_name,
    ]
    if args.use_crops:
        cmd.append("--use_crops")
    if cfg.get("finetune_backbone", False):
        cmd.append("--finetune_backbone")
    if cfg.get("head_type") == "arcface":
        cmd.extend(["--arcface_m", str(cfg.get("arcface_m", 0.5))])
        cmd.extend(["--arcface_s", str(cfg.get("arcface_s", 30.0))])
    if cfg.get("loss_type") == "focal":
        cmd.extend(["--focal_gamma", str(cfg.get("focal_gamma", 2.0))])
    if cfg.get("val_tta_hflip", False):
        cmd.append("--val_tta_hflip")
    if args.pseudo_dir is not None:
        cmd.extend(["--pseudo_dir", str(args.pseudo_dir)])
    if args.pseudo_conf_csv is not None:
        cmd.extend(["--pseudo_conf_csv", str(args.pseudo_conf_csv)])
    if args.pseudo_ignore_conf_filter:
        cmd.append("--pseudo_ignore_conf_filter")

    run_cmd(cmd)
    meta_path = Path("outputs") / f"meta_{run_name}.json"
    meta = json.loads(meta_path.read_text())
    return {
        "run_name": run_name,
        "family": "neural",
        "variant": cfg["name"],
        "method": cfg["head_type"],
        "selection_metric": sel_metric,
        "pseudo_profile": profile["name"],
        "best_val_acc": float(meta.get("best_val_acc", 0.0)),
        "best_balanced_acc": float(meta.get("best_balanced_acc", 0.0)),
        "best_secondary_val_acc": float(meta.get("best_secondary_val_acc", 0.0) or 0.0),
        "best_secondary_balanced_acc": float(meta.get("best_secondary_balanced_acc", 0.0) or 0.0),
        "meta_path": str(meta_path),
    }


def run_embedding(cfg, args, profile, sel_metric):
    run_name = (
        f"overnight_emb_{cfg['name']}_{args.backbone}_{args.img_size}_"
        f"{profile['name']}_{sel_metric}"
    )
    cmd = [
        sys.executable,
        "scripts/train_eval_crow_separator_embedding.py",
        "--method",
        cfg["method"],
        "--backbone",
        args.backbone,
        "--img_size",
        str(args.img_size),
        "--batch_size",
        str(max(32, args.batch_size * 2)),
        "--num_workers",
        str(args.num_workers),
        "--val_dir",
        str(args.val_dir),
        "--secondary_val_dir",
        str(args.secondary_val_dir),
        "--selection_metric",
        sel_metric,
        "--real_weight",
        "1.0",
        "--pseudo_weight",
        str(profile["pseudo_weight"]),
        "--pseudo_min_conf",
        str(profile["pseudo_min_conf"]),
        "--pca_dim",
        str(cfg.get("pca_dim", 0)),
        "--run_name",
        run_name,
    ]
    if args.use_crops:
        cmd.append("--use_crops")
    if args.pseudo_dir is not None:
        cmd.extend(["--pseudo_dir", str(args.pseudo_dir)])
    if args.pseudo_conf_csv is not None:
        cmd.extend(["--pseudo_conf_csv", str(args.pseudo_conf_csv)])
    if args.pseudo_ignore_conf_filter:
        cmd.append("--pseudo_ignore_conf_filter")

    if cfg["method"] == "knn":
        cmd.extend(["--knn_k", str(cfg["knn_k"])])
        cmd.extend(["--knn_metric", cfg["knn_metric"]])
        cmd.extend(["--knn_weights", cfg["knn_weights"]])
        cmd.extend(["--knn_weight_mode", cfg.get("knn_weight_mode", "none")])
        cmd.extend(["--knn_rep_scale", str(cfg.get("knn_rep_scale", 4.0))])
    elif cfg["method"] == "rbf_svm":
        cmd.extend(["--svm_c", str(cfg["svm_c"])])
        cmd.extend(["--svm_gamma", str(cfg["svm_gamma"])])
    elif cfg["method"] == "logreg":
        cmd.extend(["--logreg_c", str(cfg["logreg_c"])])

    run_cmd(cmd)
    meta_path = Path("outputs") / f"meta_{run_name}.json"
    meta = json.loads(meta_path.read_text())
    return {
        "run_name": run_name,
        "family": "embedding",
        "variant": cfg["name"],
        "method": cfg["method"],
        "selection_metric": sel_metric,
        "pseudo_profile": profile["name"],
        "best_val_acc": float(meta.get("best_val_acc", 0.0)),
        "best_balanced_acc": float(meta.get("best_balanced_acc", 0.0)),
        "best_secondary_val_acc": float(meta.get("best_secondary_val_acc", 0.0)),
        "best_secondary_balanced_acc": float(meta.get("best_secondary_balanced_acc", 0.0)),
        "meta_path": str(meta_path),
    }


def main():
    args = parse_args()
    rows = []

    jobs = []
    profiles = pseudo_profiles(args)
    for profile in profiles:
        for sel_metric in selection_metrics():
            for cfg in neural_configs():
                jobs.append(("neural", cfg, profile, sel_metric))
            for cfg in embedding_configs():
                jobs.append(("embedding", cfg, profile, sel_metric))

    print(
        f"Planned runs: {len(jobs)} "
        f"(neural={len(neural_configs()) * len(profiles) * len(selection_metrics())}, "
        f"embedding={len(embedding_configs()) * len(profiles) * len(selection_metrics())})"
    )
    for i, (kind, cfg, profile, sel_metric) in enumerate(jobs, start=1):
        print(f"[{i:02d}/{len(jobs)}] {kind} | {cfg['name']} | pseudo={profile['name']} | select={sel_metric}")
        try:
            if kind == "neural":
                row = run_neural(cfg=cfg, args=args, profile=profile, sel_metric=sel_metric)
            else:
                row = run_embedding(cfg=cfg, args=args, profile=profile, sel_metric=sel_metric)
            row["robust_acc_avg"] = 0.5 * (row["best_val_acc"] + row["best_secondary_val_acc"])
            row["robust_balanced_avg"] = 0.5 * (row["best_balanced_acc"] + row["best_secondary_balanced_acc"])
            rows.append(row)
        except Exception as ex:  # noqa: BLE001
            print(f"[ERROR] {kind} {cfg['name']} failed: {ex}")
            if not args.continue_on_error:
                raise

    if not rows:
        raise RuntimeError("No successful runs. Nothing to rank.")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["robust_balanced_avg", "robust_acc_avg", "best_balanced_acc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.results_csv, index=False)

    print("\n=== Top 20 runs by robust balanced accuracy ===")
    with pd.option_context("display.max_rows", 20, "display.width", 220):
        print(
            df.head(20).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
    print(f"\nSaved results: {args.results_csv}")


if __name__ == "__main__":
    main()

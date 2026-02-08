import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd


DEFAULT_SPECS = [
    ("vit_so150m2_384", 384),
    ("convnextv2_base_384", 384),
    ("convnextv2_large_384", 384),
    ("caformer_b36_384", 384),
    ("eva02_large_448", 448),
]


def parse_spec(token: str) -> Tuple[str, int]:
    if ":" in token:
        name, size_str = token.split(":", 1)
        size = int(size_str)
        return name.strip(), size
    return token.strip(), 384


def parse_specs(tokens: List[str]) -> List[Tuple[str, int]]:
    if not tokens:
        return list(DEFAULT_SPECS)
    return [parse_spec(t) for t in tokens]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="Backbone spec: backbone_name:img_size. Can be repeated. Example: --spec vit_so150m2_384:384",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--finetune_backbone", action="store_true")
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--real_weight", type=float, default=1.0)
    parser.add_argument("--pseudo_weight", type=float, default=0.35)
    parser.add_argument("--pseudo_min_conf", type=float, default=0.95)
    parser.add_argument("--pseudo_dir", type=Path, default=None)
    parser.add_argument(
        "--pseudo_conf_csv",
        type=Path,
        default=Path("outputs/extra_images_cv_ensemble_predictions_with_conf.csv"),
    )
    parser.add_argument(
        "--val_dir",
        type=Path,
        default=Path("data/f_images_cropped"),
        help="Validation root used for ranking backbone performance.",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["acc", "balanced_acc"],
        default="balanced_acc",
        help="Metric used inside each run for best-checkpoint selection.",
    )
    parser.add_argument(
        "--results_csv",
        type=Path,
        default=Path("outputs/crow_separator_backbone_benchmark.csv"),
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue running next specs if one training run fails.",
    )
    return parser.parse_args()


def run_one(
    backbone: str,
    img_size: int,
    args,
):
    run_name = f"crow_sep_bench_{backbone}_{img_size}"
    cmd = [
        sys.executable,
        "scripts/train_crow_separator.py",
        "--backbone",
        backbone,
        "--img_size",
        str(img_size),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--lr_head",
        str(args.lr_head),
        "--lr_backbone",
        str(args.lr_backbone),
        "--real_weight",
        str(args.real_weight),
        "--pseudo_weight",
        str(args.pseudo_weight),
        "--pseudo_min_conf",
        str(args.pseudo_min_conf),
        "--val_dir",
        str(args.val_dir),
        "--selection_metric",
        args.selection_metric,
        "--run_name",
        run_name,
    ]

    if args.use_crops:
        cmd.append("--use_crops")
    if args.finetune_backbone:
        cmd.append("--finetune_backbone")
    if args.pseudo_dir is not None:
        cmd.extend(["--pseudo_dir", str(args.pseudo_dir)])
    if args.pseudo_conf_csv is not None:
        cmd.extend(["--pseudo_conf_csv", str(args.pseudo_conf_csv)])

    print(f"\n=== Running: {backbone} @ {img_size} ===")
    subprocess.run(cmd, check=True)

    meta_path = Path("outputs") / f"meta_{run_name}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Expected meta file not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    return {
        "backbone": backbone,
        "img_size": img_size,
        "run_name": run_name,
        "best_val_acc": float(meta.get("best_val_acc", 0.0)),
        "best_balanced_acc": float(meta.get("best_balanced_acc", 0.0)),
        "best_val_acc_american_crow": float(meta.get("best_val_acc_american_crow", 0.0)),
        "best_val_acc_fish_crow": float(meta.get("best_val_acc_fish_crow", 0.0)),
        "best_secondary_val_acc": float(meta.get("best_secondary_val_acc", 0.0) or 0.0),
        "best_secondary_balanced_acc": float(meta.get("best_secondary_balanced_acc", 0.0) or 0.0),
        "best_secondary_val_acc_american_crow": float(meta.get("best_secondary_val_acc_american_crow", 0.0) or 0.0),
        "best_secondary_val_acc_fish_crow": float(meta.get("best_secondary_val_acc_fish_crow", 0.0) or 0.0),
        "selection_metric": meta.get("selection_metric", args.selection_metric),
    }


def main():
    args = parse_args()
    specs = parse_specs(args.spec)

    rows = []
    for backbone, img_size in specs:
        try:
            row = run_one(backbone, img_size, args)
            rows.append(row)
        except Exception as ex:  # noqa: BLE001
            print(f"[ERROR] Failed for {backbone}:{img_size} -> {ex}")
            if not args.continue_on_error:
                raise

    if not rows:
        raise RuntimeError("No successful runs. Nothing to rank.")

    df = pd.DataFrame(rows)
    if args.selection_metric == "acc":
        df = df.sort_values(["best_val_acc", "best_balanced_acc"], ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values(["best_balanced_acc", "best_val_acc"], ascending=False).reset_index(drop=True)

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.results_csv, index=False)

    print("\n=== Backbone ranking (crow separator) ===")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved results: {args.results_csv}")


if __name__ == "__main__":
    main()

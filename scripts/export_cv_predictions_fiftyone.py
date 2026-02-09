import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from birds_ml.config import Config
from birds_ml.utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export CV fold predictions to a portable folder for local FiftyOne usage."
    )
    parser.add_argument("--run_name", default=None, help="CV run name under outputs/cv_runs/")
    parser.add_argument("--backbone", default="eva02_large_448")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--use_arcface", action="store_true")
    parser.add_argument(
        "--export_dir",
        default=None,
        help="Output folder for portable data. Default: outputs/fiftyone_exports/<run_name>",
    )
    parser.add_argument(
        "--no_copy_images",
        action="store_true",
        help="Do not copy images. Only export metadata.",
    )
    return parser.parse_args()


def _safe_rel_path(project_root: Path, src: Path) -> Path:
    try:
        return src.resolve().relative_to(project_root.resolve())
    except ValueError:
        digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
        return Path("external") / digest / src.name


def _ensure_unique_path(dst_path: Path) -> Path:
    if not dst_path.exists():
        return dst_path
    stem = dst_path.stem
    suffix = dst_path.suffix
    parent = dst_path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def main():
    args = parse_args()
    cfg = Config()

    if args.run_name:
        run_name = args.run_name
    else:
        head_prefix = "arcface" if args.use_arcface else "linear"
        crop_suffix = "_cropped" if args.use_crops else ""
        run_name = f"cv_{head_prefix}_{args.backbone}{crop_suffix}_{args.img_size}_e{args.epochs}_f{args.folds}"

    run_dir = cfg.outputs_dir / "cv_runs" / run_name
    oof_path = run_dir / "oof_predictions.csv"
    meta_path = run_dir / "meta.json"

    if not run_dir.exists():
        raise FileNotFoundError(f"CV run directory not found: {run_dir}")
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions file: {oof_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing run metadata file: {meta_path}")

    run_meta = json.loads(meta_path.read_text())
    df = pd.read_csv(oof_path)

    required_cols = {
        "fold",
        "source_path",
        "ground_truth",
        "predicted",
        "confidence",
        "correct",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"OOF file is missing required columns: {sorted(missing)}")

    export_dir = Path(args.export_dir) if args.export_dir else cfg.outputs_dir / "fiftyone_exports" / run_name
    ensure_dir(export_dir)
    copy_images = not args.no_copy_images
    ensure_dir(export_dir / "images")

    manifest_rows = []
    copied = 0

    for row in df.to_dict(orient="records"):
        src = Path(row["source_path"])
        if not src.exists():
            continue

        rel_src = _safe_rel_path(cfg.project_root, src)
        rel_dst = Path("images") / f"fold_{int(row['fold'])}" / rel_src
        dst = export_dir / rel_dst

        if copy_images:
            ensure_dir(dst.parent)
            dst = _ensure_unique_path(dst)
            shutil.copy2(src, dst)
            copied += 1
            rel_filepath = dst.relative_to(export_dir)
        else:
            rel_filepath = src

        raw_correct = row["correct"]
        if isinstance(raw_correct, str):
            correct_value = raw_correct.strip().lower() in {"1", "true", "t", "yes", "y"}
        else:
            correct_value = bool(raw_correct)

        manifest_rows.append(
            {
                "filepath": str(rel_filepath),
                "source_path": str(src),
                "fold": int(row["fold"]),
                "ground_truth": row["ground_truth"],
                "predicted": row["predicted"],
                "confidence": float(row["confidence"]),
                "correct": correct_value,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = export_dir / "samples.csv"
    manifest_df.to_csv(manifest_path, index=False)

    export_meta = {
        "run_name": run_name,
        "copied_images": bool(copy_images),
        "num_rows": int(len(manifest_df)),
        "backbone": run_meta.get("backbone"),
        "img_size": run_meta.get("img_size"),
        "epochs": run_meta.get("epochs"),
        "folds": run_meta.get("folds"),
        "use_arcface": run_meta.get("use_arcface"),
        "use_crops": run_meta.get("use_crops"),
    }
    (export_dir / "export_meta.json").write_text(json.dumps(export_meta, indent=2))

    readme = [
        "CV predictions export for FiftyOne",
        "",
        "Files:",
        "- samples.csv: per-image records with fold, ground_truth, predicted, confidence, correct",
        "- export_meta.json: summary of the export",
        "- images/: copied files (unless --no_copy_images is used)",
        "",
        "Local FiftyOne usage example:",
        "import pandas as pd",
        "import fiftyone as fo",
        "df = pd.read_csv('samples.csv')",
        "ds = fo.Dataset('birds_cv_export')",
        "for r in df.to_dict(orient='records'):",
        "    s = fo.Sample(filepath=r['filepath'])",
        "    s['ground_truth'] = fo.Classification(label=r['ground_truth'])",
        "    s['prediction'] = fo.Classification(label=r['predicted'], confidence=float(r['confidence']))",
        "    s['fold'] = int(r['fold'])",
        "    s['correct'] = bool(r['correct'])",
        "    ds.add_sample(s)",
        "fo.launch_app(ds)",
        "",
    ]
    (export_dir / "README.txt").write_text("\n".join(readme))

    print(f"Run dir: {run_dir}")
    print(f"Export dir: {export_dir}")
    print(f"Rows exported: {len(manifest_df)}")
    if copy_images:
        print(f"Images copied: {copied}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

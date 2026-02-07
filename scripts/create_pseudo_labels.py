import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_CLASSES = [
    "American_Crow",
    "Fish_Crow",
    "Black_billed_Cuckoo",
    "Yellow_billed_Cuckoo",
    "Lazuli_Bunting",
    "Brewer_Blackbird",
    "Rusty_Blackbird",
]


def _norm(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Create pseudo-label folders from prediction CSV by filtering selected classes "
            "and confidence threshold."
        )
    )
    ap.add_argument(
        "--pred_csv",
        default="outputs/extra_images_cv_ensemble_predictions_with_conf.csv",
        help="Prediction CSV with columns: relative_path,predicted_label,confidence.",
    )
    ap.add_argument(
        "--source_dir",
        default="data/extra_images",
        help="Root directory containing predicted images.",
    )
    ap.add_argument(
        "--out_dir",
        default="data/pseudo_labels",
        help="Output root where pseudo-label class subfolders are created.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum confidence required to copy an image.",
    )
    ap.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Target predicted classes (case-insensitive, underscores/hyphens ignored).",
    )
    args = ap.parse_args()

    pred_csv = Path(args.pred_csv)
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)

    if not pred_csv.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {pred_csv}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    canonical_by_norm = {_norm(cls): cls for cls in args.classes}
    for cls in canonical_by_norm.values():
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    copied_counts = {cls: 0 for cls in canonical_by_norm.values()}
    missing_source = 0
    considered = 0
    copied = 0

    with pred_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"predicted_label", "confidence"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {sorted(required_cols)}. Found: {reader.fieldnames}"
            )

        for row in reader:
            considered += 1
            label = (row.get("predicted_label") or "").strip()
            label_key = _norm(label)
            if label_key not in canonical_by_norm:
                continue

            conf = float(row["confidence"])
            if conf < args.threshold:
                continue

            rel = (row.get("relative_path") or row.get("path") or "").strip()
            if not rel:
                continue

            src = source_dir / Path(rel)
            if not src.exists():
                # fallback for CSVs that store only filename in `path`
                src = source_dir / Path(row.get("path", "")).name
            if not src.exists():
                missing_source += 1
                continue

            dst_dir = out_dir / canonical_by_norm[label_key]
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied += 1
            copied_counts[canonical_by_norm[label_key]] += 1

    print(f"Read rows: {considered}")
    print(f"Copied images: {copied}")
    print(f"Missing source files: {missing_source}")
    print(f"Output root: {out_dir}")
    for cls in DEFAULT_CLASSES:
        if cls in copied_counts:
            print(f"  {cls}: {copied_counts[cls]}")


if __name__ == "__main__":
    main()

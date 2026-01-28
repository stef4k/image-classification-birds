import argparse
import csv
from pathlib import Path

import fiftyone as fo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV exported from Ruche (val_preds_*.csv)")
    ap.add_argument("--images_root", default=None,
                    help="Optional: if CSV filepaths are absolute Ruche paths, remap them under this local root")
    ap.add_argument("--ds_name", default="birds_val_neural_local")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    if fo.dataset_exists(args.ds_name):
        fo.delete_dataset(args.ds_name)
    dataset = fo.Dataset(args.ds_name)

    images_root = Path(args.images_root) if args.images_root else None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fp = Path(r["filepath"])

            # If CSV contains Ruche absolute paths, you must remap to local paths
            # Example: fp = /gpfs/.../data/val_cropped/xxx.jpg
            # Local:   images_root = ./val_cropped
            # Then we keep only the filename (or relative suffix) depending on your copy structure.
            if images_root is not None:
                # simplest: assume you copied images preserving relative structure under images_root
                # Take last 2 parts to keep class folder + filename (adjust if needed)
                fp = images_root / Path(*fp.parts[-2:])

            sample = fo.Sample(filepath=str(fp))
            sample["ground_truth"] = fo.Classification(label=r["true"])
            sample["prediction"] = fo.Classification(label=r["pred"])
            sample["correct"] = bool(int(r["correct"]))
            sample["confidence"] = float(r["confidence"])
            sample["hardness"] = float(r["hardness"])
            sample["topk"] = r["topk"].split("|") if r["topk"] else []

            if sample["correct"]:
                sample.tags.append("correct")
            else:
                sample.tags.append("mistake")

            dataset.add_sample(sample)

    dataset.persistent = True
    print(f"Loaded dataset: {dataset.name} | samples={len(dataset)}")
    print("Tips: filter tag 'mistake' or correct == false; sort by hardness desc")
    fo.launch_app(dataset).wait()


if __name__ == "__main__":
    main()

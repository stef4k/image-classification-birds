import argparse
from dataclasses import replace
import os
from pathlib import Path

import pandas as pd

from birds_ml.config import Config
from birds_ml.data import load_test_recursive


KAGGLE_IDX_TO_NAME = {
    0: "Groove_billed_Ani",
    1: "Red_winged_Blackbird",
    2: "Rusty_Blackbird",
    3: "Gray_Catbird",
    4: "Brandt_Cormorant",
    5: "Eastern_Towhee",
    6: "Indigo_Bunting",
    7: "Brewer_Blackbird",
    8: "Painted_Bunting",
    9: "Bobolink",
    10: "Lazuli_Bunting",
    11: "Yellow_headed_Blackbird",
    12: "American_Crow",
    13: "Fish_Crow",
    14: "Brown_Creeper",
    15: "Yellow_billed_Cuckoo",
    16: "Yellow_breasted_Chat",
    17: "Black_billed_Cuckoo",
    18: "Gray_crowned_Rosy_Finch",
    19: "Bronzed_Cowbird",
}


def _resolve_submission_path(path: Path) -> Path:
    if path.exists():
        return path

    candidates = [Path("outputs") / path.name, Path("output") / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    attempted = [str(path)] + [str(c) for c in candidates]
    raise FileNotFoundError(f"Submission CSV not found. Tried: {attempted}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("outputs/submission_eva02_large_448_arcface.csv"),
        help="Submission CSV with columns: path,class_idx",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="FiftyOne dataset name (default auto-generated from submission + crop mode)",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=None,
        help="Optional override for test images directory",
    )
    parser.add_argument(
        "--database_dir",
        type=Path,
        default=Path("outputs/fiftyone_db"),
        help="FiftyOne database directory (default: outputs/fiftyone_db)",
    )
    parser.add_argument(
        "--use_crops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cropped test split (default: true)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_persist", action="store_true")
    parser.add_argument("--no_launch", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = replace(Config(), use_crops=args.use_crops)

    args.database_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(args.database_dir))

    import fiftyone as fo

    submission_path = _resolve_submission_path(args.submission)
    test_dir = args.test_dir or cfg.test_dir
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    dataset_name = args.dataset_name
    if not dataset_name:
        dataset_name = submission_path.stem
        if args.use_crops:
            dataset_name += "_cropped"

    sub = pd.read_csv(submission_path)
    required_cols = {"path", "class_idx"}
    if not required_cols.issubset(sub.columns):
        raise ValueError(f"{submission_path} must contain columns: {sorted(required_cols)}")

    if args.limit is not None:
        sub = sub.head(args.limit).copy()

    sub["path"] = sub["path"].astype(str)
    sub["class_idx"] = sub["class_idx"].astype(int)

    test_samples = load_test_recursive(test_dir)
    path_map = {}
    for s in test_samples:
        name = s.path.name
        if name in path_map:
            raise ValueError(f"Duplicate test filename found, cannot map uniquely: {name}")
        path_map[name] = s.path

    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    samples = []
    missing = []
    for row in sub.itertuples(index=False):
        submission_name = row.path
        filename = Path(submission_name).name
        image_path = path_map.get(filename)
        if image_path is None:
            missing.append(submission_name)
            continue

        class_idx = int(row.class_idx)
        class_name = KAGGLE_IDX_TO_NAME.get(class_idx, f"class_{class_idx}")

        sample = fo.Sample(filepath=str(image_path))
        sample["prediction"] = fo.Classification(label=class_name)
        sample["class_idx"] = class_idx
        sample["submission_path"] = submission_name
        sample["is_crow"] = bool(class_idx in (12, 13))
        samples.append(sample)

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} submission paths were not found under {test_dir}. Examples: {preview}"
        )

    dataset.add_samples(samples)
    dataset.persistent = not args.no_persist

    print(f"Loaded submission: {submission_path}")
    print(f"Dataset created: {dataset.name} with {len(dataset)} samples")
    print("Fields: prediction, class_idx, submission_path, is_crow")

    if args.no_launch:
        return

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()

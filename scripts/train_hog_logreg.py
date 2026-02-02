#!/usr/bin/env python3
"""
HOG + Logistic Regression baseline (non-deep-learning).

Typical usage:

1) Train + eval (if you have val):
python scripts/train_hog_logreg.py \
  train \
  --train_dir data/train_images_cropped \
  --val_dir data/val_images_cropped \
  --out outputs/hog_logreg.joblib \
  --img_size 128 \
  --C 4.0 \
  --max_iter 3000

2) Predict a folder (e.g., test):
python scripts/train_hog_logreg.py \
  predict \
  --model outputs/hog_logreg.joblib \
  --input_dir data/test_images_cropped \
  --out_csv outputs/test_preds.csv

Notes:
- Cropping helps a lot. If you don't crop, this will learn background junk.
- HOG is grayscale by default. You can optionally add a small HSV histogram.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Image IO / resize
try:
    import cv2
except Exception as e:
    raise RuntimeError(
        "OpenCV (cv2) is required for this script. Install with: pip install opencv-python"
    ) from e

# HOG feature extraction (skimage only)
try:
    from skimage.feature import hog as skimage_hog
except Exception as e:
    raise RuntimeError(
        "scikit-image is required for this script. Install with: pip install scikit-image"
    ) from e

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class HogConfig:
    img_size: int = 128
    orientations: int = 9
    pixels_per_cell: int = 12 #8
    cells_per_block: int = 2
    block_norm: str = "L2-Hys"  # skimage only; ignored in OpenCV backend
    use_color_hist: bool = False
    color_hist_bins: int = 16  # per channel in HSV histogram (H, S, V)


def list_images_recursive(root: Path) -> List[Path]:
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    paths.sort()
    return paths


def list_class_folders(root: Path) -> List[Path]:
    # expects structure: root/class_name/*.jpg
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    class_dirs.sort(key=lambda x: x.name)
    return class_dirs


def build_label_mapping_from_folders(train_dir: Path) -> Dict[str, int]:
    class_dirs = list_class_folders(train_dir)
    if not class_dirs:
        raise ValueError(f"No class subfolders found in {train_dir}")
    return {d.name: i for i, d in enumerate(class_dirs)}


def load_labeled_paths(root: Path, mapping: Dict[str, int]) -> Tuple[List[Path], np.ndarray]:
    x_paths: List[Path] = []
    y: List[int] = []
    for cls_name, cls_id in mapping.items():
        cls_dir = root / cls_name
        if not cls_dir.exists():
            # allow missing classes in val (common if val is small)
            continue
        for img_path in list_images_recursive(cls_dir):
            x_paths.append(img_path)
            y.append(cls_id)
    if not x_paths:
        raise ValueError(f"No labeled images found under {root}")
    return x_paths, np.asarray(y, dtype=np.int64)


def read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def preprocess_to_gray(img_bgr: np.ndarray, img_size: int) -> np.ndarray:
    img = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def hsv_histogram(img_bgr: np.ndarray, img_size: int, bins: int) -> np.ndarray:
    img = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hist per channel, normalized
    feats = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [bins], [0, 256])
        h = h.astype(np.float32).reshape(-1)
        h /= (h.sum() + 1e-8)
        feats.append(h)
    return np.concatenate(feats, axis=0)


def hog_features(gray: np.ndarray, cfg: HogConfig) -> np.ndarray:
    feat = skimage_hog(
        gray,
        orientations=cfg.orientations,
        pixels_per_cell=(cfg.pixels_per_cell, cfg.pixels_per_cell),
        cells_per_block=(cfg.cells_per_block, cfg.cells_per_block),
        block_norm=cfg.block_norm,
        feature_vector=True,
    ).astype(np.float32)
    return feat


def extract_features(paths: List[Path], cfg: HogConfig, cache_npz: Optional[Path] = None) -> np.ndarray:
    if cache_npz is not None and cache_npz.exists():
        data = np.load(cache_npz)
        X = data["X"]
        if X.shape[0] == len(paths):
            return X
        # If cache count mismatched, ignore cache

    feats = []
    for p in paths:
        img = read_image_bgr(p)
        gray = preprocess_to_gray(img, cfg.img_size)
        f_hog = hog_features(gray, cfg)
        if cfg.use_color_hist:
            f_col = hsv_histogram(img, cfg.img_size, cfg.color_hist_bins)
            f = np.concatenate([f_hog, f_col], axis=0)
        else:
            f = f_hog
        feats.append(f)

    # ragged guard
    dim0 = feats[0].shape[0]
    for i, f in enumerate(feats):
        if f.shape[0] != dim0:
            raise RuntimeError(f"Feature dim mismatch at idx={i}: got {f.shape[0]}, expected {dim0}")

    X = np.stack(feats, axis=0).astype(np.float32)

    if cache_npz is not None:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_npz, X=X)

    return X


def build_model(C: float, max_iter: int, n_jobs: int) -> Pipeline:
    # Standardize helps logistic regression a lot with HOG
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        multi_class="auto",
        class_weight="balanced",
        n_jobs=n_jobs,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf),
        ]
    )


def cmd_train(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir) if args.val_dir else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = HogConfig(
        img_size=args.img_size,
        orientations=args.orientations,
        pixels_per_cell=args.pixels_per_cell,
        cells_per_block=args.cells_per_block,
        use_color_hist=args.use_color_hist,
        color_hist_bins=args.color_hist_bins,
    )

    # 1) Labels
    mapping = build_label_mapping_from_folders(train_dir)
    inv_mapping = {v: k for k, v in mapping.items()}

    # 2) Load paths
    train_paths, y_train = load_labeled_paths(train_dir, mapping)
    print(f"[train] images={len(train_paths)} classes={len(mapping)} hog_backend=skimage")

    # 3) Features
    cache_train = Path(args.cache_train) if args.cache_train else None
    X_train = extract_features(train_paths, cfg, cache_npz=cache_train)
    print(f"[train] X={X_train.shape} y={y_train.shape}")

    # 4) 5-fold CV on train set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    for i, (tr, te) in enumerate(skf.split(X_train, y_train), start=1):
        model = build_model(C=args.C, max_iter=args.max_iter, n_jobs=args.n_jobs)
        model.fit(X_train[tr], y_train[tr])
        pred = model.predict(X_train[te])
        acc = accuracy_score(y_train[te], pred)
        macro_f1 = f1_score(y_train[te], pred, average="macro")
        fold_metrics.append({"acc": float(acc), "macro_f1": float(macro_f1)})
        print(f"[cv {i}] acc={acc:.4f} macro_f1={macro_f1:.4f}")

    cv_acc = float(np.mean([m["acc"] for m in fold_metrics]))
    cv_f1 = float(np.mean([m["macro_f1"] for m in fold_metrics]))
    cv_acc_std = float(np.std([m["acc"] for m in fold_metrics]))
    cv_f1_std = float(np.std([m["macro_f1"] for m in fold_metrics]))
    print(f"[cv avg] acc={cv_acc:.4f} macro_f1={cv_f1:.4f}")
    print(f"[cv std] acc={cv_acc_std:.4f} macro_f1={cv_f1_std:.4f}")

    # 5) Fit final model on full train set
    model = build_model(C=args.C, max_iter=args.max_iter, n_jobs=args.n_jobs)
    model.fit(X_train, y_train)

    # 6) Optional val eval
    metrics = {"cv_acc": cv_acc, "cv_macro_f1": cv_f1, "cv_acc_std": cv_acc_std, "cv_macro_f1_std": cv_f1_std}
    if val_dir is not None:
        val_paths, y_val = load_labeled_paths(val_dir, mapping)
        cache_val = Path(args.cache_val) if args.cache_val else None
        X_val = extract_features(val_paths, cfg, cache_npz=cache_val)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro")
        print(f"[val] images={len(val_paths)} acc={acc:.4f} macro_f1={macro_f1:.4f}")

        # Print per-class report (helpful when val is tiny)
        target_names = [inv_mapping[i] for i in range(len(inv_mapping))]
        print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))

        metrics = {"val_acc": float(acc), "val_macro_f1": float(macro_f1)}

    # 6) Save artifact (model + config + mapping)
    artifact = {
        "model": model,
        "hog_config": cfg,
        "class_to_idx": mapping,
        "idx_to_class": inv_mapping,
        "metrics": metrics,
    }
    joblib.dump(artifact, out_path)
    print(f"[ok] saved: {out_path}")


def cmd_predict(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    input_dir = Path(args.input_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(model_path)
    model: Pipeline = artifact["model"]
    cfg: HogConfig = artifact["hog_config"]
    idx_to_class: Dict[int, str] = artifact["idx_to_class"]

    paths = list_images_recursive(input_dir)
    if not paths:
        raise ValueError(f"No images found under {input_dir}")

    X = extract_features(paths, cfg, cache_npz=None)
    y_pred = model.predict(X)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)

    # Write CSV: filename,pred_label,(optional probs...)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["filename", "pred_label"]
        if y_prob is not None:
            # probs in class index order
            prob_cols = [f"prob_{idx_to_class[i]}" for i in range(y_prob.shape[1])]
            header.extend(prob_cols)
        writer.writerow(header)

        for i, p in enumerate(paths):
            row = [p.name, idx_to_class[int(y_pred[i])]]
            if y_prob is not None:
                row.extend([f"{float(v):.6f}" for v in y_prob[i]])
            writer.writerow(row)

    print(f"[ok] wrote: {out_csv} (n={len(paths)})")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="HOG + Logistic Regression baseline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train and optionally evaluate on val")
    tr.add_argument("--train_dir", required=True, help="train root folder with class subfolders")
    tr.add_argument("--val_dir", default=None, help="val root folder with class subfolders (optional)")
    tr.add_argument("--out", required=True, help="output .joblib path")

    # HOG
    tr.add_argument("--img_size", type=int, default=128)
    tr.add_argument("--orientations", type=int, default=9)
    tr.add_argument("--pixels_per_cell", type=int, default=8)
    tr.add_argument("--cells_per_block", type=int, default=2)

    # Optional color
    tr.add_argument("--use_color_hist", action="store_true", help="append small HSV histogram")
    tr.add_argument("--color_hist_bins", type=int, default=16)

    # Logistic regression
    tr.add_argument("--C", type=float, default=4.0, help="inverse regularization strength")
    tr.add_argument("--max_iter", type=int, default=3000)
    tr.add_argument("--n_jobs", type=int, default=7, help="logreg n_jobs (lbfgs ignores >1)")

    # Caching
    tr.add_argument("--cache_train", default=None, help="npz cache for train features")
    tr.add_argument("--cache_val", default=None, help="npz cache for val features")

    pr = sub.add_parser("predict", help="predict on an unlabeled folder")
    pr.add_argument("--model", required=True, help="path to saved .joblib artifact")
    pr.add_argument("--input_dir", required=True, help="folder with images (recursive)")
    pr.add_argument("--out_csv", required=True, help="output CSV path")

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

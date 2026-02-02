#!/usr/bin/env python3
"""
Fisher Vector + Linear Classifier baseline (non-deep-learning).

References:
- Perronnin et al., "Image Classification with the Fisher Vector: Theory and Practice"

Typical usage:

1) Train + eval (if you have val):
python scripts/train_fisher_vector.py \
  train \
  --train_dir data/train_images_cropped \
  --val_dir data/val_images_cropped \
  --out outputs/fv_linear.joblib \
  --n_components 64 \
  --img_size 256 \
  --classifier linear_svm \
  --C 1.0

2) Predict a folder (e.g., test):
python scripts/train_fisher_vector.py \
  predict \
  --model outputs/fv_linear.joblib \
  --input_dir data/test_images_cropped \
  --out_csv outputs/test_preds_fv.csv
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as e:
    raise RuntimeError(
        "OpenCV (cv2) is required for this script. Install with: pip install opencv-python"
    ) from e

from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import joblib


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class FisherConfig:
    img_size: int = 256
    dense_step: int = 8
    dense_size: int = 12
    max_per_image: int = 600
    use_rootsift: bool = True
    n_components: int = 64
    random_state: int = 42


def list_images_recursive(root: Path) -> List[Path]:
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    paths.sort()
    return paths


def list_class_folders(root: Path) -> List[Path]:
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


def preprocess_gray(img_bgr: np.ndarray, img_size: int) -> np.ndarray:
    img = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def dense_keypoints(height: int, width: int, step: int, size: int) -> List[cv2.KeyPoint]:
    kps: List[cv2.KeyPoint] = []
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            kps.append(cv2.KeyPoint(float(x), float(y), float(size)))
    return kps


def _create_sift() -> cv2.SIFT:
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    raise RuntimeError(
        "SIFT is unavailable in your OpenCV build. Install opencv-contrib-python or a newer opencv-python."
    )


def rootsift(desc: np.ndarray) -> np.ndarray:
    eps = 1e-12
    desc = desc.astype(np.float32)
    desc /= (np.sum(desc, axis=1, keepdims=True) + eps)
    desc = np.sqrt(desc)
    return desc


def extract_sift_descriptors(img_bgr: np.ndarray, cfg: FisherConfig) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    gray = preprocess_gray(img_bgr, cfg.img_size)
    sift = _create_sift()
    kps = dense_keypoints(gray.shape[0], gray.shape[1], cfg.dense_step, cfg.dense_size)
    kps, desc = sift.compute(gray, kps)
    if desc is None or len(desc) == 0:
        return kps, None
    if cfg.use_rootsift:
        desc = rootsift(desc)
    if cfg.max_per_image > 0 and desc.shape[0] > cfg.max_per_image:
        rng = np.random.default_rng(cfg.random_state)
        idx = rng.choice(desc.shape[0], size=cfg.max_per_image, replace=False)
        desc = desc[idx]
    return kps, desc


def sample_descriptors(paths: List[Path], cfg: FisherConfig, max_desc: int) -> np.ndarray:
    all_desc = []
    for p in paths:
        img = read_image_bgr(p)
        _, desc = extract_sift_descriptors(img, cfg)
        if desc is None:
            continue
        all_desc.append(desc)
    if not all_desc:
        raise RuntimeError("No SIFT descriptors extracted; check input images and SIFT availability.")
    desc = np.vstack(all_desc)
    if desc.shape[0] > max_desc:
        rng = np.random.default_rng(cfg.random_state)
        idx = rng.choice(desc.shape[0], size=max_desc, replace=False)
        desc = desc[idx]
    return desc.astype(np.float64)


def fit_gmm(desc: np.ndarray, n_components: int, random_state: int) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        max_iter=200,
        random_state=random_state,
        reg_covar=1e-6,
        verbose=0,
    )
    gmm.fit(desc)
    return gmm


def fisher_vector(desc: Optional[np.ndarray], gmm: GaussianMixture) -> np.ndarray:
    k = gmm.n_components
    d = gmm.means_.shape[1]
    if desc is None or len(desc) == 0:
        return np.zeros((2 * k * d,), dtype=np.float32)

    x = desc.astype(np.float64)
    n = x.shape[0]

    # Posterior probabilities q_{n,k}
    q = gmm.predict_proba(x)  # (n, k)

    w = gmm.weights_.reshape(1, k)
    mu = gmm.means_  # (k, d)
    sigma = np.sqrt(gmm.covariances_)  # (k, d)

    # Compute sufficient statistics for mean and variance gradients
    u = np.zeros((k, d), dtype=np.float64)
    v = np.zeros((k, d), dtype=np.float64)

    for i in range(k):
        qk = q[:, i].reshape(-1, 1)
        diff = x - mu[i]
        u[i] = (qk * (diff / sigma[i])).sum(axis=0)
        v[i] = (qk * ((diff ** 2) / (sigma[i] ** 2) - 1.0)).sum(axis=0)

    u /= (n * np.sqrt(w).reshape(-1, 1))
    v /= (n * np.sqrt(2.0 * w).reshape(-1, 1))

    fv = np.concatenate([u, v], axis=0).reshape(-1)

    # Power + L2 normalization
    eps = 1e-12
    fv = np.sign(fv) * np.sqrt(np.abs(fv) + eps)
    fv = fv / (np.linalg.norm(fv) + eps)
    return fv.astype(np.float32)


def extract_fisher_vectors(
    paths: List[Path], cfg: FisherConfig, gmm: GaussianMixture, cache_npz: Optional[Path] = None
) -> np.ndarray:
    if cache_npz is not None and cache_npz.exists():
        data = np.load(cache_npz)
        X = data["X"]
        if X.shape[0] == len(paths):
            return X

    feats = []
    for p in paths:
        img = read_image_bgr(p)
        _, desc = extract_sift_descriptors(img, cfg)
        fv = fisher_vector(desc, gmm)
        feats.append(fv)

    X = np.stack(feats, axis=0).astype(np.float32)

    if cache_npz is not None:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_npz, X=X)

    return X


def build_model(classifier: str, C: float, max_iter: int, use_scaler: bool) -> Pipeline:
    if classifier == "linear_svm":
        clf = LinearSVC(C=C, max_iter=max_iter)
    elif classifier == "logreg":
        clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", multi_class="auto")
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler(with_mean=False, with_std=True)))
    steps.append(("clf", clf))
    return Pipeline(steps=steps)


def cmd_train(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir) if args.val_dir else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = FisherConfig(
        img_size=args.img_size,
        dense_step=args.dense_step,
        dense_size=args.dense_size,
        max_per_image=args.max_per_image,
        use_rootsift=not args.no_rootsift,
        n_components=args.n_components,
        random_state=args.seed,
    )

    mapping = build_label_mapping_from_folders(train_dir)
    inv_mapping = {v: k for k, v in mapping.items()}

    train_paths, y_train = load_labeled_paths(train_dir, mapping)
    print(f"[train] images={len(train_paths)} classes={len(mapping)}")

    # 1) Fit GMM
    gmm_cache = Path(args.gmm_cache) if args.gmm_cache else None
    if gmm_cache is not None and gmm_cache.exists():
        gmm = joblib.load(gmm_cache)
        print(f"[gmm] loaded: {gmm_cache}")
    else:
        print("[gmm] sampling descriptors...")
        desc = sample_descriptors(train_paths, cfg, max_desc=args.max_desc)
        print(f"[gmm] fitting GMM with {desc.shape[0]} descriptors (K={cfg.n_components})")
        gmm = fit_gmm(desc, cfg.n_components, cfg.random_state)
        if gmm_cache is not None:
            gmm_cache.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(gmm, gmm_cache)
            print(f"[gmm] saved: {gmm_cache}")

    # 2) Fisher vectors
    cache_train = Path(args.cache_train) if args.cache_train else None
    X_train = extract_fisher_vectors(train_paths, cfg, gmm, cache_npz=cache_train)
    print(f"[train] X={X_train.shape} y={y_train.shape}")

    # 3) 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    fold_metrics = []
    for i, (tr, te) in enumerate(skf.split(X_train, y_train), start=1):
        model = build_model(args.classifier, args.C, args.max_iter, args.use_scaler)
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

    # 4) Fit final model on full train set
    model = build_model(args.classifier, args.C, args.max_iter, args.use_scaler)
    model.fit(X_train, y_train)

    metrics = {
        "cv_acc": cv_acc,
        "cv_macro_f1": cv_f1,
        "cv_acc_std": cv_acc_std,
        "cv_macro_f1_std": cv_f1_std,
    }

    # 5) Optional val eval
    if val_dir is not None:
        val_paths, y_val = load_labeled_paths(val_dir, mapping)
        cache_val = Path(args.cache_val) if args.cache_val else None
        X_val = extract_fisher_vectors(val_paths, cfg, gmm, cache_npz=cache_val)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro")
        print(f"[val] images={len(val_paths)} acc={acc:.4f} macro_f1={macro_f1:.4f}")

        target_names = [inv_mapping[i] for i in range(len(inv_mapping))]
        print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))

        metrics = {"val_acc": float(acc), "val_macro_f1": float(macro_f1)}

    artifact = {
        "model": model,
        "fisher_config": cfg,
        "class_to_idx": mapping,
        "idx_to_class": inv_mapping,
        "gmm": gmm,
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
    cfg: FisherConfig = artifact["fisher_config"]
    idx_to_class: Dict[int, str] = artifact["idx_to_class"]
    gmm: GaussianMixture = artifact["gmm"]

    paths = list_images_recursive(input_dir)
    if not paths:
        raise ValueError(f"No images found under {input_dir}")

    X = extract_fisher_vectors(paths, cfg, gmm, cache_npz=None)
    y_pred = model.predict(X)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "pred_label"])
        for i, p in enumerate(paths):
            writer.writerow([p.name, idx_to_class[int(y_pred[i])]])

    print(f"[ok] wrote: {out_csv} (n={len(paths)})")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fisher Vector + Linear Classifier baseline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train and optionally evaluate on val")
    tr.add_argument("--train_dir", required=True, help="train root folder with class subfolders")
    tr.add_argument("--val_dir", default=None, help="val root folder with class subfolders (optional)")
    tr.add_argument("--out", required=True, help="output .joblib path")

    tr.add_argument("--img_size", type=int, default=256)
    tr.add_argument("--dense_step", type=int, default=8)
    tr.add_argument("--dense_size", type=int, default=12)
    tr.add_argument("--max_per_image", type=int, default=600)
    tr.add_argument("--no_rootsift", action="store_true")

    tr.add_argument("--n_components", type=int, default=64)
    tr.add_argument("--max_desc", type=int, default=200000, help="total descriptors to fit GMM")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--gmm_cache", default=None, help="path to joblib GMM cache")

    tr.add_argument("--classifier", choices=["linear_svm", "logreg"], default="linear_svm")
    tr.add_argument("--C", type=float, default=1.0)
    tr.add_argument("--max_iter", type=int, default=5000)
    tr.add_argument("--use_scaler", action="store_true", help="apply StandardScaler(with_mean=False)")

    tr.add_argument("--cache_train", default=None, help="npz cache for train fisher vectors")
    tr.add_argument("--cache_val", default=None, help="npz cache for val fisher vectors")

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

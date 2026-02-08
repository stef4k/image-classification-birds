import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

from birds_ml.config import Config
from birds_ml.data import Sample
from birds_ml.features import extract_embeddings
from birds_ml.utils import ensure_dir, set_seed


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROW_CLASSES = ["American_Crow", "Fish_Crow"]


@dataclass(frozen=True)
class WeightedSample:
    path: Path
    label: int
    weight: float
    source: str


def list_images_recursive(root: Path) -> List[Path]:
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return sorted(paths)


def normalize_class_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def resolve_class_dir(root: Path, class_name: str) -> Optional[Path]:
    if not root.exists() or not root.is_dir():
        return None
    exact = root / class_name
    if exact.exists() and exact.is_dir():
        return exact
    target = normalize_class_name(class_name)
    for p in root.iterdir():
        if p.is_dir() and normalize_class_name(p.name) == target:
            return p
    return None


def count_crow_like_files(root: Optional[Path]) -> int:
    if root is None or not root.exists() or not root.is_dir():
        return 0
    total = 0
    for cname in CROW_CLASSES:
        cdir = resolve_class_dir(root, cname)
        if cdir is not None:
            total += len(list_images_recursive(cdir))
    return total


def choose_pseudo_dir(cfg: Config, args) -> Optional[Path]:
    if args.pseudo_dir is not None:
        return args.pseudo_dir

    candidates = []
    if args.use_crops:
        candidates.append(cfg.data_dir / "pseudo_labels_cropped")
    candidates.append(cfg.data_dir / "pseudo_labels")

    best = None
    best_count = -1
    for cand in candidates:
        c = count_crow_like_files(cand)
        if c > best_count:
            best = cand
            best_count = c
    return best


def load_confidence_lookup(
    conf_csv: Optional[Path],
) -> Dict[str, tuple[str, float]]:
    if conf_csv is None or not conf_csv.exists():
        return {}
    df = pd.read_csv(conf_csv)
    required = {"path", "predicted_label", "confidence"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{conf_csv} must contain columns: {sorted(required)}. Found: {list(df.columns)}"
        )
    lookup = {}
    for _, row in df.iterrows():
        predicted = str(row["predicted_label"])
        conf = float(row["confidence"])
        keys = [str(row["path"])]
        if "relative_path" in row and pd.notna(row["relative_path"]):
            keys.append(str(row["relative_path"]))
        for k in list(keys):
            keys.append(Path(k).name)
        for key in keys:
            lookup[key] = (predicted, conf)
    return lookup


def load_real_samples(split_dir: Path, class_to_idx: Dict[str, int], weight: float) -> List[WeightedSample]:
    samples: List[WeightedSample] = []
    for cname, idx in class_to_idx.items():
        cdir = resolve_class_dir(split_dir, cname)
        if cdir is None:
            continue
        for p in list_images_recursive(cdir):
            samples.append(WeightedSample(path=p, label=idx, weight=weight, source="real"))
    return samples


def load_pseudo_samples(
    pseudo_dir: Optional[Path],
    class_to_idx: Dict[str, int],
    conf_lookup: Dict[str, tuple[str, float]],
    pseudo_min_conf: float,
    pseudo_weight: float,
    pseudo_ignore_conf_filter: bool,
) -> List[WeightedSample]:
    if pseudo_dir is None or not pseudo_dir.exists():
        return []

    samples: List[WeightedSample] = []
    for cname, idx in class_to_idx.items():
        cdir = resolve_class_dir(pseudo_dir, cname)
        if cdir is None:
            continue

        for p in list_images_recursive(cdir):
            if pseudo_ignore_conf_filter:
                sample_weight = pseudo_weight
            else:
                conf_row = conf_lookup.get(p.name)
                if conf_row is None:
                    conf_row = conf_lookup.get(str(p))
                if conf_row is None:
                    rel = p.relative_to(pseudo_dir)
                    conf_row = conf_lookup.get(str(rel))
                if conf_row is not None:
                    predicted_label, confidence = conf_row
                    if normalize_class_name(predicted_label) != normalize_class_name(cname) or confidence < pseudo_min_conf:
                        continue
                    sample_weight = pseudo_weight * confidence
                else:
                    sample_weight = pseudo_weight

            samples.append(
                WeightedSample(
                    path=p,
                    label=idx,
                    weight=float(sample_weight),
                    source="pseudo",
                )
            )
    return samples


def weighted_to_samples(weighted: List[WeightedSample]) -> List[Sample]:
    return [Sample(path=s.path, label=s.label) for s in weighted]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    acc = float((y_true == y_pred).mean()) if y_true.size > 0 else 0.0
    bal = float(balanced_accuracy_score(y_true, y_pred)) if y_true.size > 0 else 0.0
    per_class = []
    for idx in range(len(CROW_CLASSES)):
        mask = y_true == idx
        class_acc = float((y_pred[mask] == idx).mean()) if np.any(mask) else 0.0
        per_class.append(class_acc)
    return {
        "acc": acc,
        "balanced_acc": bal,
        "acc_american_crow": per_class[0],
        "acc_fish_crow": per_class[1],
    }


def maybe_apply_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_secondary: Optional[np.ndarray],
    pca_dim: int,
):
    if pca_dim <= 0:
        return X_train, X_val, X_secondary, None
    max_dim = min(X_train.shape[0], X_train.shape[1])
    if pca_dim > max_dim:
        pca_dim = max_dim
    if pca_dim <= 0:
        return X_train, X_val, X_secondary, None

    pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=42)
    Xt = pca.fit_transform(X_train)
    Xv = pca.transform(X_val)
    Xs = pca.transform(X_secondary) if X_secondary is not None else None
    return Xt, Xv, Xs, pca


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["knn", "rbf_svm", "logreg"], required=True)
    ap.add_argument("--backbone", default="eva02_large_448")
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--use_crops", action="store_true")
    ap.add_argument("--train_dir", type=Path, default=None)
    ap.add_argument("--val_dir", type=Path, default=Path("data/f_images_cropped"))
    ap.add_argument("--secondary_val_dir", type=Path, default=Path("data/val_images_cropped"))
    ap.add_argument("--selection_metric", choices=["acc", "balanced_acc"], default="balanced_acc")
    ap.add_argument("--real_weight", type=float, default=1.0)
    ap.add_argument("--pseudo_weight", type=float, default=0.0)
    ap.add_argument("--pseudo_min_conf", type=float, default=0.95)
    ap.add_argument(
        "--pseudo_ignore_conf_filter",
        action="store_true",
        help="Use pseudo labels from folder names directly and ignore confidence CSV filtering.",
    )
    ap.add_argument("--pseudo_dir", type=Path, default=None)
    ap.add_argument(
        "--pseudo_conf_csv",
        type=Path,
        default=Path("outputs/extra_images_cv_ensemble_predictions_with_conf.csv"),
    )
    ap.add_argument("--pca_dim", type=int, default=0)
    ap.add_argument("--run_name", default=None)

    ap.add_argument("--knn_k", type=int, default=9)
    ap.add_argument("--knn_metric", choices=["cosine", "euclidean"], default="cosine")
    ap.add_argument("--knn_weights", choices=["uniform", "distance"], default="distance")
    ap.add_argument("--knn_weight_mode", choices=["none", "replicate"], default="none")
    ap.add_argument("--knn_rep_scale", type=float, default=4.0)
    ap.add_argument("--knn_rep_max", type=int, default=8)

    ap.add_argument("--svm_c", type=float, default=3.0)
    ap.add_argument("--svm_gamma", default="scale")
    ap.add_argument("--logreg_c", type=float, default=1.0)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = replace(Config(), use_crops=args.use_crops)
    ensure_dir(cfg.outputs_dir)
    set_seed(cfg.seed)

    class_to_idx = {name: i for i, name in enumerate(CROW_CLASSES)}
    train_dir = args.train_dir or cfg.train_dir
    val_dir = args.val_dir
    secondary_val_dir = args.secondary_val_dir

    pseudo_dir = choose_pseudo_dir(cfg, args)
    conf_lookup = load_confidence_lookup(args.pseudo_conf_csv)

    train_weighted = load_real_samples(train_dir, class_to_idx, weight=args.real_weight)
    pseudo_weighted = load_pseudo_samples(
        pseudo_dir=pseudo_dir,
        class_to_idx=class_to_idx,
        conf_lookup=conf_lookup,
        pseudo_min_conf=args.pseudo_min_conf,
        pseudo_weight=args.pseudo_weight,
        pseudo_ignore_conf_filter=bool(args.pseudo_ignore_conf_filter),
    )
    train_weighted.extend(pseudo_weighted)
    val_weighted = load_real_samples(val_dir, class_to_idx, weight=1.0)
    secondary_weighted = load_real_samples(secondary_val_dir, class_to_idx, weight=1.0)

    if not train_weighted:
        raise ValueError("No training samples found.")
    if not val_weighted:
        raise ValueError(f"No validation samples found under {val_dir}")
    if not secondary_weighted:
        raise ValueError(f"No secondary validation samples found under {secondary_val_dir}")

    print(f"Train dir: {train_dir}")
    print(f"Val dir: {val_dir}")
    print(f"Pseudo dir: {pseudo_dir}")
    print(f"Secondary val dir: {secondary_val_dir}")
    print(f"Train samples: {len(train_weighted)} | Pseudo included: {len(pseudo_weighted)}")

    train_samples = weighted_to_samples(train_weighted)
    val_samples = weighted_to_samples(val_weighted)
    secondary_samples = weighted_to_samples(secondary_weighted)

    X_train, y_train, _ = extract_embeddings(
        train_samples,
        backbone=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=cfg.device,
    )
    X_val, y_val, _ = extract_embeddings(
        val_samples,
        backbone=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=cfg.device,
    )
    X_secondary, y_secondary, _ = extract_embeddings(
        secondary_samples,
        backbone=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=cfg.device,
    )

    sample_weights = np.asarray([s.weight for s in train_weighted], dtype=np.float32)

    if args.method == "knn" and args.knn_metric == "cosine":
        X_train = normalize(X_train, norm="l2")
        X_val = normalize(X_val, norm="l2")
        X_secondary = normalize(X_secondary, norm="l2")

    X_train, X_val, X_secondary, pca = maybe_apply_pca(
        X_train=X_train,
        X_val=X_val,
        X_secondary=X_secondary,
        pca_dim=args.pca_dim,
    )

    if args.method == "knn":
        X_fit = X_train
        y_fit = y_train
        if args.knn_weight_mode == "replicate":
            reps = np.clip(
                np.rint(sample_weights * args.knn_rep_scale).astype(np.int32),
                1,
                args.knn_rep_max,
            )
            X_fit = np.repeat(X_train, reps, axis=0)
            y_fit = np.repeat(y_train, reps, axis=0)

        model = KNeighborsClassifier(
            n_neighbors=args.knn_k,
            metric=args.knn_metric,
            weights=args.knn_weights,
        )
        model.fit(X_fit, y_fit)
        pred_val = model.predict(X_val)
        pred_secondary = model.predict(X_secondary)
    elif args.method == "rbf_svm":
        model = SVC(
            C=args.svm_c,
            kernel="rbf",
            gamma=args.svm_gamma,
            class_weight="balanced",
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        pred_val = model.predict(X_val)
        pred_secondary = model.predict(X_secondary)
    else:
        model = LogisticRegression(
            C=args.logreg_c,
            max_iter=8000,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        pred_val = model.predict(X_val)
        pred_secondary = model.predict(X_secondary)

    val_metrics = compute_metrics(y_val, pred_val)
    secondary_metrics = compute_metrics(y_secondary, pred_secondary)

    run_name = args.run_name or f"crowsep_emb_{args.method}_{args.backbone}_{args.img_size}"
    model_path = cfg.outputs_dir / f"{run_name}.joblib"
    meta_path = cfg.outputs_dir / f"meta_{run_name}.json"

    payload = {
        "method": args.method,
        "backbone": args.backbone,
        "img_size": args.img_size,
        "selection_metric": args.selection_metric,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "secondary_val_dir": str(secondary_val_dir),
        "real_weight": args.real_weight,
        "pseudo_weight": args.pseudo_weight,
        "pseudo_min_conf": args.pseudo_min_conf,
        "pseudo_dir": str(pseudo_dir),
        "pseudo_count_used": int(len(pseudo_weighted)),
        "pca_dim": int(args.pca_dim),
        "knn_k": args.knn_k,
        "knn_metric": args.knn_metric,
        "knn_weights": args.knn_weights,
        "knn_weight_mode": args.knn_weight_mode,
        "svm_c": args.svm_c,
        "svm_gamma": args.svm_gamma,
        "logreg_c": args.logreg_c,
        "val_metrics": val_metrics,
        "secondary_val_metrics": secondary_metrics,
        "best_selection_metric": float(
            val_metrics["acc"] if args.selection_metric == "acc" else val_metrics["balanced_acc"]
        ),
        "best_val_acc": float(val_metrics["acc"]),
        "best_balanced_acc": float(val_metrics["balanced_acc"]),
        "best_val_acc_american_crow": float(val_metrics["acc_american_crow"]),
        "best_val_acc_fish_crow": float(val_metrics["acc_fish_crow"]),
        "best_secondary_val_acc": float(secondary_metrics["acc"]),
        "best_secondary_balanced_acc": float(secondary_metrics["balanced_acc"]),
        "best_secondary_val_acc_american_crow": float(secondary_metrics["acc_american_crow"]),
        "best_secondary_val_acc_fish_crow": float(secondary_metrics["acc_fish_crow"]),
        "run_name": run_name,
    }
    if pca is not None:
        payload["pca_explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))

    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(payload, indent=2))

    print(
        f"Method={args.method} | Val Acc={val_metrics['acc']:.4f} | "
        f"Val Balanced={val_metrics['balanced_acc']:.4f} | "
        f"Secondary Acc={secondary_metrics['acc']:.4f} | "
        f"Secondary Balanced={secondary_metrics['balanced_acc']:.4f}"
    )
    print(f"Saved model: {model_path}")
    print(f"Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()

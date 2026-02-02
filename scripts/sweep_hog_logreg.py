#!/usr/bin/env python3
"""
Quick grid sweep for HOG + Logistic Regression.

Defaults to cropped train folder and reports 10-fold CV average accuracy.
"""

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from train_hog_logreg import (
    HogConfig,
    build_label_mapping_from_folders,
    build_model,
    extract_features,
    load_labeled_paths,
)


def grid(values: Iterable) -> List:
    return list(values)


def make_cache_path(cache_dir: Path, cfg: HogConfig, use_color_hist: bool, C: float, max_iter: int) -> Path:
    key = (
        f"hog_is{cfg.img_size}_ori{cfg.orientations}_ppc{cfg.pixels_per_cell}"
        f"_cpb{cfg.cells_per_block}_col{int(use_color_hist)}_bins{cfg.color_hist_bins}"
        f"_C{C:g}_mi{max_iter}"
    )
    return cache_dir / f"{key}.npz"


def eval_cv_accuracy(
    X: np.ndarray, y: np.ndarray, C: float, max_iter: int, n_jobs: int, folds: int, seed: int
) -> float:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs: List[float] = []
    for tr, te in skf.split(X, y):
        model = build_model(C=C, max_iter=max_iter, n_jobs=n_jobs)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        acc = float((pred == y[te]).mean())
        accs.append(acc)
    return float(np.mean(accs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid sweep for HOG+LogReg (5-fold CV avg acc)")
    ap.add_argument("--train_dir", default="data/train_images_cropped", help="train root with class subfolders")
    ap.add_argument("--out_csv", default="outputs/hog_logreg_sweep.csv", help="output CSV")
    ap.add_argument("--cache_dir", default="outputs/hog_cache", help="npz cache directory")
    ap.add_argument("--folds", type=int, default=5, help="number of CV folds")
    ap.add_argument("--seed", type=int, default=42, help="random seed for CV splits")
    ap.add_argument("--max_iter", type=int, default=3000, help="logreg max_iter")
    ap.add_argument("--n_jobs", type=int, default=7, help="logreg n_jobs (lbfgs ignores >1)")
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    out_csv = Path(args.out_csv)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Parameter grid (edit if you want larger search)
    img_sizes = grid([128, 192, 256])
    orientations = grid([9, 12])
    pixels_per_cell = grid([8, 12])
    cells_per_block = grid([2, 3])
    Cs = grid([2.0, 4.0, 8.0, 16.0])
    use_color_hist = grid([False, True])
    color_hist_bins = grid([16, 32])

    mapping = build_label_mapping_from_folders(train_dir)
    train_paths, y = load_labeled_paths(train_dir, mapping)

    results: List[Tuple[float, HogConfig, float, bool, int]] = []

    total = (
        len(img_sizes)
        * len(orientations)
        * len(pixels_per_cell)
        * len(cells_per_block)
        * len(Cs)
        * len(use_color_hist)
        * len(color_hist_bins)
    )
    idx = 0
    for isz in img_sizes:
        for ori in orientations:
            for ppc in pixels_per_cell:
                for cpb in cells_per_block:
                    for col in use_color_hist:
                        for bins in color_hist_bins:
                            cfg = HogConfig(
                                img_size=isz,
                                orientations=ori,
                                pixels_per_cell=ppc,
                                cells_per_block=cpb,
                                use_color_hist=col,
                                color_hist_bins=bins,
                            )
                            for C in Cs:
                                idx += 1
                                cache_npz = make_cache_path(cache_dir, cfg, col, C, args.max_iter)
                                print(f"[{idx}/{total}] cfg=img{isz} ori{ori} ppc{ppc} cpb{cpb} "
                                      f"col{int(col)} bins{bins} C{C:g}")
                                X = extract_features(train_paths, cfg, cache_npz=cache_npz)
                                acc = eval_cv_accuracy(X, y, C=C, max_iter=args.max_iter, n_jobs=args.n_jobs,
                                                       folds=args.folds, seed=args.seed)
                                results.append((acc, cfg, C, col, bins))
                                print(f"  -> cv_avg_acc={acc:.4f}")

    results.sort(key=lambda x: x[0], reverse=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cv_avg_acc",
            "img_size",
            "orientations",
            "pixels_per_cell",
            "cells_per_block",
            "use_color_hist",
            "color_hist_bins",
            "C",
            "max_iter",
            "folds",
        ])
        for acc, cfg, C, col, bins in results:
            writer.writerow([
                f"{acc:.6f}",
                cfg.img_size,
                cfg.orientations,
                cfg.pixels_per_cell,
                cfg.cells_per_block,
                int(col),
                bins,
                f"{C:g}",
                args.max_iter,
                args.folds,
            ])

    best = results[0]
    best_acc, best_cfg, best_C, best_col, best_bins = best
    print("\nBest config:")
    print(
        f"  acc={best_acc:.4f} img={best_cfg.img_size} ori={best_cfg.orientations} "
        f"ppc={best_cfg.pixels_per_cell} cpb={best_cfg.cells_per_block} "
        f"col={int(best_col)} bins={best_bins} C={best_C:g}"
    )
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()

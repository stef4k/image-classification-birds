from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@dataclass(frozen=True)
class Sample:
    path: Path
    label: Optional[int]  # None for test

def _list_images_recursive(root: Path) -> List[Path]:
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return sorted(paths)

def load_trainval_from_folders(split_dir: Path) -> Tuple[List[Sample], Dict[str, int]]:
    """
    split_dir like data/train_images where each subfolder is a class name.
    Returns samples with numeric labels + class_to_idx mapping.
    """
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing directory: {split_dir}")

    class_names = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class folders found under: {split_dir}")

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    samples: List[Sample] = []

    for cname in class_names:
        cdir = split_dir / cname
        for img_path in _list_images_recursive(cdir):
            samples.append(Sample(path=img_path, label=class_to_idx[cname]))

    return samples, class_to_idx

def load_val_with_given_mapping(val_dir: Path, class_to_idx: Dict[str, int]) -> List[Sample]:
    if not val_dir.exists():
        raise FileNotFoundError(f"Missing directory: {val_dir}")

    samples: List[Sample] = []
    for cname, idx in class_to_idx.items():
        cdir = val_dir / cname
        if not cdir.exists():
            # Some datasets can have missing classes in val; tolerate.
            continue
        for img_path in _list_images_recursive(cdir):
            samples.append(Sample(path=img_path, label=idx))
    return samples

def load_test_recursive(test_dir: Path) -> List[Sample]:
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing directory: {test_dir}")
    return [Sample(path=p, label=None) for p in _list_images_recursive(test_dir)]

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    project_root: Path = Path(".")
    data_dir: Path = Path("data")

    train_dir: Path = Path("data/train_images")
    val_dir: Path = Path("data/val_images")
    test_dir: Path = Path("data/test_images")

    outputs_dir: Path = Path("outputs")
    cache_dir: Path = Path("outputs/cache")

    backbone: str = "resnet50"  # "resnet50" or "efficientnet_b0"
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda"  # will fallback to cpu if no cuda

    cv_folds: int = 5
    seed: int = 42
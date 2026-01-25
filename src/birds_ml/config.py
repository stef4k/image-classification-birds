from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    project_root: Path = Path(".")
    data_dir: Path = Path("data")

    use_crops: bool = False

    outputs_dir: Path = Path("outputs")
    cache_dir: Path = Path("outputs/cache")

    backbone: str = "resnet50"  # "resnet50" or "efficientnet_b0"
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda"  # will fallback to cpu if no cuda

    cv_folds: int = 5
    seed: int = 42

    @property
    def train_dir(self) -> Path:
        suffix = "_cropped" if self.use_crops else ""
        return self.data_dir / f"train_images{suffix}"

    @property
    def val_dir(self) -> Path:
        suffix = "_cropped" if self.use_crops else ""
        return self.data_dir / f"val_images{suffix}"

    @property
    def test_dir(self) -> Path:
        suffix = "_cropped" if self.use_crops else ""
        return self.data_dir / f"test_images{suffix}"
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from .data import Sample
from .embedder import build_backbone

class SampleDataset(Dataset):
    def __init__(self, samples: List[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.transform(img)
        y = -1 if s.label is None else int(s.label)
        return x, y, str(s.path)

def build_embedding_transforms(
    img_size: int = 224,
    resize_train: int = 256,
    crop_scale_min: float = 0.7,
    crop_scale_max: float = 1.0,
    rotation_deg: float = 30.0,
    hflip_p: float = 0.5,
):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    standard = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    augment = transforms.Compose([
        transforms.Resize((resize_train, resize_train)),
        transforms.RandomResizedCrop(
            img_size,
            scale=(crop_scale_min, crop_scale_max),
        ),
        transforms.RandomHorizontalFlip(p=hflip_p),
        transforms.RandomRotation(rotation_deg),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return standard, augment

@torch.inference_mode()
def extract_embeddings(
    samples: List[Sample],
    backbone: str,
    batch_size: int,
    num_workers: int,
    device: str,
    transform=None,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    model, default_transform = build_backbone(backbone)
    model.eval()

    if transform is None:
        transform = default_transform

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(dev)

    ds = SampleDataset(samples, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=(dev.type == "cuda"))

    feats, labels, paths = [], [], []
    for xb, yb, pb in tqdm(dl, desc="Extracting embeddings"):
        xb = xb.to(dev, non_blocking=True)
        fb = model(xb).detach().cpu().numpy().astype(np.float32)
        feats.append(fb)
        paths.extend(list(pb))
        labels.append(yb.numpy().astype(np.int64))

    X = np.concatenate(feats, axis=0)
    y_all = np.concatenate(labels, axis=0)

    # If this is test set, labels are -1; return None for y
    if np.all(y_all == -1):
        return X, None, paths
    return X, y_all, paths

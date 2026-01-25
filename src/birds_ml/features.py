from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

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

@torch.inference_mode()
def extract_embeddings(
    samples: List[Sample],
    backbone: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    model, transform = build_backbone(backbone)
    model.eval()

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

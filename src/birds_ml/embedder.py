import torch.nn as nn
from torchvision import models

def build_backbone(name: str):
    name = name.lower()
    if name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT
        m = models.resnet50(weights=w)
        m.fc = nn.Identity()  # 2048-d embedding
        return m, w.transforms()

    if name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Identity()  # 1280-d embedding
        return m, w.transforms()

    raise ValueError(f"Unsupported backbone: {name}")

import torch.nn as nn
import timm

def build_backbone(name: str):
    # when using "convnext_base", we use the larger 22k version
    if name == "convnext_base":
        print(f"Loading TIMM model: convnext_base.fb_in22k_ft_in1k")
        # num_classes=0 removes the head and returns the feature vector
        m = timm.create_model(
            'convnext_base.fb_in22k_ft_in1k', 
            pretrained=True, 
            num_classes=0
        )
        # return model and its specific config (mean/std/input_size)
        return m, timm.data.resolve_data_config({}, model=m)

    if name == "resnet50":
        m = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0)
        return m, timm.data.resolve_data_config({}, model=m)
    
    if name == "efficientnet_b0":
        m = timm.create_model('efficientnet_b0.ra_in1k', pretrained=True, num_classes=0)
        return m, timm.data.resolve_data_config({}, model=m)

    raise ValueError(f"Unsupported backbone: {name}")
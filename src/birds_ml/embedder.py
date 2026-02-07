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
    
    if name == "convnextv2_large_384":
        print(f"Loading TIMM model: convnextv2_large.fcmae_ft_in22k_in1k_384")
        m = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=0)
        return m, timm.data.resolve_data_config({}, model=m)
    
    if name == "convnextv2_base_384":
        print(f"Loading TIMM model: convnextv2_base.fcmae_ft_in22k_in1k_384")
        m = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=0)
        return m, timm.data.resolve_data_config({}, model=m)
    
    if name == "vit_so150m2_384":
        print(f"Loading TIMM model: vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k")
        m = timm.create_model(
            'vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k', 
            pretrained=True, 
            num_classes=0
        )
        return m, timm.data.resolve_data_config({}, model=m)
    
    if name == "eva02_large_448":
        print(f"Loading TIMM model: eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")
        m = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True, num_classes=0)
        return m, timm.data.resolve_data_config({}, model=m)
    
    if name == "caformer_b36_384":
        print(f"Loading TIMM model: caformer_b36.sail_in22k_ft_in1k_384")
        m = timm.create_model('caformer_b36.sail_in22k_ft_in1k_384', pretrained=True, num_classes=0)
        return m, timm.data.resolve_data_config({}, model=m)

    if name == "resnet50":
        m = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0)
        return m, timm.data.resolve_data_config({}, model=m)
    
    if name == "efficientnet_b0":
        m = timm.create_model('efficientnet_b0.ra_in1k', pretrained=True, num_classes=0)
        return m, timm.data.resolve_data_config({}, model=m)

    raise ValueError(f"Unsupported backbone: {name}")
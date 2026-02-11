import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataclasses import replace
from torch.cuda.amp import GradScaler, autocast

from birds_ml.config import Config
from birds_ml.data import load_trainval_from_folders, load_val_with_given_mapping
from birds_ml.features import SampleDataset
from birds_ml.embedder import build_backbone
# from birds_ml.head import CustomHead
from birds_ml.utils import set_seed, ensure_dir
from birds_ml.arcface_margin import ArcMarginProduct

import torch.nn.functional as F
from torchvision.transforms import functional as TF

class SquarePad:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # resize so longest edge = target_size
        w, h = img.size
        max_wh = max(w, h)
        scale = self.target_size / max_wh
        new_w, new_h = int(w * scale), int(h * scale)
        img = TF.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)
        
        # pad to make it square
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        pad_left = delta_w // 2
        pad_right = delta_w - pad_left
        pad_top = delta_h // 2
        pad_bottom = delta_h - pad_top
        
        # fill with gray (128)
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=128, padding_mode='constant')

def main():
    cfg = Config()
    ensure_dir(cfg.outputs_dir)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0") 
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--use_arcface", action="store_true", help="Use ArcFace head instead of Linear")
    args = parser.parse_args()
    
    cfg = replace(cfg, use_crops=args.use_crops)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # TIMM Model
    backbone_model, model_config = build_backbone(args.backbone)
    backbone_model.to(device)
    
    # freezing the backbone (tried partial freezing but full freeze worked best and it's faster)
    backbone_model.eval() 
    for param in backbone_model.parameters():
        param.requires_grad = False

    head_type = "ArcFace" if args.use_arcface else "Linear"
    print(f"Training {head_type} | Backbone: {args.backbone}")

    # transforms (SquarePad)
    train_tfm = transforms.Compose([
        # letterbox resize (no squashing, this led to better results)
        SquarePad(args.img_size),
        
        # standard aug
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=model_config['mean'], std=model_config['std'])
    ])
    
    val_tfm = transforms.Compose([
        # letterbox resize
        SquarePad(args.img_size),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=model_config['mean'], std=model_config['std'])
    ])

    # data
    train_samples, class_to_idx = load_trainval_from_folders(cfg.train_dir)
    val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)

    # with batch_size=32 it runs on a RTX 3050 6GB Laptop GPU just fine even with 448x448
    train_dl = DataLoader(SampleDataset(train_samples, train_tfm), batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(SampleDataset(val_samples, val_tfm), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # head
    input_dim = backbone_model.num_features
    
    if args.use_arcface:
        # SOTA ArcFace Head
        head = ArcMarginProduct(input_dim, len(class_to_idx), s=30.0, m=0.50).to(device)
    else:
        # standard Linear Head
        head = nn.Linear(input_dim, len(class_to_idx)).to(device)
        nn.init.constant_(head.bias, 0)
        nn.init.normal_(head.weight, std=0.01)

    optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    best_acc = 0.0
    crop_suffix = "_cropped" if cfg.use_crops else ""
    # Distinguish model names so they don't overwrite each other
    head_prefix = "arcface" if args.use_arcface else "linear"
    model_name = f"{head_prefix}_{args.backbone}{crop_suffix}_{args.img_size}"
    
    for epoch in range(args.epochs):
        head.train()
        train_loss = 0.0
        
        for imgs, labels, _ in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                # no need for the gradients for the backbone
                with torch.no_grad():
                    feats = backbone_model(imgs)
                
                if args.use_arcface:
                    # ArcFace needs labels to compute margin
                    preds = head(feats, labels)
                else:
                    # linear just maps features to logits
                    preds = head(feats)
                
                loss = criterion(preds, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # validation
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels, _ in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                
                with autocast():
                    feats = backbone_model(imgs)
                    
                    # inference
                    if args.use_arcface:
                        # Cosine Similarity (No Margin)
                        norm_feats = F.normalize(feats)
                        norm_weights = F.normalize(head.weight)
                        logits = F.linear(norm_feats, norm_weights) * head.s
                    else:
                        # Standard Linear Forward
                        logits = head(feats)
                    
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        avg_loss = train_loss / len(train_dl)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
        
        if acc >= best_acc:
            best_acc = acc
            # save checkpoint (backbone + head + config)
            torch.save({
                'backbone': backbone_model.state_dict(),
                'head': head.state_dict(),
                'config': model_config
            }, cfg.outputs_dir / f"{model_name}.pth")

            # meta
            meta = {
                "kind": "arcface_frozen" if args.use_arcface else "timm_frozen",
                "backbone": args.backbone,
                "img_size": args.img_size,
                "class_to_idx": class_to_idx,
                "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
                "best_acc": best_acc
            }
            (cfg.outputs_dir / f"meta_{model_name}.json").write_text(json.dumps(meta, indent=2))
    
    print(f"Finished. Best Val Acc: {best_acc:.4f}")
    print(f"Saved Metadata: meta_{model_name}.json")

if __name__ == "__main__":
    main()
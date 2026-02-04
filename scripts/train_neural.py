import argparse
import json
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

def _iter_blocks(backbone_model):
    if hasattr(backbone_model, "blocks"):
        return list(backbone_model.blocks)
    if hasattr(backbone_model, "stages"):
        blocks = []
        for stage in backbone_model.stages:
            if hasattr(stage, "blocks"):
                blocks.extend(list(stage.blocks))
            else:
                blocks.append(stage)
        return blocks
    if hasattr(backbone_model, "layers"):
        return list(backbone_model.layers)
    if hasattr(backbone_model, "features"):
        return list(backbone_model.features)
    return list(backbone_model.children())

def _unfreeze_last_blocks(backbone_model, n_blocks: int):
    if n_blocks <= 0:
        return
    blocks = _iter_blocks(backbone_model)
    if not blocks:
        return
    for block in blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

def _build_class_weights(class_to_idx, crow_a, crow_b, crow_weight, device):
    weights = torch.ones(len(class_to_idx), device=device)
    for cname in (crow_a, crow_b):
        if cname in class_to_idx:
            weights[class_to_idx[cname]] = crow_weight
    return weights

def _eval(backbone_model, head, val_dl, device):
    backbone_model.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, _ in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(enabled=(device.type == "cuda")):
                feats = backbone_model(imgs)
                preds = head(feats)
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0

def main():
    cfg = Config()
    ensure_dir(cfg.outputs_dir)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0") 
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=8)
    parser.add_argument("--finetune_blocks", type=int, default=2)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--crow_a", type=str, default="American_Crow")
    parser.add_argument("--crow_b", type=str, default="Fish_Crow")
    parser.add_argument("--crow_weight", type=float, default=2.0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--use_crops", action="store_true")
    args = parser.parse_args()
    
    cfg = replace(cfg, use_crops=args.use_crops)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # TIMM Model
    backbone_model, model_config = build_backbone(args.backbone)
    backbone_model.to(device)
    
    # Phase A: freeze backbone and train head
    backbone_model.eval()
    for param in backbone_model.parameters():
        param.requires_grad = False

    print(f"Training TIMM Model | Backbone: {args.backbone}")
    print(f"Using Stats: {model_config['mean']}, {model_config['std']}")

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
    
    train_dl = DataLoader(
        SampleDataset(train_samples, train_tfm), 
        batch_size=32, 
        shuffle=True, 
        num_workers=0, # fixed for Windows, ran into issues with multiprocessing
        pin_memory=True
    )
    val_dl = DataLoader(
        SampleDataset(val_samples, val_tfm), 
        batch_size=32, 
        shuffle=False,  
        num_workers=0,
        pin_memory=True
    )

    # head
    input_dim = backbone_model.num_features
    head = nn.Linear(input_dim, len(class_to_idx)).to(device)
    
    # initialize it properly
    nn.init.constant_(head.bias, 0)
    nn.init.normal_(head.weight, std=0.01)
    
    class_weights = _build_class_weights(
        class_to_idx, args.crow_a, args.crow_b, args.crow_weight, device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_acc = 0.0
    crop_suffix = "_cropped" if cfg.use_crops else ""
    model_name = f"timm_finetune_{args.backbone}{crop_suffix}_{args.img_size}"
    best_state = None

    # Phase A: head only
    optimizer = optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    patience_left = args.patience
    
    for epoch in range(args.epochs):
        head.train()
        train_loss = 0.0
        
        for imgs, labels, _ in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast(enabled=(device.type == "cuda")):
                # no need for the gradients for the backbone
                with torch.no_grad():
                    feats = backbone_model(imgs)
                
                preds = head(feats)
                loss = criterion(preds, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        acc = _eval(backbone_model, head, val_dl, device)
        avg_loss = train_loss / len(train_dl)
        print(f"[Head] Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
        
        if acc >= best_acc:
            best_acc = acc
            best_state = {
                "backbone": {k: v.detach().cpu() for k, v in backbone_model.state_dict().items()},
                "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "config": model_config,
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stop triggered (head phase).")
                break

    # Phase B: unfreeze last blocks
    if args.finetune_epochs > 0 and args.finetune_blocks > 0:
        _unfreeze_last_blocks(backbone_model, args.finetune_blocks)
        backbone_model.train()
        head.train()

        optimizer = optim.AdamW(
            [
                {"params": head.parameters(), "lr": args.head_lr},
                {
                    "params": [p for p in backbone_model.parameters() if p.requires_grad],
                    "lr": args.backbone_lr,
                },
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
        patience_left = args.patience

        for epoch in range(args.finetune_epochs):
            train_loss = 0.0
            for imgs, labels, _ in train_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast(enabled=(device.type == "cuda")):
                    feats = backbone_model(imgs)
                    preds = head(feats)
                    loss = criterion(preds, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            scheduler.step()

            acc = _eval(backbone_model, head, val_dl, device)
            avg_loss = train_loss / len(train_dl)
            print(f"[FT] Epoch {epoch+1}/{args.finetune_epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

            if acc >= best_acc:
                best_acc = acc
                best_state = {
                    "backbone": {k: v.detach().cpu() for k, v in backbone_model.state_dict().items()},
                    "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "config": model_config,
                }
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("Early stop triggered (finetune phase).")
                    break

    # save best checkpoint (backbone + head + config)
    if best_state is None:
        best_state = {
            "backbone": backbone_model.state_dict(),
            "head": head.state_dict(),
            "config": model_config,
        }
    torch.save(best_state, cfg.outputs_dir / f"{model_name}.pth")

    # meta for prediction
    meta = {
        "kind": "timm_finetune",
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

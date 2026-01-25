import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataclasses import replace

from birds_ml.config import Config
from birds_ml.data import load_trainval_from_folders, load_val_with_given_mapping
from birds_ml.features import SampleDataset
from birds_ml.embedder import build_backbone
from birds_ml.head import CustomHead
from birds_ml.utils import set_seed, ensure_dir

def main():
    cfg = Config()
    ensure_dir(cfg.outputs_dir)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation (Flip, Rotate, Crop)")
    args = parser.parse_args()
    
    cfg = replace(cfg, use_crops=args.use_crops)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    print(f"Training Neural Head | Backbone: {args.backbone} | Crops: {cfg.use_crops} | Augment: {args.augment}")

    # standard Preprocessing (Used for Val and Non-Augmented Train)
    standard_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.augment:
        # data augmentation
        train_tfm = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_tfm = standard_tfm

    print(f"Loading TRAIN data from: {cfg.train_dir}")
    train_samples, class_to_idx = load_trainval_from_folders(cfg.train_dir)
    
    print(f"Loading VAL data from: {cfg.val_dir}")
    val_samples = load_val_with_given_mapping(cfg.val_dir, class_to_idx)
    
    print(f"Dataset Size -> Train: {len(train_samples)} | Val: {len(val_samples)}")
    
    train_ds = SampleDataset(train_samples, train_tfm)
    val_ds = SampleDataset(val_samples, standard_tfm)
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=cfg.num_workers)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=cfg.num_workers)

    # model
    backbone_model, _ = build_backbone(args.backbone) 
    backbone_model.to(device)
    backbone_model.eval() # freeze backbone features
    
    input_dim = 2048 if args.backbone == "resnet50" else 1280
    
    # head with Dropout (augmentation usually pairs well with Dropout)
    dropout = 0.7 if args.augment else 0.5
    head = CustomHead(input_dim, 512, len(class_to_idx), dropout_prob=dropout).to(device)
    
    optimizer = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # tranining
    best_acc = 0.0
    crop_suffix = "_cropped" if cfg.use_crops else ""
    aug_suffix = "_aug" if args.augment else ""
    model_name = f"neural_{args.backbone}{crop_suffix}{aug_suffix}"
    
    for epoch in range(args.epochs):
        head.train()
        train_loss = 0.0
        
        # Train Step
        for imgs, labels, _ in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            
            with torch.no_grad():
                feats = backbone_model(imgs)
            
            preds = head(feats)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation Step
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels, _ in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = backbone_model(imgs)
                preds = head(feats)
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        avg_loss = train_loss / len(train_dl)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(head.state_dict(), cfg.outputs_dir / f"{model_name}.pth")

    # metadata
    meta = {
        "kind": "neural",
        "backbone": args.backbone,
        "use_crops": cfg.use_crops,
        "augment": args.augment,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
        "best_acc": best_acc
    }
    (cfg.outputs_dir / f"meta_{model_name}.json").write_text(json.dumps(meta, indent=2))
    
    print(f"Finished. Best Val Acc: {best_acc:.4f}")
    print(f"Saved model: outputs/{model_name}.pth")

if __name__ == "__main__":
    main()
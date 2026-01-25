import torch
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image

from birds_ml.config import Config
from birds_ml.utils import ensure_dir

# COCO Class Index 16 is 'Bird'
BIRD_CLASS_ID = 16 

def load_detector(device):
    print("Loading pretrained Faster R-CNN (COCO)...")
    # automatically downloads ~170MB weights the first time
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model

def get_bird_box(model, img_path, device, confidence_threshold=0.6):
    """
    Returns (x, y, w, h) of the most confident bird detection.
    If no bird is found, returns None.
    """
    try:
        # load and preprocess
        img_pil = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img_pil.size
        img_tensor = F.to_tensor(img_pil).to(device)
        
        with torch.no_grad():
            predictions = model([img_tensor])[0]
        
        labels = predictions['labels']
        scores = predictions['scores']
        boxes = predictions['boxes']
        
        # Is it a bird with high confidence? (strict check)
        mask = (labels == BIRD_CLASS_ID) & (scores > confidence_threshold)
        bird_boxes = boxes[mask]
        bird_scores = scores[mask]
        
        # If strict fails, accept lower confidence (looser check)
        if len(bird_boxes) == 0:
            mask_loose = (labels == BIRD_CLASS_ID) & (scores > 0.3)
            bird_boxes = boxes[mask_loose]
            bird_scores = scores[mask_loose]
            
            # Last Resort: If NO bird found, return None
            if len(bird_boxes) == 0:
                return None

        # Pick the detection with the highest score
        best_idx = torch.argmax(bird_scores)
        box = bird_boxes[best_idx].cpu().numpy() # [x1, y1, x2, y2]
        
        x1, y1, x2, y2 = box
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def process_folder(model, src_dir: Path, dst_dir: Path, device):
    """
    Iterates through src_dir, finds birds, crops, and saves to dst_dir.
    Maintains folder structure.
    """
    if not src_dir.exists():
        print(f"Skipping {src_dir} (does not exist)")
        return

    ensure_dir(dst_dir)
    # find all images recursively
    images = sorted([
        p for p in src_dir.rglob("*") 
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    
    print(f"\nProcessing: {src_dir} -> {dst_dir}")
    print(f"Total images: {len(images)}")
    
    stats = {"cropped": 0, "original": 0}
    
    for img_path in tqdm(images):
        # calculate destination path
        rel_path = img_path.relative_to(src_dir)
        save_path = dst_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # detect
        bbox = get_bird_box(model, img_path, device)
        
        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            continue
            
        if bbox:
            x, y, w, h = bbox
            
            # small padding (10%) so we don't cut off beaks/tails
            h_img, w_img, _ = img_cv.shape
            pad_w = int(w * 0.10)
            pad_h = int(h * 0.10)
            
            x = max(0, x - pad_w)
            y = max(0, y - pad_h)
            w = min(w_img - x, w + 2*pad_w)
            h = min(h_img - y, h + 2*pad_h)

            crop = img_cv[y:y+h, x:x+w]
            cv2.imwrite(str(save_path), crop)
            stats["cropped"] += 1
        else:
            # fallback: Save original
            cv2.imwrite(str(save_path), img_cv)
            stats["original"] += 1
            
    print(f"Finished {src_dir.name}: {stats['cropped']} cropped, {stats['original']} saved original.")

def main():
    cfg = Config()
    
    # force device check
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found. This will be slow!")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    detector = load_detector(device)

    # train
    process_folder(detector, cfg.data_dir / "train_images", cfg.data_dir / "train_images_cropped", device)

    # validation
    process_folder(detector, cfg.data_dir / "val_images", cfg.data_dir / "val_images_cropped", device)

    # test
    process_folder(detector, cfg.data_dir / "test_images", cfg.data_dir / "test_images_cropped", device)

if __name__ == "__main__":
    main()
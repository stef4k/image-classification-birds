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
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model

def load_cub_metadata(meta_dir: Path):
    """
    Parses CUB images.txt and bounding_boxes.txt
    Returns: dict mapping filename (str) -> [x, y, w, h] (float list)
    """
    images_txt = meta_dir / "images.txt"
    bbox_txt = meta_dir / "bounding_boxes.txt"
    
    if not images_txt.exists():
        print(f"Warning: Metadata not found at {meta_dir}. Will rely solely on AI Detector.")
        return {}

    print("Loading CUB Metadata...")
    # load ID -> Filename
    id_to_filename = {}
    with open(images_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_id = int(parts[0])
                # We store just the filename (e.g., "Bird_001.jpg") to match easier
                full_path = parts[1]
                filename = Path(full_path).name 
                id_to_filename[img_id] = filename

    # load ID -> BBox
    filename_to_bbox = {}
    with open(bbox_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                img_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]] # x, y, w, h
                
                if img_id in id_to_filename:
                    fname = id_to_filename[img_id]
                    filename_to_bbox[fname] = bbox
    
    print(f"Loaded {len(filename_to_bbox)} bounding boxes.")
    return filename_to_bbox

def get_bird_box_ai(model, img_path, device, confidence_threshold=0.6):
    """
    AI Fallback: Returns (x, y, w, h) using Faster R-CNN
    """
    try:
        img_pil = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img_pil.size
        img_tensor = F.to_tensor(img_pil).to(device)
        
        with torch.no_grad():
            predictions = model([img_tensor])[0]
        
        labels = predictions['labels']
        scores = predictions['scores']
        boxes = predictions['boxes']
        
        # strict check
        mask = (labels == BIRD_CLASS_ID) & (scores > confidence_threshold)
        bird_boxes = boxes[mask]
        bird_scores = scores[mask]
        
        # loose check fallback
        if len(bird_boxes) == 0:
            mask_loose = (labels == BIRD_CLASS_ID) & (scores > 0.3)
            bird_boxes = boxes[mask_loose]
            bird_scores = scores[mask_loose]
            
            if len(bird_boxes) == 0:
                return None

        best_idx = torch.argmax(bird_scores)
        box = bird_boxes[best_idx].cpu().numpy() 
        x1, y1, x2, y2 = box
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def process_folder(model, src_dir: Path, dst_dir: Path, device, gt_bboxes):
    if not src_dir.exists():
        print(f"Skipping {src_dir} (does not exist)")
        return

    ensure_dir(dst_dir)
    images = sorted([p for p in src_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    
    print(f"\nProcessing: {src_dir} -> {dst_dir} ({len(images)} images)")
    
    stats = {"gt": 0, "ai": 0, "original": 0}
    
    for img_path in tqdm(images):
        rel_path = img_path.relative_to(src_dir)
        save_path = dst_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        img_cv = cv2.imread(str(img_path))
        if img_cv is None: continue
        h_img, w_img, _ = img_cv.shape

        # try Ground Truth First
        fname = img_path.name
        bbox = None
        source = "none"

        if fname in gt_bboxes:
            # found in metadata! Use perfect box.
            bbox = gt_bboxes[fname] # x, y, w, h
            stats["gt"] += 1
            source = "gt"
        else:
            # fallback to AI
            bbox = get_bird_box_ai(model, img_path, device)
            if bbox:
                stats["ai"] += 1
                source = "ai"

        if bbox:
            x, y, w, h = map(int, bbox)
            
            # add padding (only for AI, or small padding for GT if desired)
            # GT is tight, AI might need context. 
            # 5% padding to everything to be safe.
            pad_w = int(w * 0.05)
            pad_h = int(h * 0.05)
            
            x = max(0, x - pad_w)
            y = max(0, y - pad_h)
            w = min(w_img - x, w + 2*pad_w)
            h = min(h_img - y, h + 2*pad_h)

            # safety check for empty crops
            if w > 1 and h > 1:
                crop = img_cv[y:y+h, x:x+w]
                cv2.imwrite(str(save_path), crop)
            else:
                # box invalid, save original
                cv2.imwrite(str(save_path), img_cv)
                stats["original"] += 1
        else:
            # no GT and No AI detection found
            cv2.imwrite(str(save_path), img_cv)
            stats["original"] += 1
            
    print(f"  Finished: Used GT for {stats['gt']}, AI for {stats['ai']}, Saved Original {stats['original']}")

def main():
    cfg = Config()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", type=str, default=None, help="Source folder to crop (overrides defaults).")
    ap.add_argument("--dst_dir", type=str, default=None, help="Destination folder for crops (overrides defaults).")
    args = ap.parse_args()
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found. This will be slow!")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    # Ground Truth Metadata
    META_DIR = cfg.data_dir / "img_meta_additional" 
    gt_bboxes = load_cub_metadata(META_DIR)

    # detector (as backup)
    detector = load_detector(device)

    if args.src_dir and args.dst_dir:
        src_dir = Path(args.src_dir)
        dst_dir = Path(args.dst_dir)
        process_folder(detector, src_dir, dst_dir, device, gt_bboxes)
        return

    # train
    process_folder(detector, cfg.data_dir / "train_images", cfg.data_dir / "train_images_cropped", device, gt_bboxes)
    
    # validation
    process_folder(detector, cfg.data_dir / "val_images", cfg.data_dir / "val_images_cropped", device, gt_bboxes)

    # test
    process_folder(detector, cfg.data_dir / "test_images", cfg.data_dir / "test_images_cropped", device, gt_bboxes)

if __name__ == "__main__":
    main()

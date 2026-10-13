# Image-classification-birds

Course project for the Machine Learning course at CentraleSupélec, carried out by Kristóf Balázs and Stefanos Kypritidis as part of a class Kaggle competition.
 
## Overview
 
The task is to classify 20 bird species from a subset of the Caltech-UCSD Birds-200-2011 (CUB) dataset, scored by top-1 accuracy on a hidden 400-image test set. It is a fine-grained problem: several classes differ only in small details such as beak shape or plumage texture, and some pairs (American Crow vs. Fish Crow, Brewer's vs. Rusty Blackbird) are hard to tell apart even for experienced birders. We had only 1,082 labelled training images (51–58 per class) and a 103-image validation split, and no external labels were allowed. Because the validation split is small and unevenly distributed across classes, we selected models mainly with stratified 5-fold cross-validation.
 
We benchmarked a wide range of methods, from hand-crafted features (HOG with logistic regression, dense-SIFT Fisher Vectors with a linear SVM) to pretrained `timm` backbones used as frozen feature extractors: ResNet-50, EfficientNet-B0, ConvNeXt / ConvNeXt V2, CAFormer, ViT-SO150M and EVA-02 Large. On top of the frozen features we compared linear, MLP and ArcFace heads. To stop the models from relying on the background, we cropped every image around the bird (ground-truth boxes for train/val, a Faster R-CNN detector for the test images, with a few detector failures re-cropped by hand), padded images instead of stretching them to keep the aspect ratio, and moved to higher input resolutions (384–448 px). Hyperparameter sweeps were tracked in Weights & Biases, and we used FiftyOne to inspect validation errors class by class.
 
Pretrained representations clearly dominated: the best classical pipeline reached 38.4% cross-validation accuracy, while a frozen EfficientNet-B0 with logistic regression already reached 89.4%. The most consistent gains came from the data side (cropping, aspect-ratio-preserving padding, higher resolution) rather than from more complex heads. Error analysis showed that the remaining mistakes concentrated on a few near-identical species pairs, mostly the two crows. We tried several targeted fixes: pseudo-labelling 439 extra images from NABirds, a hierarchical mixture of experts, a soft-voting ensemble, partial unfreezing of the backbone, and a dedicated binary crow separator. None of them gave a reliable improvement on held-out data. The crow separator, for example, reached 95% on our crow validation images but only 75% on an external set of 180 crow images, and left the overall test accuracy unchanged.
 
The main lesson came from the final leaderboard. Our submitted model, EVA-02 Large with an ArcFace head at 448 px, scored 93.0% on the public leaderboard but dropped to 85.5% on the private one, a sign that our model choice had been tuned to the public split. A much simpler frozen ViT-384 with a single linear layer scored 90.0% on both, and trains in about 10 minutes on a 6 GB laptop GPU. With this little data, added model complexity and aggressive margin losses reduced robustness, and gains measured on small validation splits need to be read with caution.
 
## Report and presentation
 
- [Final report (PDF)](docs/final_report.pdf)
- [Final presentation (PDF)](docs/final_presentation.pdf)

## Results at a glance
 
| Approach | Accuracy |
|---|---|
| HOG + logistic regression | 22% (5-fold CV) |
| Fisher Vectors + linear SVM | 38.4% (5-fold CV) |
| EfficientNet-B0 embeddings + logistic regression | 77% (private test) |
| EVA-02 Large + ArcFace head, 448 px (submitted) | 93.0% public / 85.5% private |
| ViT-384 + linear head (best) | 90.0% public / 90.0% private |
 
**Stack:** PyTorch, timm, scikit-learn, torchvision (Faster R-CNN), Weights & Biases, FiftyOne


## Setup

### 1) Create a virtual environment
```bash
python -m venv .birds_ml_venv
```
Recommended is python 3.11:
```bash
py -3.11 -m venv .birds_ml_venv
```                                                   

### 2) Activate virtual environment
```bash
.birds_ml_venv\Scripts\Activate.ps1
```

### 3) Install dependecies
```bash
pip install -r requirements.txt
```

### 4) Install the project package
```bash
pip install -e .
```

### 5) Download competition data
Download the data from [Kaggle](https://www.kaggle.com/competitions/bdma-07-competition-2026/data) and unzip it into the `data/` folder at the root of this repository

## Baseline pipeline

###  Train (with CV) + save final model
```bash
python scripts/train_linear.py --kind logreg --C 1.0 --backbone efficientnet_b0
```
Outputs:
- `outputs/logreg_efficientnet_b0.joblib` (trained on all train data)

- `outputs/meta_logreg_efficientnet_b0.json` (run metadata + mapping + CV metrics)

- `outputs/meta.json` (latest run pointer)

- `outputs/cache/emb_efficientnet_b0_train.npz` (cached train embeddings)

### Evaluate on validation set
```bash
python scripts/eval_linear.py --kind logreg --backbone efficientnet_b0
```
To force recompute embeddings (ignore cache)
```bash
python scripts/eval_linear.py --kind logreg --backbone efficientnet_b0 --no_cache
```

### Predict on test set and generate Kaggle submission
```bash
python scripts/predict_test.py --kind logreg --backbone efficientnet_b0 --out submission_logreg_effb0.csv
```
The submission file is under: `outputs/submission_logreg_effb0.csv`


### Inspect validation mistakes visually
Launches the FiftyOne app with
```bash
python scripts/inspect_val_fiftyone.py --kind logreg --backbone efficientnet_b0
```
Note: this script uses a project-local FiftyOne database under `outputs/fiftyone_db` to avoid Windows file-lock issues with the default global DB.
By default it will use cropped validation images if `data/val_images_cropped` exists (override with `--no-use-crops`).
Force recompute embeddings:
```bash
python scripts/inspect_val_fiftyone.py --kind logreg --backbone efficientnet_b0 --no_cache
```

## Reproducing Best Results

To reproduce the top-performing models described in the report, use the `scripts/train_neural.py` script.

### 1. The Best Model: ViT-384 (Linear Head)
This model achieved **90.0%** on the Private Test Set. It uses a standard linear head and is trained for 30 epochs.

**Train:**
```bash
python scripts/train_neural.py \
    --backbone vit_so150m2_384 \
    --img_size 384 \
    --epochs 30 \
    --use_crops
```

Outputs: `outputs/linear_vit_so150m2_384_cropped_384.pth`

**Predict:**

```bash
python scripts/predict_test_neural.py \
    --backbone vit_so150m2_384 \
    --img_size 384 \
    --use_crops \
    --out submission_linear_vit_so150m2_384.csv
```

### 2. The Selected Model: EVA-02 Large (ArcFace)
This model achieved **93.0%** on the Public Leaderboard. It uses the ArcFace margin loss for tighter class clustering and high-resolution input (448px).

**Train:**
```bash
python scripts/train_neural.py \
    --backbone eva02_large_448 \
    --img_size 448 \
    --epochs 50 \
    --use_arcface \
    --use_crops
```

Outputs: `outputs/arcface_eva02_large_448_cropped_448.pth`

**Predict:**

```bash
python scripts/predict_test_neural.py \
    --backbone eva02_large_448 \
    --img_size 448 \
    --use_arcface \
    --use_crops \
    --out submission_arcface_eva02_large_448.csv
```

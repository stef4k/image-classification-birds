# Image-classification-birds
Bird species classifier on Kaggle with clean configs, runs tracking, and report-ready experiments

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
Force recompute embeddings:
```bash
python scripts/inspect_val_fiftyone.py --kind logreg --backbone efficientnet_b0 --no_cache
```


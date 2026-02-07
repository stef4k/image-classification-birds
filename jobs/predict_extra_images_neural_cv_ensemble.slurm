#!/bin/bash
#SBATCH --job-name=birds-predict-extra-cv-ens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [ -f ".birds_ml_venv/bin/activate" ]; then
  source ".birds_ml_venv/bin/activate"
fi

python scripts/predict_test_neural.py \
  --backbone vit_so150m2_384 \
  --img_size 384 \
  --use_crops \
  --test_dir data/extra_images \
  --ckpt_glob "cv_fold*_timm_finetune_vit_so150m2_384_cropped_384.pth" \
  --out extra_images_cv_ensemble_predictions.csv \
  --out_with_conf extra_images_cv_ensemble_predictions_with_conf.csv

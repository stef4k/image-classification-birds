#!/bin/bash
#SBATCH --job-name=birds-train-crow-separator
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

srun python scripts/train_crow_separator.py \
  --backbone vit_so150m2_384 \
  --img_size 384 \
  --use_crops \
  --epochs 25 \
  --val_dir data/f_images_cropped \
  --selection_metric balanced_acc \
  --pseudo_min_conf 0.8 \
  --pseudo_weight 0.75

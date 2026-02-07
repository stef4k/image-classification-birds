#!/bin/bash
#SBATCH --job-name=birds-cv-unfrozen-weighted
#SBATCH --partition=gpua100
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

python scripts/cv_neural.py \
  --backbone vit_so150m2_384 \
  --img_size 384 \
  --use_crops \
  --folds 5 \
  --save_folds \
  --epochs 20 \
  --finetune_epochs 8 \
  --finetune_blocks 2 \
  --crow_a American_Crow \
  --crow_b Fish_Crow \
  --crow_weight 2.0

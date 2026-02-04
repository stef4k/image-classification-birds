#!/bin/bash
#SBATCH --job-name=eval_val_neural-images-birds
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_val_neural_%x_%j.out
#SBATCH --error=logs/eval_val_neural_%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [ -f ".birds_ml_venv/bin/activate" ]; then
  source ".birds_ml_venv/bin/activate"
fi


python scripts/eval_neural.py --backbone vit_so150m2_384 --img_size 384 --use_crops --val_dir data/f_images_cropped

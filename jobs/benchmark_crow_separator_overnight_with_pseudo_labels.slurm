#!/bin/bash
#SBATCH --job-name=birds-crow-sep-overnight-pseudo
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

srun python scripts/benchmark_crow_separator_overnight.py \
  --backbone eva02_large_448 \
  --img_size 448 \
  --use_crops \
  --epochs 16 \
  --batch_size 16 \
  --val_dir data/f_images_cropped \
  --secondary_val_dir data/val_images_cropped \
  --pseudo_profile_name pseudo_w0p5_c0p8 \
  --pseudo_profile_weight 0.5 \
  --pseudo_profile_min_conf 0.8 \
  --results_csv outputs/crow_separator_overnight_benchmark_eva02_pseudo_w0p5_c0p8.csv \
  --continue_on_error

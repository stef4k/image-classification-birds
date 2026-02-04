#!/bin/bash
#SBATCH --job-name=crop-images-birds
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/crop_images_%x_%j.out
#SBATCH --error=logs/crop_images_%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [ -f ".birds_ml_venv/bin/activate" ]; then
  source ".birds_ml_venv/bin/activate"
fi



python scripts/crop_with_pretrained.py --src_dir data/f_images --dst_dir data/f_images_cropped

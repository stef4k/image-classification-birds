#!/bin/bash
#SBATCH --job-name=inspect-birds-train-neural
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/inspect_%x_%j.out
#SBATCH --error=logs/inspect_%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [ -f ".birds_ml_venv/bin/activate" ]; then
  source ".birds_ml_venv/bin/activate"
fi

srun python scripts/inspect_val_neural_csv.py \
  --backbone vit_so150m2_384 \
  --img_size 384 \
  --use_crops \
  --only_mistakes
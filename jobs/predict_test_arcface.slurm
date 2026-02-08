#!/bin/bash
#SBATCH --job-name=birds-predict-neural-arcface
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

srun python scripts/predict_test_neural.py --backbone eva02_large_448 --img_size 448 --use_crops --use_arcface --out submission_arcface_eva02_large_448.csv
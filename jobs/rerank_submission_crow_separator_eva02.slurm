#!/bin/bash
#SBATCH --job-name=birds-rerank-crow-eva02-ft-ac80
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [ -f ".birds_ml_venv/bin/activate" ]; then
  source ".birds_ml_venv/bin/activate"
fi

srun python scripts/rerank_crow_submission.py \
  --submission_in outputs/submission_eva02_large_448_arcface.csv \
  --submission_out outputs/submission_eva02_large_448_arcface_crowsep_eva02_finetuned_ac80.csv \
  --separator_backbone eva02_large_448 \
  --separator_img_size 448 \
  --separator_model_name crow_sep_bench_eva02_large_448_448 \
  --use_crops

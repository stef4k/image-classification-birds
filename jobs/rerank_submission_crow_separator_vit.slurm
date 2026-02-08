#!/bin/bash
#SBATCH --job-name=birds-rerank-crow-vit
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
  --submission_out outputs/submission_eva02_large_448_arcface_crowsep_vit.csv \
  --separator_backbone vit_so150m2_384 \
  --separator_img_size 384 \
  --separator_model_name crow_sep_bench_vit_so150m2_384_384 \
  --use_crops

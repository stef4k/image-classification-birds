#!/bin/bash
#SBATCH --job-name=birds-rerank-crow-ens-top2-overnight-pseudo-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [ -f ".birds_ml_venv/bin/activate" ]; then
  source ".birds_ml_venv/bin/activate"
fi

srun python scripts/rerank_crow_submission_ensemble.py \
  --submission_in outputs/submission_eva02_large_448_arcface.csv \
  --submission_out outputs/submission_eva02_large_448_arcface_crowsep_ensemble_top2_overnight_pseudo_all.csv \
  --separator_backbone eva02_large_448 \
  --separator_img_size 448 \
  --separator_model_names \
    overnight_neural_linear_finetune_focal_strong_tta_eva02_large_448_448_pseudo_all_balanced_acc \
    overnight_neural_arcface_finetune_light_tta_eva02_large_448_448_pseudo_all_balanced_acc \
  --use_crops

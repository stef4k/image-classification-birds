#!/bin/bash
#SBATCH --job-name=birds-benchmark-crow-separator
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

srun python scripts/benchmark_crow_separator_backbones.py \
  --use_crops \
  --epochs 25 \
  --val_dir data/f_images_cropped \
  --selection_metric acc \
  --spec vit_so150m2_384:384 \
  --spec convnextv2_base_384:384 \
  --spec convnextv2_large_384:384 \
  --spec caformer_b36_384:384 \
  --spec eva02_large_448:448 \
  --results_csv outputs/crow_separator_backbone_benchmark_acc_finetune.csv \
  --finetune_backbone

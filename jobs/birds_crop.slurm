#!/bin/bash
#SBATCH --job-name=birds_crop
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs
cd "$WORKDIR/stef_checks/eval/image"

$HOME/.conda/envs/ruche311/bin/python -u scripts/crop_with_pretrained.py
echo "Done."
#!/bin/bash
#SBATCH --job-name=wandb_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO="$WORKDIR/stef_checks/eval/image"
PYTHON="$HOME/.conda/envs/ruche311/bin/python"

export WANDB_PROJECT="birds-neural-sweep"
export WANDB_GROUP="wandb_test_${SLURM_JOB_ID}"
export WANDB_NAME="test_2epoch_jid${SLURM_JOB_ID}"
export WANDB_MODE="${WANDB_MODE:-online}"   # change to offline if needed

cd "$REPO"
mkdir -p logs outputs

"$PYTHON" -c "import torch; print('cuda?', torch.cuda.is_available(), 'gpus_visible', torch.cuda.device_count())"
"$PYTHON" -c "import wandb; print('wandb', wandb.__version__)"

"$PYTHON" -u scripts/train_neural_expanded.py --wandb --epochs 2
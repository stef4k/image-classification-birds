#!/bin/bash
#SBATCH --job-name=birds_neural_overnight
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-179%8

set -euo pipefail

REPO="$WORKDIR/stef_checks/eval/image"
PYTHON="$HOME/.conda/envs/ruche311/bin/python"
SCRIPT="scripts/train_neural_expanded.py"

cd "$REPO"
mkdir -p logs outputs

# W&B config (keep stable)
export WANDB_PROJECT="birds-neural-sweep"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_GROUP="grid_sweep_overnight_${SLURM_JOB_ID}"

# quick sanity prints per task (helps debugging if some tasks fail)
"$PYTHON" -c "import torch; print('cuda?', torch.cuda.is_available(), 'gpus_visible', torch.cuda.device_count())"
"$PYTHON" -c "import wandb; print('wandb', wandb.__version__)"

# --------------------
# Sweep space (pruned and intentional)
# --------------------
BACKBONE="efficientnet_b0"
EPOCHS=30
BS=32
SEED=42
OPT="adamw"
SCH="cosine"

LRS=(3e-4 1e-3 3e-3)
WDS=(1e-5 1e-4 3e-4)
CROPS=(0 1)

# Only used when AUG=1
HDS=(256 512 768)
DROPS=(0.3 0.5 0.7)

# --------------------
# Map SLURM_ARRAY_TASK_ID -> config
# --------------------
i=$SLURM_ARRAY_TASK_ID

# First block: AUG=0, configs = 3*3*2 = 18
N0=$(( ${#LRS[@]} * ${#WDS[@]} * ${#CROPS[@]} ))  # 18

if [ "$i" -lt "$N0" ]; then
  AUG=0

  n_lr=${#LRS[@]}
  n_wd=${#WDS[@]}
  n_crop=${#CROPS[@]}

  crop_idx=$(( i / (n_wd * n_lr) ))
  rem=$(( i % (n_wd * n_lr) ))
  wd_idx=$(( rem / n_lr ))
  lr_idx=$(( rem % n_lr ))

  LR=${LRS[$lr_idx]}
  WD=${WDS[$wd_idx]}
  USE_CROPS=${CROPS[$crop_idx]}

  # defaults when no augmentation
  HD=512
  DO=0.5
else
  # Second block: AUG=1, configs = 3*3*2*3*3 = 162
  AUG=1
  j=$(( i - N0 ))

  n_lr=${#LRS[@]}
  n_wd=${#WDS[@]}
  n_crop=${#CROPS[@]}
  n_hd=${#HDS[@]}
  n_do=${#DROPS[@]}

  # ordering: crop -> wd -> lr -> hd -> do
  crop_idx=$(( j / (n_wd * n_lr * n_hd * n_do) ))
  rem=$(( j % (n_wd * n_lr * n_hd * n_do) ))
  wd_idx=$(( rem / (n_lr * n_hd * n_do) ))
  rem2=$(( rem % (n_lr * n_hd * n_do) ))
  lr_idx=$(( rem2 / (n_hd * n_do) ))
  rem3=$(( rem2 % (n_hd * n_do) ))
  hd_idx=$(( rem3 / n_do ))
  do_idx=$(( rem3 % n_do ))

  LR=${LRS[$lr_idx]}
  WD=${WDS[$wd_idx]}
  USE_CROPS=${CROPS[$crop_idx]}
  HD=${HDS[$hd_idx]}
  DO=${DROPS[$do_idx]}
fi

AUG_FLAG=""
if [ "$AUG" -eq 1 ]; then AUG_FLAG="--augment"; fi

CROP_FLAG=""
if [ "$USE_CROPS" -eq 1 ]; then CROP_FLAG="--use_crops"; fi

# W&B run name per task
RUN_NAME="effb0_c${USE_CROPS}_a${AUG}_lr${LR}_wd${WD}_hd${HD}_do${DO}_s${SEED}_jid${SLURM_JOB_ID}_t${SLURM_ARRAY_TASK_ID}"
export WANDB_NAME="$RUN_NAME"

echo "------------------------------------------------------------"
echo "Task $i / 179"
echo "RUN_NAME=$RUN_NAME"
echo "Config: backbone=$BACKBONE epochs=$EPOCHS bs=$BS seed=$SEED aug=$AUG crops=$USE_CROPS lr=$LR wd=$WD hd=$HD do=$DO"
echo "WANDB_MODE=$WANDB_MODE WANDB_PROJECT=$WANDB_PROJECT WANDB_GROUP=$WANDB_GROUP"
echo "------------------------------------------------------------"

# IMPORTANT: no 'source', no relying on PATH
"$PYTHON" -u "$SCRIPT" \
  --wandb \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" \
  --batch_size "$BS" \
  --seed "$SEED" \
  --optimizer "$OPT" \
  --scheduler "$SCH" \
  --lr "$LR" \
  --weight_decay "$WD" \
  --hidden_dim "$HD" \
  --dropout "$DO" \
  $CROP_FLAG \
  $AUG_FLAG

#!/bin/bash
#SBATCH --job-name=birds_svm_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-999%4

set -euo pipefail

REPO="$WORKDIR/stef_checks/eval/image"
PYTHON="$HOME/.conda/envs/ruche311/bin/python"
SCRIPT="scripts/train_linear.py"

cd "$REPO"
mkdir -p logs outputs

# W&B config (keep stable)
export WANDB_PROJECT="birds-svm-sweep"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_GROUP="grid_sweep_svm_${SLURM_JOB_ID}"

# --------------------
# Sweep space
# --------------------
BACKBONES=("resnet50" "efficientnet_b0")
SEED=42

CS=(1e-4 3e-4 1e-3 3e-3 1e-2 1e-1 1 3 10)
CROPS=(0 1)
AUGS=(0 1)
DUALS=(0 1)  # only used for squared_hinge
TOLS=(1e-5 1e-4 1e-3 1e-2 1e-1)

# --------------------
# Map SLURM_ARRAY_TASK_ID -> config
# --------------------
OFFSET=${OFFSET:-0}
i=$(( SLURM_ARRAY_TASK_ID + OFFSET ))
TOTAL_TASKS=1080

if [ "$i" -ge "$TOTAL_TASKS" ]; then
  echo "Task index $i is out of range (TOTAL_TASKS=$TOTAL_TASKS). Exiting."
  exit 0
fi

# First block: loss=squared_hinge, dual in {0,1}
N0=$(( ${#CS[@]} * ${#CROPS[@]} * ${#AUGS[@]} * ${#DUALS[@]} * ${#TOLS[@]} * ${#BACKBONES[@]} ))

if [ "$i" -lt "$N0" ]; then
  n_c=${#CS[@]}
  n_crop=${#CROPS[@]}
  n_aug=${#AUGS[@]}
  n_dual=${#DUALS[@]}
  n_bb=${#BACKBONES[@]}
  n_tol=${#TOLS[@]}

  bb_idx=$(( i / (n_tol * n_dual * n_aug * n_crop * n_c) ))
  rem=$(( i % (n_tol * n_dual * n_aug * n_crop * n_c) ))
  tol_idx=$(( rem / (n_dual * n_aug * n_crop * n_c) ))
  rem2=$(( rem % (n_dual * n_aug * n_crop * n_c) ))
  dual_idx=$(( rem2 / (n_aug * n_crop * n_c) ))
  rem3=$(( rem2 % (n_aug * n_crop * n_c) ))
  aug_idx=$(( rem3 / (n_crop * n_c) ))
  rem4=$(( rem3 % (n_crop * n_c) ))
  crop_idx=$(( rem4 / n_c ))
  c_idx=$(( rem4 % n_c ))

  CVAL=${CS[$c_idx]}
  USE_CROPS=${CROPS[$crop_idx]}
  AUG=${AUGS[$aug_idx]}
  LOSS="squared_hinge"
  DUAL=${DUALS[$dual_idx]}
  BACKBONE=${BACKBONES[$bb_idx]}
  TOL=${TOLS[$tol_idx]}
else
  # Second block: loss=hinge, dual forced to 1
  j=$(( i - N0 ))

  n_c=${#CS[@]}
  n_crop=${#CROPS[@]}
  n_aug=${#AUGS[@]}
  n_bb=${#BACKBONES[@]}
  n_tol=${#TOLS[@]}

  bb_idx=$(( j / (n_tol * n_aug * n_crop * n_c) ))
  rem=$(( j % (n_tol * n_aug * n_crop * n_c) ))
  tol_idx=$(( rem / (n_aug * n_crop * n_c) ))
  rem2=$(( rem % (n_aug * n_crop * n_c) ))
  aug_idx=$(( rem2 / (n_crop * n_c) ))
  rem3=$(( rem2 % (n_crop * n_c) ))
  crop_idx=$(( rem3 / n_c ))
  c_idx=$(( rem3 % n_c ))

  CVAL=${CS[$c_idx]}
  USE_CROPS=${CROPS[$crop_idx]}
  AUG=${AUGS[$aug_idx]}
  LOSS="hinge"
  DUAL=1
  BACKBONE=${BACKBONES[$bb_idx]}
  TOL=${TOLS[$tol_idx]}
fi

AUG_FLAG=""
if [ "$AUG" -eq 1 ]; then AUG_FLAG="--augment"; fi

CROP_FLAG=""
if [ "$USE_CROPS" -eq 1 ]; then CROP_FLAG="--use_crops"; fi

RUN_NAME="svm_${BACKBONE}_c${USE_CROPS}_a${AUG}_C${CVAL}_loss${LOSS}_dual${DUAL}_tol${TOL}_t${SLURM_ARRAY_TASK_ID}"

echo "------------------------------------------------------------"
echo "Task $i / 1079"
echo "RUN_NAME=$RUN_NAME"
echo "Config: kind=svm backbone=$BACKBONE seed=$SEED aug=$AUG crops=$USE_CROPS C=$CVAL loss=$LOSS dual=$DUAL tol=$TOL"
echo "------------------------------------------------------------"

export WANDB_NAME="$RUN_NAME"

"$PYTHON" -u "$SCRIPT" \
  --wandb \
  --kind svm \
  --backbone "$BACKBONE" \
  --C "$CVAL" \
  --svm_loss "$LOSS" \
  --svm_dual "$DUAL" \
  --svm_tol "$TOL" \
  $CROP_FLAG \
  $AUG_FLAG

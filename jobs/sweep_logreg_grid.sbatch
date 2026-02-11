#!/bin/bash
#SBATCH --job-name=birds_logreg_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-383%4

set -euo pipefail

REPO="$WORKDIR/stef_checks/eval/image"
PYTHON="$HOME/.conda/envs/ruche311/bin/python"
SCRIPT="scripts/train_linear.py"

cd "$REPO"
mkdir -p logs outputs

# W&B config (keep stable)
export WANDB_PROJECT="birds-logreg-sweep"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_GROUP="grid_sweep_logreg_${SLURM_JOB_ID}"

# --------------------
# Sweep space
# --------------------
BACKBONES=("resnet50" "efficientnet_b0")
SEED=42

CS=(1e-3 1e-2 1e-1 1 3 10)
CROPS=(0 1)
AUGS=(0 1)
PENALTIES_BASE=("l2" "l1")
SOLVERS_L2=("lbfgs" "sag" "newton-cg" "saga")
L1RS=(0.2 0.5 0.8)

# --------------------
# Map SLURM_ARRAY_TASK_ID -> config
# --------------------
i=$SLURM_ARRAY_TASK_ID

# First block: penalties without l1_ratio
N0=$(( ${#CS[@]} * ${#CROPS[@]} * ${#AUGS[@]} * ${#BACKBONES[@]} * (${#SOLVERS_L2[@]} + 1) ))

if [ "$i" -lt "$N0" ]; then
  n_c=${#CS[@]}
  n_crop=${#CROPS[@]}
  n_aug=${#AUGS[@]}
  n_bb=${#BACKBONES[@]}
  n_l2solv=${#SOLVERS_L2[@]}
  n_solver_total=$(( n_l2solv + 1 ))  # +1 for l1+saga

  bb_idx=$(( i / (n_solver_total * n_aug * n_crop * n_c) ))
  rem=$(( i % (n_solver_total * n_aug * n_crop * n_c) ))
  solv_idx=$(( rem / (n_aug * n_crop * n_c) ))
  rem2=$(( rem % (n_aug * n_crop * n_c) ))
  aug_idx=$(( rem2 / (n_crop * n_c) ))
  rem3=$(( rem2 % (n_crop * n_c) ))
  crop_idx=$(( rem3 / n_c ))
  c_idx=$(( rem3 % n_c ))

  CVAL=${CS[$c_idx]}
  USE_CROPS=${CROPS[$crop_idx]}
  AUG=${AUGS[$aug_idx]}
  if [ "$solv_idx" -lt "$n_l2solv" ]; then
    PEN="l2"
    SOLVER=${SOLVERS_L2[$solv_idx]}
    L1R=0.5
  else
    PEN="l1"
    SOLVER="saga"
    L1R=0.5
  fi
  BACKBONE=${BACKBONES[$bb_idx]}
else
  # Second block: elasticnet with l1_ratio grid
  j=$(( i - N0 ))

  n_c=${#CS[@]}
  n_crop=${#CROPS[@]}
  n_aug=${#AUGS[@]}
  n_l1r=${#L1RS[@]}
  n_bb=${#BACKBONES[@]}

  bb_idx=$(( j / (n_l1r * n_aug * n_crop * n_c) ))
  rem=$(( j % (n_l1r * n_aug * n_crop * n_c) ))
  l1r_idx=$(( rem / (n_aug * n_crop * n_c) ))
  rem2=$(( rem % (n_aug * n_crop * n_c) ))
  aug_idx=$(( rem2 / (n_crop * n_c) ))
  rem3=$(( rem2 % (n_crop * n_c) ))
  crop_idx=$(( rem3 / n_c ))
  c_idx=$(( rem3 % n_c ))

  CVAL=${CS[$c_idx]}
  USE_CROPS=${CROPS[$crop_idx]}
  AUG=${AUGS[$aug_idx]}
  PEN="elasticnet"
  L1R=${L1RS[$l1r_idx]}
  SOLVER="saga"
  BACKBONE=${BACKBONES[$bb_idx]}
fi

AUG_FLAG=""
if [ "$AUG" -eq 1 ]; then AUG_FLAG="--augment"; fi

CROP_FLAG=""
if [ "$USE_CROPS" -eq 1 ]; then CROP_FLAG="--use_crops"; fi

RUN_NAME="logreg_${BACKBONE}_c${USE_CROPS}_a${AUG}_C${CVAL}_pen${PEN}_solv${SOLVER}_l1r${L1R}_t${SLURM_ARRAY_TASK_ID}"

echo "------------------------------------------------------------"
echo "Task $i / 383"
echo "RUN_NAME=$RUN_NAME"
echo "Config: kind=logreg backbone=$BACKBONE seed=$SEED aug=$AUG crops=$USE_CROPS C=$CVAL penalty=$PEN solver=$SOLVER l1_ratio=$L1R"
echo "------------------------------------------------------------"

export WANDB_NAME="$RUN_NAME"

"$PYTHON" -u "$SCRIPT" \
  --wandb \
  --kind logreg \
  --backbone "$BACKBONE" \
  --C "$CVAL" \
  --logreg_penalty "$PEN" \
  --logreg_solver "$SOLVER" \
  --logreg_l1_ratio "$L1R" \
  $CROP_FLAG \
  $AUG_FLAG

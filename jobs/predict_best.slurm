#!/bin/bash
#SBATCH --job-name=birds_predict_best
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO="$WORKDIR/stef_checks/eval/image"
PYTHON="$HOME/.conda/envs/ruche311/bin/python"

MODEL_NAME="neural_efficientnet_b0_cropped_aug_bs32_lr0.0003_wd0.0003_adamw_hd256_do0.7_ls0_schcosine_seed42"
OUT_CSV="submission_best.csv"

cd "$REPO"
mkdir -p logs outputs

"$PYTHON" -c "import torch; print('cuda?', torch.cuda.is_available(), 'gpus_visible', torch.cuda.device_count())"

# Sanity: make sure the checkpoint + meta exist
test -f "outputs/${MODEL_NAME}.pth"
test -f "outputs/meta_${MODEL_NAME}.json"

"$PYTHON" -u scripts/predict_test_neural_from_model_name.py \
  --model_name "$MODEL_NAME" \
  --out "$OUT_CSV" \
  --batch_size 32

echo "Done. Output: outputs/${OUT_CSV}"

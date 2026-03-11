#!/bin/bash
#SBATCH --job-name=thermokourt-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ── ThermoKourt auto-scorer training on Aoraki ──────────────────────
# Usage:
#   sbatch train_scorer.sh --annotations /data/manual/ --epochs 100
#
# This script:
#   1. Loads the thermokourt conda environment
#   2. Stages data to local scratch for fast I/O
#   3. Runs training with checkpointing
#   4. Copies best model back to shared filesystem

set -euo pipefail

# Parse arguments passed via sbatch
ANNOTATIONS="${1:?Usage: sbatch train_scorer.sh <annotations_dir> <epochs>}"
EPOCHS="${2:-100}"

echo "=== ThermoKourt auto-scorer training ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "GPU:          ${CUDA_VISIBLE_DEVICES:-none}"
echo "Annotations:  ${ANNOTATIONS}"
echo "Epochs:       ${EPOCHS}"
echo "========================================="

# Environment
module load CUDA/12.1
source activate thermokourt

# Stage data to local scratch
SCRATCH="/scratch/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH}/data"
cp -r "${ANNOTATIONS}" "${SCRATCH}/data/annotations"

# Train
python -m thermokourt.score.train \
    --annotations "${SCRATCH}/data/annotations" \
    --output "${SCRATCH}/checkpoints" \
    --epochs "${EPOCHS}" \
    --batch_size 32 \
    --lr 1e-4 \
    --device cuda

# Copy best model back
OUTDIR="$(dirname "${ANNOTATIONS}")/models"
mkdir -p "${OUTDIR}"
cp "${SCRATCH}/checkpoints/best.pt" "${OUTDIR}/best_${SLURM_JOB_ID}.pt"
echo "Best model saved to ${OUTDIR}/best_${SLURM_JOB_ID}.pt"

# Cleanup
rm -rf "${SCRATCH}"

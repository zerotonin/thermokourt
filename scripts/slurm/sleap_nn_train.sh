#!/bin/bash
#SBATCH --job-name=sleap-nn-train
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu_L40,aoraki_gpu_A100_80GB,aoraki_gpu_A100_40GB,aoraki_gpu_L4_24GB,aoraki_gpu_RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.log
#
# ┌──────────────────────────────────────────────────────────────┐
# │ sleap_nn_train.sh  « fine-tune a top-down pair, then score » │
# └──────────────────────────────────────────────────────────────┘
# Train a SLEAP 1.6 (sleap-nn, PyTorch) centroid + centred-instance pair that
# starts from an existing model (old TF .h5 or sleap-nn .ckpt), then compare
# it with the current model on every validation set.
#
# Usage (from $PROJ):
#   sbatch scripts/slurm/sleap_nn_train.sh <train.pkg.slp> <run_name> <base_run> \
#       [<compare_run>] [<name>=<val.pkg.slp> ...]
#
# Outputs: models/<run>.centroid/, models/<run>.centered_instance/
#          models/<run>.eval.json   one row per (model, validation set)
#          models/<run>.done        written last; rigconductor polls for it
set -euo pipefail

PROJ="${PROJ:-/projects/sciences/zoology/geurten_lab/HeatAggression}"
TRAIN="${1:?Usage: sleap_nn_train.sh <train.slp> <run> <base_run> [compare_run] [name=val.slp ...]}"
RUN="${2:?run name missing}"
BASE="${3:?base run missing}"
COMPARE="${4:-$BASE}"
shift 4 2>/dev/null || shift $#
VALS=("$@")
NN_ENV="${NN_ENV:-$PROJ/envs/uv-tools/sleap}"
EPOCHS="${EPOCHS:-200}"
N_ANIMALS="${N_ANIMALS:-3}"
PY="$NN_ENV/bin/python"
NN="$NN_ENV/bin/sleap-nn"
cd "$PROJ"
sleep 5   # let the project mount surface on a fresh node

echo "=== sleap-nn training ==="
echo "Job      : ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-?} (${SLURM_JOB_PARTITION:-?})"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo none)"
echo "Env      : $NN_ENV  sleap-nn $("$PY" -c 'import sleap_nn; print(sleap_nn.__version__)')"
echo "Train    : $TRAIN"
echo "Run      : $RUN   base: $BASE   compare: $COMPARE"
echo "Val sets : ${VALS[*]:-none}"
echo "Start    : $(date -Is)"
echo "========================="

# Fail fast: PyTorch must see the GPU, or training silently runs on CPU.
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU visible to PyTorch"
EOF

WORK="$PROJ/runs/train_${RUN}_${SLURM_JOB_ID:-local}"
mkdir -p "$WORK" "$PROJ/models"
rm -f "$PROJ/models/$RUN.done" "$PROJ/models/$RUN.failed"
trap 'echo "failed at line $LINENO" > "$PROJ/models/$RUN.failed"' ERR

# The validation sets only score the models: sleap-nn picks its checkpoints on
# 10 % of the training frames (nn_config.py), so the scores stay unbiased.
for head in centroid centered_instance; do
  t0=$(date +%s)
  "$PY" scripts/sleap/nn_config.py "models/$BASE.$head" "$TRAIN" "$WORK/$head.yaml" \
    --ckpt-dir "$PROJ/models" --run "$RUN.$head" --epochs "$EPOCHS"
  "$NN" train --config "$WORK/$head.yaml"
  echo "$head trained in $(( $(date +%s) - t0 )) s"
done

rows=()
for spec in "${VALS[@]}"; do
  name="${spec%%=*}"; val="${spec#*=}"
  for model in "$RUN" "$COMPARE"; do
    pred="$WORK/pred_${model}_${name}.slp"
    "$NN" track -i "$val" --only_labeled_frames \
      -m "models/$model.centroid" -m "models/$model.centered_instance" \
      --max_instances "$N_ANIMALS" -o "$pred"
    "$PY" scripts/sleap/nn_eval.py "$val" "$pred" "$WORK/eval_${model}_${name}.json" \
      --model "$model" --set "$name"
    rows+=("$WORK/eval_${model}_${name}.json")
  done
done
"$PY" - "$PROJ/models/$RUN.eval.json" "${rows[@]}" <<'EOF'
import json, sys
from pathlib import Path
rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[2:]]
Path(sys.argv[1]).write_text(json.dumps(rows, indent=2), encoding="utf-8")
EOF
date -Is > "$PROJ/models/$RUN.done"
echo "End      : $(date -Is)"
echo "Models   : models/$RUN.centroid  models/$RUN.centered_instance"
echo "Eval     : models/$RUN.eval.json"

#!/bin/bash
#SBATCH --job-name=sleap-train
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu_L40,aoraki_gpu_A100_40GB,aoraki_gpu_A100_80GB,aoraki_gpu_L4_24GB,aoraki_gpu_RTX3090,aoraki_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log
#
# Train a SLEAP top-down model pair (centroid + centred instance) for the
# triad arena, starting from the public flies13 labels plus our own.
#
# Usage (from the project dir on the cluster):
#   cd $PROJ && sbatch scripts/slurm/sleap_train.sh <labels.slp> [run_name]
#
# Fine-tune from an existing model pair (same architecture):
#   BASE_RUN=flies13_base CFG_DIR=$PROJ/configs/finetune \
#     sbatch scripts/slurm/sleap_train.sh sleap_labels/lab_v1.pkg.slp lab_v1
#
# Inputs:  <labels.slp>  merged SLEAP labels (flies13 + lab frames)
# Outputs: $PROJ/models/<run_name>.centroid/ and <run_name>.centered_instance/
#          (runs_folder "models" comes from the config; do NOT pass --prefix,
#          SLEAP prepends it to the run name instead of the path)
set -euo pipefail

PROJ="${PROJ:-/projects/sciences/zoology/geurten_lab/HeatAggression}"
LABELS="${1:?Usage: sbatch sleap_train.sh <labels.slp> [run_name]}"
RUN="${2:-$(date +%Y%m%d)_triad}"
CFG="${CFG_DIR:-$PROJ/configs}"
BASE_RUN="${BASE_RUN:-}"

# The env's activate hook appends to LD_LIBRARY_PATH, which is unset on
# compute nodes and trips `set -u`; define it first.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${SLEAP_ENV:-sleap}"
# The sleap-* console scripts fail on this env (pkg_resources cannot resolve
# setuptools for pynwb); the module entry points work, so use python -m.

echo "=== SLEAP training ==="
echo "Job      : $SLURM_JOB_ID on $SLURMD_NODENAME ($SLURM_JOB_PARTITION)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python   : $(which python)  sleap $(python -c 'import sleap;print(sleap.__version__)')"
echo "Labels   : $LABELS"
echo "Run      : $RUN"
echo "Base     : ${BASE_RUN:-none}  configs: $CFG"
echo "Start    : $(date -Is)"
echo "======================"

# Fail fast if TensorFlow cannot see the GPU — otherwise it silently trains on CPU.
python - <<'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TF GPUs:", gpus)
assert gpus, "No GPU visible to TensorFlow"
EOF

mkdir -p "$PROJ/models" "$PROJ/logs"
cd "$PROJ"

t0=$(date +%s)
base_args() { [ -n "$BASE_RUN" ] && echo "--base_checkpoint $PROJ/models/$BASE_RUN.$1"; true; }
# shellcheck disable=SC2046
python -m sleap.nn.training "$CFG/centroid.json" "$LABELS" --run_name "$RUN.centroid" $(base_args centroid)
echo "centroid done in $(( $(date +%s) - t0 )) s"

t1=$(date +%s)
# shellcheck disable=SC2046
python -m sleap.nn.training "$CFG/centered_instance.json" "$LABELS" --run_name "$RUN.centered_instance" \
  $(base_args centered_instance)
echo "centered_instance done in $(( $(date +%s) - t1 )) s"

echo "End      : $(date -Is)"
echo "Models   : $PROJ/models/$RUN.centroid  $PROJ/models/$RUN.centered_instance"

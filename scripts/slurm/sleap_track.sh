#!/bin/bash
#SBATCH --job-name=sleap-track
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_L40,aoraki_gpu_L4_24GB,aoraki_gpu_RTX3090,aoraki_gpu_A100_40GB,aoraki_gpu_A100_80GB,aoraki_gpu_H100,aoraki_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.log
#
# Run SLEAP inference + tracking on every arena clip of one trial.
# Submitted as an array (one task per clip) by thermokourt's Aoraki hand-off.
#
# Usage:
#   cd $PROJ && sbatch --array=0-9 scripts/slurm/sleap_track.sh <trial_id> <run_name>
#
# Inputs : $PROJ/clips/<trial_id>/<trial_id>_arena_NN.mp4   (NN = array index)
#          $PROJ/models/<run_name>.centroid, <run_name>.centered_instance
# Outputs: $PROJ/pose/<trial_id>/<trial_id>_arena_NN.slp  and  .h5 (analysis export)
set -euo pipefail

PROJ="${PROJ:-/projects/sciences/zoology/geurten_lab/HeatAggression}"
TRIAL="${1:?Usage: sbatch --array=0-N sleap_track.sh <trial_id> <run_name>}"
RUN="${2:?run_name missing}"
N_ANIMALS="${N_ANIMALS:-3}"          # 2 males + decapitated female
INVERT="${INVERT:-1}"                # 1: invert video (our flies are dark on white; the
                                     #    flies13-based models expect bright flies on black)
PREPROC="${PREPROC:-}"               # extra ffmpeg -vf filters after the inversion, e.g.
                                     #   "scale=358:358" or "eq=contrast=0.5" (test variants)
FRAMES="${FRAMES:-}"                 # e.g. "0-3000" for a quick test; empty = all
IDX=$(printf "%02d" "${SLURM_ARRAY_TASK_ID:-0}")

CLIP="$PROJ/clips/$TRIAL/${TRIAL}_arena_${IDX}.mp4"
OUT_DIR="$PROJ/pose/$TRIAL"
OUT="$OUT_DIR/${TRIAL}_arena_${IDX}.slp"

# The env's activate hook appends to LD_LIBRARY_PATH, which is unset on
# compute nodes and trips `set -u`; define it first.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
# TensorFlow grabs the whole GPU by default; with growth on, several clips can
# share one card (sleap_worker.sh packs 3-5 per GPU).
export TF_FORCE_GPU_ALLOW_GROWTH=true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${SLEAP_ENV:-sleap}"
# The sleap-* console scripts fail on this env (pkg_resources cannot resolve
# setuptools for pynwb); the module entry points work, so use python -m.

echo "=== SLEAP tracking ==="
echo "Job   : $SLURM_JOB_ID[$IDX] on $SLURMD_NODENAME"
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Clip  : $CLIP"
echo "Model : $RUN"
echo "Start : $(date -Is)"
echo "======================"

[ -s "$CLIP" ] || { echo "clip missing: $CLIP"; exit 2; }
mkdir -p "$OUT_DIR"

# Stage to node-local scratch: many small reads from /projects are slow.
SCR="${SLURM_TMPDIR:-/tmp}/sleap_${SLURM_JOB_ID}_${IDX}"
mkdir -p "$SCR"
LOCAL="$SCR/$(basename "$CLIP")"
VF=""
[ "$INVERT" = "1" ] && VF="negate"
[ -n "$PREPROC" ] && VF="${VF:+$VF,}$PREPROC"
if [ -n "$VF" ]; then
  ffmpeg -v error -y -i "$CLIP" -vf "$VF" -pix_fmt yuv420p -c:v libx264 -preset veryfast -crf 16 -an "$LOCAL"
  echo "preprocessed clip (-vf $VF) -> $LOCAL"
else
  cp "$CLIP" "$LOCAL"
fi
FRAME_ARGS=()
[ -n "$FRAMES" ] && FRAME_ARGS=(--frames "$FRAMES")

t0=$(date +%s)
# `python -m sleap.nn.inference` imports but never runs main() in 1.4.1; call it.
python -c "import sys; from sleap.nn.inference import main; main(sys.argv[1:])" "$LOCAL" \
  -m "$PROJ/models/$RUN.centroid" \
  -m "$PROJ/models/$RUN.centered_instance" \
  --tracking.tracker simple \
  --tracking.similarity centroid \
  --tracking.match hungarian \
  --tracking.track_window 5 \
  --tracking.target_instance_count "$N_ANIMALS" \
  --tracking.post_connect_single_breaks 1 \
  --tracking.max_tracking 1 \
  --tracking.max_tracks "$N_ANIMALS" \
  --max_instances "$N_ANIMALS" \
  --batch_size 16 \
  "${FRAME_ARGS[@]}" \
  -o "$OUT"
echo "inference+tracking: $(( $(date +%s) - t0 )) s"

# An arena without animals gives a .slp with no instances (and no skeleton), on
# which the analysis export crashes. Record it as empty instead of failing.
# sleap's imports log to stdout, so only the last line is the count.
N_INST=$(python -c "import sys, sleap; l = sleap.load_file(sys.argv[1]); print(sum(len(f.instances) for f in l))" "$OUT" 2>/dev/null | tail -n 1)
echo "instances found: $N_INST"
if [ "$N_INST" = "0" ]; then
  echo "no animals found in this clip" > "${OUT%.slp}.empty"
  rm -rf "$SCR"
  echo "End   : $(date -Is)"
  echo "Out   : no animals found -> ${OUT%.slp}.empty"
  exit 0
fi
rm -f "${OUT%.slp}.empty"

# Analysis HDF5 (tracks, node names, scores) for thermokourt.posttrack
python -m sleap.io.convert --format analysis -o "${OUT%.slp}.h5" "$OUT"
rm -rf "$SCR"

echo "End   : $(date -Is)"
echo "Out   : $OUT  ${OUT%.slp}.h5"

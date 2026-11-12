#!/usr/bin/env bash
#
# ┌──────────────────────────────────────────────────────────────┐
# │ sleap_submit.sh  « opportunistic SLEAP across the GPU pool » │
# └──────────────────────────────────────────────────────────────┘
# Queue the clips of one trial and start pull-queue workers (sleap_worker.sh)
# on several GPU partitions. Each GPU partition allows only 2 running jobs per
# user (QOS p_aoraki_gpu_*: MaxJobsPU=2), so one array over a partition list
# stalls; instead every worker gets its own single-partition sbatch. Workers
# that start after the queue is empty exit within seconds.
#
# Usage:  cd $PROJ && bash scripts/slurm/sleap_submit.sh <trial_id> <run_name> <idx,idx,...>
# Prints  JOBS=<id>,<id>,...  as its last line.
# Env:    PARTS (space separated), MAX_PER_PART (2), SLOTS (auto per card),
#         N_ANIMALS / INVERT / PREPROC / FRAMES passed through to sleap_track.sh.
# The TF back end (SLEAP 1.4.1: TF 2.7 / CUDA 11.2, kernels up to compute_80)
# leaves out the H100 (sm_90): it JIT-compiles every kernel from PTX first,
# ~14 min per worker, then predicts no faster than an A100. sleap-nn models
# (models/<run>.centroid/best.ckpt, PyTorch cu128) do use it.
# RTX6000 (Blackwell, sm_120) and H200 are left out; smoke-test before adding.
set -uo pipefail

PROJ="${PROJ:-/projects/sciences/zoology/geurten_lab/HeatAggression}"
TRIAL="${1:?Usage: sleap_submit.sh <trial_id> <run_name> <idx,idx,...>}"
RUN="${2:?run_name missing}"
IDXS="${3:?arena indices missing}"
TF_PARTS="aoraki_gpu_L40 aoraki_gpu_L4_24GB aoraki_gpu_RTX3090 aoraki_gpu_A100_40GB aoraki_gpu_A100_80GB"
if [ -f "$PROJ/models/$RUN.centroid/best.ckpt" ]; then
  PARTS="${PARTS:-$TF_PARTS aoraki_gpu_H100}"
else
  PARTS="${PARTS:-$TF_PARTS}"
fi
MAX_PER_PART="${MAX_PER_PART:-2}"
MIN_SLOTS=3
TIME="${TIME:-04:00:00}"

cd "$PROJ" || exit 1
Q="$PROJ/pose/$TRIAL/_queue"
mkdir -p "$Q" logs
tr ',' '\n' <<< "$IDXS" | sed '/^$/d' > "$Q/clips.txt"
n_clips=$(wc -l < "$Q/clips.txt")
for raw in $(cat "$Q/clips.txt"); do       # a new submit gives every clip fresh tries
  idx=$(printf "%02d" "$raw")
  rm -f "$Q/strikes_$idx" "$Q/parked_$idx"
done
find "$Q" -maxdepth 1 -name 'claim_*' -mmin +5 -exec rm -rf {} + 2>/dev/null

read -r -a parts <<< "$PARTS"
n_parts=${#parts[@]}
want=$(( (n_clips + MIN_SLOTS - 1) / MIN_SLOTS ))
[ "$want" -lt "$n_parts" ] && want=$n_parts
max=$(( n_parts * MAX_PER_PART ))
[ "$want" -gt "$max" ] && want=$max

jobs=()
for i in $(seq 0 $(( want - 1 ))); do
  p=${parts[$(( i % n_parts ))]}
  case "$p" in
    aoraki_gpu_L4_24GB|aoraki_gpu_RTX3090) cpus=8;  mem=48G ;;   # QOS cap: cpu=8, mem=60G
    *)                                     cpus=12; mem=64G ;;
  esac
  jid=$(sbatch --parsable --partition="$p" --cpus-per-task="$cpus" --mem="$mem" \
          --time="$TIME" --job-name="sleap-w-${TRIAL: -15}" \
          --output="logs/sleap-worker_%j.log" --export=ALL \
          scripts/slurm/sleap_worker.sh "$TRIAL" "$RUN")
  if [ -n "$jid" ]; then
    jobs+=("${jid%%;*}")
    echo "worker ${jid%%;*} on $p ($cpus cpus, $mem)"
  else
    echo "sbatch refused a worker on $p"
  fi
done
[ "${#jobs[@]}" -gt 0 ] || { echo "no worker could be submitted"; exit 1; }
echo "$n_clips clips, ${#jobs[@]} workers"
echo "JOBS=$(IFS=,; echo "${jobs[*]}")"

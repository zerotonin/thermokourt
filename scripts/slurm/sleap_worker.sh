#!/usr/bin/env bash
#SBATCH --job-name=sleap-worker
#SBATCH --account=geuba03p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#
# ┌──────────────────────────────────────────────────────────────┐
# │ sleap_worker.sh  « one GPU, several clips at once »          │
# └──────────────────────────────────────────────────────────────┘
# A pull-queue worker (Aoraki opportunistic pattern). sleap_submit.sh starts
# a few of these on different GPU partitions; each one packs SLOTS clips onto
# its card and keeps pulling clips of one trial until none are left.
#
# Queue: $PROJ/pose/<trial>/_queue/
#   clips.txt         arena indices to track (written by sleap_submit.sh)
#   claim_NN/         atomic mkdir = arena NN taken; owner file, heartbeat mtime
#   strikes_NN        failures so far; parked_NN after MAX_STRIKES
#   arena_NN.log      sleap_track.sh output for that arena
# Done = <trial>_arena_NN.h5 or .empty exists in $PROJ/pose/<trial>/.
#
# Usage (normally via sleap_submit.sh):
#   sbatch --partition=aoraki_gpu_L40 scripts/slurm/sleap_worker.sh <trial_id> <run_name>
# No `set -e`: one failing clip must not take the other slots down.
set -uo pipefail

PROJ="${PROJ:-/projects/sciences/zoology/geurten_lab/HeatAggression}"
TRIAL="${1:?Usage: sleap_worker.sh <trial_id> <run_name>}"
RUN="${2:?run_name missing}"
POSE="$PROJ/pose/$TRIAL"
Q="$POSE/_queue"
STALE_MIN="${STALE_MIN:-5}"      # claim not touched for this long = dead worker
MAX_STRIKES="${MAX_STRIKES:-3}"
HEARTBEAT_S=60
sleep 5                          # let the project mount surface

vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "${SLOTS:-}" ]; then     # clips per card; SLEAP is light, 24 GB cards take 3
  if [ "${vram_mib:-0}" -ge 70000 ]; then SLOTS=5
  elif [ "${vram_mib:-0}" -ge 40000 ]; then SLOTS=4
  else SLOTS=3; fi
fi

echo "=== SLEAP worker ==="
echo "Job   : ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} (${SLURM_JOB_PARTITION:-?})"
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) (${vram_mib:-?} MiB)"
echo "Trial : $TRIAL   model: $RUN   slots: $SLOTS"
echo "Start : $(date -Is)"
echo "===================="

is_done()  { [ -e "$POSE/${TRIAL}_arena_$1.h5" ] || [ -e "$POSE/${TRIAL}_arena_$1.empty" ]; }
is_stale() { [ -n "$(find "$Q/claim_$1" -maxdepth 0 -mmin +"$STALE_MIN" 2>/dev/null)" ]; }

next_clip() {
  local raw idx
  while read -r raw; do
    [ -z "$raw" ] && continue
    idx=$(printf "%02d" "$raw")
    is_done "$idx" && continue
    [ -e "$Q/parked_$idx" ] && continue
    if [ -d "$Q/claim_$idx" ]; then
      is_stale "$idx" || continue
      rm -rf "$Q/claim_$idx"      # its worker died; take it over
    fi
    if mkdir "$Q/claim_$idx" 2>/dev/null; then
      echo "${SLURM_JOB_ID:-local} ${SLURMD_NODENAME:-$(hostname)} $(date +%s)" > "$Q/claim_$idx/owner"
      echo "$idx"
      return 0
    fi
  done < "$Q/clips.txt"
  return 1
}

run_clip() {
  local idx=$1 hb rc t0
  ( while sleep "$HEARTBEAT_S"; do touch "$Q/claim_$idx"; done ) &
  hb=$!
  t0=$(date +%s)
  SLURM_ARRAY_TASK_ID=$((10#$idx)) bash "$PROJ/scripts/slurm/sleap_track.sh" "$TRIAL" "$RUN" \
    > "$Q/arena_$idx.log" 2>&1
  rc=$?
  kill "$hb" 2>/dev/null
  if [ "$rc" -eq 0 ]; then
    echo "arena $idx: done in $(( $(date +%s) - t0 )) s"
    rm -f "$Q/strikes_$idx"
  else
    local n=$(( $(cat "$Q/strikes_$idx" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$Q/strikes_$idx"
    echo "arena $idx: FAILED (rc $rc, strike $n/$MAX_STRIKES), see $Q/arena_$idx.log"
    [ "$n" -ge "$MAX_STRIKES" ] && touch "$Q/parked_$idx"
  fi
  rm -rf "$Q/claim_$idx"
}

slot() {
  local idx
  while idx=$(next_clip); do
    run_clip "$idx"
  done
}

[ -s "$Q/clips.txt" ] || { echo "no queue at $Q/clips.txt"; exit 2; }
for _ in $(seq "$SLOTS"); do
  slot &
  sleep 2                        # stagger model loading
done
wait
echo "End   : $(date -Is) (no clips left)"

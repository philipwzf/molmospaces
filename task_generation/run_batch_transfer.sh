#!/usr/bin/env bash
#
# Generate N transfer-food tasks on random tables using
# empty_scene_pipeline.  Each episode runs in its own process
# (Isaac Sim cannot be re-launched within a single process).
#
# Usage:
#   bash run_batch_transfer.sh              # 100 episodes, 300 steps, default
#   bash run_batch_transfer.sh 50 100       # 50 episodes, 100 steps
#   N=100 STEPS=300 PARALLEL=4 bash run_batch_transfer.sh
#
# Outputs land in  outputs/pipeline_runs/batch_transfer_<timestamp>/
# with one sub-directory per episode.

set -euo pipefail

N="${N:-${1:-100}}"
STEPS="${STEPS:-${2:-300}}"
PARALLEL="${PARALLEL:-${3:-4}}"          # concurrent jobs
BASE_SEED="${BASE_SEED:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"            # set to 0 to skip video

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="$PROJECT_ROOT/outputs/pipeline_runs/batch_transfer_${TIMESTAMP}"
mkdir -p "$BATCH_DIR"

export OMNIGIBSON_HEADLESS=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

VIDEO_FLAG=""
[ "$SAVE_VIDEO" = "1" ] && VIDEO_FLAG="--save-video"

echo "[Batch] Episodes: $N, Steps: $STEPS, Parallel: $PARALLEL"
echo "[Batch] Output:   $BATCH_DIR"
echo "[Batch] Started:  $(date)"

run_episode() {
    local ep="$1"
    local seed=$((BASE_SEED + ep * 1000))
    local run_dir="$BATCH_DIR/ep$(printf '%03d' "$ep")"
    local log_file="$run_dir/run.log"
    mkdir -p "$run_dir"

    echo "[Batch] Starting episode $ep (seed=$seed)"

    python -m omnigibson.task_generation.empty_scene_pipeline \
        --setup transfer \
        --episodes 1 \
        --steps "$STEPS" \
        --seed "$seed" \
        --run-dir "$run_dir" \
        --no-strict-gate \
        $VIDEO_FLAG \
        > "$log_file" 2>&1

    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[Batch] Episode $ep done (seed=$seed)"
    else
        echo "[Batch] Episode $ep FAILED (rc=$rc, seed=$seed)" >&2
    fi
    return $rc
}

export -f run_episode
export BATCH_DIR STEPS VIDEO_FLAG BASE_SEED

# Run episodes in parallel using xargs (available everywhere).
seq 0 $((N - 1)) | xargs -P "$PARALLEL" -I {} bash -c 'run_episode "$@"' _ {}

# Collect diagnostics into a single file.
COMBINED="$BATCH_DIR/diagnostics_all.jsonl"
: > "$COMBINED"
for d in "$BATCH_DIR"/ep*/diagnostics.jsonl; do
    [ -f "$d" ] && cat "$d" >> "$COMBINED"
done

TOTAL=$(wc -l < "$COMBINED")
FAILED=$((N - TOTAL))
echo ""
echo "[Batch] Complete: $TOTAL/$N succeeded, $FAILED failed"
echo "[Batch] Combined diagnostics: $COMBINED"
echo "[Batch] Finished: $(date)"

#!/usr/bin/env bash
# Run the full benchmark suite: 10 tasks per pipeline across different scenes.
# Each pipeline picks 10 eligible scenes (varies per pipeline type).
#
# Usage:
#   bash OmniGibson/omnigibson/task_generation/run_batch_benchmark.sh
#
# Environment:
#   OMNIGIBSON_HEADLESS=1 and CUDA_VISIBLE_DEVICES=0 are set automatically.

set -euo pipefail

export OMNIGIBSON_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="outputs/benchmark_runs/full_${TIMESTAMP}"
mkdir -p "${OUTPUT_ROOT}"

STEPS=100
EPISODES=1
TIMEOUT=600
N_SCENES=10

# Table-based pipelines — pick 10 diverse scenes with tables.
TABLE_SCENES="Rs_int Merom_1_int Beechwood_0_int Wainscott_0_int Pomaria_1_int Ihlen_1_int house_single_floor office_large Benevolence_2_int restaurant_diner"

# Cabinet-based pipelines — pick 10 scenes with cabinets.
CABINET_SCENES="Rs_int Beechwood_0_int Beechwood_1_int Wainscott_0_int Pomaria_1_int Ihlen_1_int Merom_1_int house_single_floor house_double_floor_lower office_cubicles_left"

# Door-based pipelines — scenes with revolute-door cabinets.
DOOR_SCENES="Beechwood_0_int Beechwood_1_int Wainscott_0_int Wainscott_1_int Pomaria_1_int Pomaria_2_int Ihlen_1_int house_single_floor house_double_floor_lower Rs_int"

echo "============================================================"
echo "  SENTINEL-Lite Full Benchmark Suite"
echo "  Output: ${OUTPUT_ROOT}"
echo "  Steps: ${STEPS}, Episodes: ${EPISODES}, Timeout: ${TIMEOUT}s"
echo "  Target: ${N_SCENES} scenes per pipeline"
echo "============================================================"

run_pipeline() {
    local PIPELINE=$1
    local SCENES=$2
    local EXTRA_ARGS=${3:-}

    echo ""
    echo "============================================================"
    echo "  Pipeline: ${PIPELINE}"
    echo "============================================================"

    conda run -n behavior python -m omnigibson.task_generation.run_benchmark \
        --pipeline "${PIPELINE}" \
        --scenes ${SCENES} \
        --steps "${STEPS}" \
        --episodes "${EPISODES}" \
        --timeout "${TIMEOUT}" \
        --no-strict-gate \
        --output-dir "${OUTPUT_ROOT}/${PIPELINE}" \
        ${EXTRA_ARGS} \
        2>&1 | tee "${OUTPUT_ROOT}/${PIPELINE}_log.txt"
}

# 1. Table clutter (fragile proximity, open surface)
run_pipeline "table" "${TABLE_SCENES}" "--density medium --randomize"

# 2. Stack retrieval (structural instability)
run_pipeline "stack" "${TABLE_SCENES}" "--stack-height medium"

# 3. Food transfer (no-contact constraint)
run_pipeline "transfer" "${TABLE_SCENES}"

# 4. Cabinet clutter (enclosed space retrieval)
run_pipeline "cabinet" "${CABINET_SCENES}" "--density medium"

# 5. Liquid transport (spill/tilt during motion)
run_pipeline "liquid" "${TABLE_SCENES}" "--difficulty medium"

# 6. Blocked open door (pre-planning: clear before open)
run_pipeline "blocked_door" "${DOOR_SCENES}"

# 7. Blocked close door (pre-planning: clear before close)
run_pipeline "blocked_close" "${DOOR_SCENES}"

echo ""
echo "============================================================"
echo "  Benchmark Complete"
echo "  Output: ${OUTPUT_ROOT}"
echo "============================================================"

# Generate combined summary.
echo ""
echo "Per-pipeline summaries:"
for dir in "${OUTPUT_ROOT}"/*/; do
    if [ -f "${dir}/summary.csv" ]; then
        PIPELINE=$(basename "${dir}")
        TOTAL=$(tail -n +2 "${dir}/summary.csv" | wc -l)
        SUCCESS=$(tail -n +2 "${dir}/summary.csv" | grep -c "success" || true)
        GATE=$(tail -n +2 "${dir}/summary.csv" | grep -c "True" || true)
        echo "  ${PIPELINE}: ${SUCCESS}/${TOTAL} success, ${GATE}/${TOTAL} gate pass"
    fi
done

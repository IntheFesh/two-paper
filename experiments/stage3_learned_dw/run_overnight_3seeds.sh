#!/usr/bin/env bash
# run_overnight_3seeds.sh — robust multi-seed Stage-3 run.
#
# Calls run_stage3.sh three times (seed = 0, 1, 2) at 1M env steps per
# agent. Each seed gets its own log file and its phase1 results JSON is
# preserved separately so the LAST seed does not silently overwrite
# earlier seeds' verdicts.
#
# Expected wall-clock:
#   - Consumer GPU (RTX 3090 / 4090 class): ~2-4 hours total.
#   - 8-core CPU (no GPU):                  ~6-7 hours total.
#
# A failed seed does NOT abort the rest of the run -- the next seed
# still launches, so an overnight run yields as much data as possible.
#
# Usage:
#     cd experiments/stage3_learned_dw/
#     nohup bash run_overnight_3seeds.sh > overnight_master.log 2>&1 &

set -uo pipefail   # NOTE: deliberately no -e, so a failed seed does not abort the loop

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# Activate the local venv if present (created during setup).
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.venv/bin/activate"
fi

STEPS="${STAGE3_PHASE1_STEPS:-1000000}"
EVAL_EPS="${STAGE3_PHASE1_EVAL_EPISODES:-80}"
SEEDS="${STAGE3_OVERNIGHT_SEEDS:-0 1 2}"

echo "=================================================================="
echo "Stage-3 multi-seed overnight run"
echo "  start time : $(date)"
echo "  seeds      : $SEEDS"
echo "  steps/agent: $STEPS"
echo "  eval episodes: $EVAL_EPS"
echo "=================================================================="

mkdir -p "$HERE/phase1_results"

declare -A SEED_STATUS
for seed in $SEEDS; do
    echo
    echo "------------------------------------------------------------------"
    echo "[seed=$seed] starting at $(date)"
    echo "------------------------------------------------------------------"
    log="$HERE/stage3_seed${seed}.log"
    t0=$SECONDS
    STAGE3_PHASE1_STEPS="$STEPS" \
    STAGE3_PHASE1_EVAL_EPISODES="$EVAL_EPS" \
    STAGE3_PHASE1_SEED="$seed" \
        bash "$HERE/run_stage3.sh" > "$log" 2>&1
    rc=$?
    elapsed=$((SECONDS - t0))
    # Preserve per-seed results regardless of exit code -- a FAIL verdict
    # is still data we want to inspect; the original overnight run lost
    # all per-seed JSONs because they were only copied on exit 0.
    if [ -f "$HERE/phase1_results/phase1_results.json" ]; then
        cp "$HERE/phase1_results/phase1_results.json" \
           "$HERE/phase1_results/phase1_results_seed${seed}.json"
    fi
    if [ -f "$HERE/phase1_results/dw_hat.pt" ]; then
        cp "$HERE/phase1_results/dw_hat.pt" \
           "$HERE/phase1_results/dw_hat_seed${seed}.pt"
    fi
    if [ $rc -eq 0 ]; then
        SEED_STATUS[$seed]="OK (${elapsed}s)"
        echo "[seed=$seed] OK at $(date) (${elapsed}s)"
    else
        SEED_STATUS[$seed]="FAIL exit=$rc (${elapsed}s)"
        echo "[seed=$seed] FAILED (exit $rc) at $(date) (${elapsed}s) -- continuing"
    fi
done

echo
echo "=================================================================="
echo "All seeds finished at $(date)"
echo "------------------------------------------------------------------"
for seed in $SEEDS; do
    printf "  seed=%s : %s\n" "$seed" "${SEED_STATUS[$seed]}"
done
echo "------------------------------------------------------------------"
echo "Per-seed logs    : stage3_seed{0,1,2}.log"
echo "Per-seed verdict : phase1_results/phase1_results_seed{0,1,2}.json"
echo "Per-seed dw_hat  : phase1_results/dw_hat_seed{0,1,2}.pt"
echo "=================================================================="

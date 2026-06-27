#!/usr/bin/env bash
# run_stage3.sh — Stage-3 learned-D_w embodied experiment driver.
#
# Per the task spec, this is the ONE-BUTTON driver: it runs Phase 0
# (FA kill-gate) and, only if Phase 0 PASSES, automatically continues
# to Phase 1 (MiniGrid deep RL). Phase 0 FAIL exits non-zero and does
# NOT proceed to Phase 1.
#
# Usage:
#     bash experiments/stage3_learned_dw/run_stage3.sh
#
#     # Optional Phase 1 knobs (defaults in phase1.py):
#     STAGE3_PHASE1_STEPS=500000 \
#     STAGE3_PHASE1_EVAL_EPISODES=80 \
#         bash experiments/stage3_learned_dw/run_stage3.sh
#
#     # Skip Phase 1 entirely (rare; e.g. CI smoke):
#     bash experiments/stage3_learned_dw/run_stage3.sh --phase0-only
#
# Runtime targets:
#   - Phase 0:  ~3 s, single CPU core.
#   - Phase 1:  ~30-60 min on a single consumer GPU at the default
#               200k env steps per agent (6 agents = 3 agents x 2 twins
#               + ~40 s D_w-hat pretraining + ~30 s eval). Each
#               individual training is well under the spec's 2 h
#               single-training cap, so no user re-confirmation is
#               required at the default settings. On CPU expect ~4-5 h.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# Activate the local venv if present (created during setup).
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.venv/bin/activate"
fi

# --- Phase 0 -----------------------------------------------------------
echo "=========================================="
echo "Stage-3 learned-D_w — Phase 0 (FA kill-gate)"
echo "=========================================="
if ! python -u "$HERE/phase0.py"; then
    echo
    echo "Phase 0 FAILED. Do NOT proceed to Phase 1."
    echo "Diagnose the failed condition above before retrying."
    exit 1
fi
echo
echo "Phase 0 PASSED."

# Honour an explicit --phase0-only escape hatch.
if [ "${1:-}" = "--phase0-only" ]; then
    echo
    echo "Phase 1 skipped by request (--phase0-only)."
    exit 0
fi

# --- Phase 1 -----------------------------------------------------------
echo
echo "=========================================="
echo "Stage-3 learned-D_w — Phase 1 (MiniGrid deep RL)"
echo "=========================================="
if [ ! -f "$HERE/phase1.py" ]; then
    echo "Phase 1 driver missing at $HERE/phase1.py."
    exit 2
fi
# Use -u so PPO progress prints reach the log file in real time when
# the script is launched under `nohup ... > stage3_train.log 2>&1 &`.
python -u "$HERE/phase1.py"

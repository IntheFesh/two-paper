#!/usr/bin/env bash
# run_stage3.sh — Stage-3 learned-D_w embodied experiment driver.
#
# Phase 0 (FA kill-gate) runs ALWAYS. If it FAILS we exit non-zero and
# do NOT proceed to Phase 1; the task spec is explicit that Phase 0
# FAIL means the mechanism does not survive approximation in the
# cheapest setting and Phase 1 would just amplify the failure.
#
# Phase 1 (MiniGrid deep RL) is gated on Phase 0 PASS AND a separate
# opt-in flag (the spec requires user confirmation before any single
# training estimated >2 h or >$30 in GPU). Pass --phase1 to opt in:
#
#     bash experiments/stage3_learned_dw/run_stage3.sh           # Phase 0 only
#     bash experiments/stage3_learned_dw/run_stage3.sh --phase1  # Phase 0 + Phase 1
#
# Runtime targets:
#   - Phase 0:  ~3 s, single CPU core.
#   - Phase 1:  bounded by --max-gpu-h (default 20); each individual
#               training step must estimate < 2 h or pause for
#               confirmation per spec.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# Activate the local venv if present (created by the agent during setup).
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.venv/bin/activate"
fi

echo "=========================================="
echo "Stage-3 learned-D_w — Phase 0 (FA kill-gate)"
echo "=========================================="
if ! python "$HERE/phase0.py"; then
    echo
    echo "Phase 0 FAILED. Do NOT proceed to Phase 1."
    echo "Diagnose the failed condition above before retrying."
    exit 1
fi
echo
echo "Phase 0 PASSED."

if [ "${1:-}" != "--phase1" ]; then
    echo
    echo "Phase 1 not requested. Pass --phase1 to opt in once the GPU"
    echo "budget (estimated cost + wall-clock) has been confirmed with the user."
    exit 0
fi

echo
echo "=========================================="
echo "Stage-3 learned-D_w — Phase 1 (MiniGrid deep RL)"
echo "=========================================="
if [ ! -f "$HERE/phase1.py" ]; then
    echo "Phase 1 driver not yet implemented at $HERE/phase1.py."
    echo "Implement Phase 1 only AFTER user has confirmed the GPU budget."
    exit 2
fi
python "$HERE/phase1.py"

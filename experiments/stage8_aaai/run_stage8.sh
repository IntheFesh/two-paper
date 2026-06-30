#!/usr/bin/env bash
# experiments/stage8_aaai/run_stage8.sh
# =====================================
#
# Run all four Stage-8 blocks sequentially and emit an aggregate verdict.
# Each block is independently runnable.  A non-zero exit code from any
# block marks that block as FAILED in the aggregate report, but later
# blocks still run (so we always get a full picture).
#
# Block verdicts at the per-block level:
#   block 1: PASS / FAIL
#   block 2: PASS / FAIL
#   block 3: PASS / PARTIAL / FAIL
#   block 4: PASS / PARTIAL / FAIL  (partial is an honest, expected outcome)
#
# Final exit code: 0 iff every block exited 0 (PASS or PARTIAL where the
# block treats PARTIAL as success).
#
# Expected total runtime: ~6 minutes on CPU.  No GPU.

set -u
cd "$(dirname "$0")/../.."

mkdir -p experiments/stage8_aaai/results

declare -a BLOCK_NAMES=(
  "Block 1 -- embodied DoorKey-Lava proxy"
  "Block 2 -- margin theorem validation"
  "Block 3 -- three perturbation families"
  "Block 4 -- non-oracle reach supervision"
)
declare -a BLOCK_FILES=(
  "experiments/stage8_aaai/block1_embodied_proxy.py"
  "experiments/stage8_aaai/block2_margin_theorem.py"
  "experiments/stage8_aaai/block3_perturbation_comparison.py"
  "experiments/stage8_aaai/block4_non_oracle.py"
)
declare -a BLOCK_EXIT=()
declare -a BLOCK_VERDICT=()

OVERALL_EXIT=0

for i in "${!BLOCK_NAMES[@]}"; do
  echo
  echo "############################################################"
  echo "# ${BLOCK_NAMES[$i]}"
  echo "############################################################"
  python "${BLOCK_FILES[$i]}"
  ec=$?
  BLOCK_EXIT+=("$ec")
  if [[ "$ec" -eq 0 ]]; then
    BLOCK_VERDICT+=("PASS or PARTIAL")
  else
    BLOCK_VERDICT+=("FAIL (exit $ec)")
    OVERALL_EXIT=1
  fi
done

echo
echo "============================================================"
echo "Stage-8 aggregate"
echo "============================================================"
for i in "${!BLOCK_NAMES[@]}"; do
  printf "  %-50s %s\n" "${BLOCK_NAMES[$i]}" "${BLOCK_VERDICT[$i]}"
done
echo "============================================================"
echo "Per-block JSON: experiments/stage8_aaai/results/block{1..4}_results.json"
echo "Per-block PDF : experiments/stage8_aaai/results/block{1..4}_*.pdf"
if [[ "$OVERALL_EXIT" -eq 0 ]]; then
  echo "Overall: ALL BLOCKS NON-FAIL"
else
  echo "Overall: AT LEAST ONE BLOCK FAILED (see above)"
fi
exit "$OVERALL_EXIT"

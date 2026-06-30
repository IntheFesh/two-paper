#!/usr/bin/env bash
# stage9_embodied_family/run_stage9.sh
# ================================================
#
# Run the Stage-9 embodied environment family: three genuine 2D
# gridworlds with structurally distinct irreversibility types, each
# evaluated for the MRC mechanism (separation / recovery / collapse) and
# the margin theorem, with a CountedMDP cheat-check on every rollout.
#
# Pure CPU; a few minutes total.  No GPU.
#
# Exit code 0 iff the family verdict is PASS (all three environments pass
# and the margin theorem is clean in each).

set -u
cd "$(dirname "$0")/.."

mkdir -p stage9_embodied_family/results

echo "############################################################"
echo "# Stage-9 embodied environment family"
echo "############################################################"

# Quick standalone exact-model sanity per environment (fast fail surface).
for env in env1_doorkey_lava env2_sokoban_barrier env3_resource_depletion; do
  echo
  echo "--- exact-model sanity: ${env} ---"
  python "stage9_embodied_family/${env}.py"
done

echo
echo "############################################################"
echo "# Full family evaluation (learned world models, all seeds)"
echo "############################################################"
python stage9_embodied_family/run_stage9.py
ec=$?

echo
if [[ "$ec" -eq 0 ]]; then
  echo "Stage-9: FAMILY PASS"
else
  echo "Stage-9: FAMILY NOT PASS (see verdict table above)"
fi
exit "$ec"

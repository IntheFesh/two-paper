#!/usr/bin/env bash
# experiments/stage10_minigrid/run_stage10.sh
# ===========================================
#
# Run the Stage-10 native-MiniGrid MRC validation: exact-model sanity,
# then the full learned-world-model mechanism (separation / recovery /
# collapse) + margin-preservation theorem, with a CountedMDP cheat-check
# on every rollout.
#
# Requires: minigrid (Farama) + gymnasium + torch (CPU) + numpy +
# matplotlib.  Pure CPU; ~2 minutes.  No GPU.
#
# Exit code 0 iff the Stage-10 verdict is PASS.

set -u
cd "$(dirname "$0")/../.."

mkdir -p experiments/stage10_minigrid/results

echo "############################################################"
echo "# Stage-10 native MiniGrid MRC validation"
echo "############################################################"

# Ensure MiniGrid is importable; report clearly if not.
python - <<'PY'
import sys
try:
    import minigrid, gymnasium
    print(f"minigrid {minigrid.__version__}, gymnasium {gymnasium.__version__} OK")
except Exception as e:
    print("MiniGrid import FAILED:", e)
    sys.exit(2)
PY
if [[ $? -ne 0 ]]; then
  echo "Install with: pip install minigrid gymnasium"
  exit 2
fi

echo
echo "--- exact-model sanity (enumerated from the real simulator) ---"
python experiments/stage10_minigrid/stage10_minigrid_env.py

echo
echo "--- full learned-WM evaluation (mechanism + margin) ---"
python experiments/stage10_minigrid/run_stage10.py
ec=$?

echo
if [[ "$ec" -eq 0 ]]; then
  echo "Stage-10: PASS"
else
  echo "Stage-10: NOT PASS (see verdict above)"
fi
exit "$ec"

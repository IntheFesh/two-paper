#!/usr/bin/env bash
# run.sh -- one-click reproduction of the MRC experiment suite.
# ============================================================
#
# Runs every stage's validation in order and prints a verdict table.
# Pure CPU; no GPU required.  Full run is ~15-20 minutes.
#
#   bash run.sh              # run the whole suite
#   MRC_QUICK=1 bash run.sh  # smoke test: headline stages only (~3 min)
#
# The heavy Stage-3 Phase-1 (model-free PPO, 1M steps x 3 seeds -- the
# documented FAILURE that motivated the model-based redesign) is NOT run
# here; see stage3_learned_dw/run_stage3.sh to reproduce it.
#
# Note on Stage-6: its pre-registered verdict is a NEGATIVE result (the
# collapse fragility of a naively-learned D_w).  Its non-zero exit is the
# EXPECTED, honest outcome -- Stage-7 then repairs it.  The summary labels
# it accordingly rather than as a script error.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PYTHON:-python3}"
QUICK="${MRC_QUICK:-0}"

bar() { printf '%.0s=' {1..74}; printf '\n'; }

# ---- dependency check -------------------------------------------------
"$PY" - <<'PY' || { echo "Missing Python deps. Install: pip install numpy torch matplotlib minigrid gymnasium"; exit 3; }
import importlib.util, sys
need = ["numpy", "torch", "matplotlib"]
opt = {"minigrid": "Stage-10", "gymnasium": "Stage-10"}
miss = [m for m in need if importlib.util.find_spec(m) is None]
if miss:
    print("Missing required:", ", ".join(miss)); sys.exit(1)
for m, who in opt.items():
    if importlib.util.find_spec(m) is None:
        print(f"(note) optional '{m}' not installed -- {who} will be skipped")
print("dependencies OK")
PY

# ---- stage runner -----------------------------------------------------
declare -a NAMES=() OUTCOMES=() TIMES=()

run_stage() {
  # $1 = label, $2 = note, $3.. = command
  local label="$1"; local note="$2"; shift 2
  bar; printf '>>> %s   (%s)\n' "$label" "$note"; bar
  local t0 t1 ec
  t0=$(date +%s 2>/dev/null || echo 0)
  "$@"; ec=$?
  t1=$(date +%s 2>/dev/null || echo 0)
  NAMES+=("$label"); TIMES+=("$((t1 - t0))s")
  if [[ "$label" == "Stage-6" ]]; then
    # Pre-registered negative result; non-zero exit is expected.
    OUTCOMES+=("NEGATIVE (expected; repaired by Stage-7)")
  elif [[ "$ec" -eq 0 ]]; then
    OUTCOMES+=("PASS")
  else
    OUTCOMES+=("FAIL (exit $ec)")
  fi
}

# ---- headline (quick) stages -----------------------------------------
run_stage "Stage-1"  "unified V1..V5 exact validation" \
  "$PY" stage1_unified/stage1_unified_validation.py
run_stage "Stage-4"  "exact-dynamics model-based planning" \
  "$PY" stage4_modelbased/stage4_modelbased_planning.py
run_stage "Stage-9"  "embodied family (3 irreversibility types)" \
  "$PY" stage9_embodied_family/run_stage9.py
if "$PY" -c "import minigrid, gymnasium" 2>/dev/null; then
  run_stage "Stage-10" "native MiniGrid environment" \
    "$PY" stage10_minigrid/run_stage10.py
fi

if [[ "$QUICK" != "1" ]]; then
  # ---- full suite ----------------------------------------------------
  run_stage "Stage-2A" "paper figures + Table 1" \
    "$PY" stage2_paper/part_a_figures.py
  run_stage "Stage-2B" "DoorKey-style benchmark" \
    "$PY" stage2_paper/part_b_benchmark.py
  run_stage "Stage-3"  "learned-D_w kill-gate (Phase 0 only)" \
    "$PY" stage3_learned_dw/phase0.py
  run_stage "Stage-5"  "learned world model + decision-time D_w" \
    "$PY" stage5_learned_wm/stage5_learned_wm.py
  run_stage "Stage-6"  "collapse under noisy/imperfect WM" \
    "$PY" stage6_noisy_wm/stage6_noisy_wm.py
  run_stage "Stage-7"  "decision-aware WM repairs collapse" \
    "$PY" stage7_decision_aware_wm/stage7_decision_aware.py
  run_stage "Stage-8"  "AAAI reviewer-response blocks" \
    bash stage8_aaai/run_stage8.sh
fi

# ---- summary ----------------------------------------------------------
echo; bar; echo "MRC suite summary"; bar
printf '  %-9s  %-12s  %s\n' "stage" "time" "outcome"
printf '  %-9s  %-12s  %s\n' "-----" "----" "-------"
for i in "${!NAMES[@]}"; do
  printf '  %-9s  %-12s  %s\n' "${NAMES[$i]}" "${TIMES[$i]}" "${OUTCOMES[$i]}"
done
bar
echo "Per-stage results: stage*/results*.json and stage*/results/*.{json,pdf}"
if [[ "$QUICK" == "1" ]]; then
  echo "(quick mode: ran headline stages only; run 'bash run.sh' for the full suite)"
fi

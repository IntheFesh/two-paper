#!/usr/bin/env bash
# One-click runner for the MRC paper experiments.
#
# Runs, in order:
#   (1) Stage-1 unified validation       (V1..V5 — framework self-consistency)
#   (2) Stage-2 / Part A figures + Table 1 (paper-quality PDFs + CSV)
#   (3) Stage-2 / Part B benchmark        (DoorKey-style matched-twin benchmark)
#
# Pure CPU, numpy + matplotlib only. End-to-end runtime: well under 30 seconds.
# No GPU is touched at any point.
#
# Exit status is the logical AND of all three: a single non-zero exit means
# at least one pre-registered PASS condition was violated; do not retune to
# mask it — diagnose the failure printed above.

set -uo pipefail   # NOT -e: we want to capture per-step status and decide
                   # the overall verdict ourselves.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

PY=${PYTHON:-python3}

bar() { printf '%.0s=' {1..72}; printf '\n'; }
say() { printf '\n'; bar; printf '%s\n' "$*"; bar; }

cd "${REPO_ROOT}"

say "[1/3] Stage-1 unified validation  (V1..V5)"
"${PY}" experiments/stage1_unified/stage1_unified_validation.py
S1=$?

say "[2/3] Stage-2 / Part A — paper-quality figures + Table 1"
"${PY}" experiments/stage2_paper/part_a_figures.py
S2=$?

say "[3/3] Stage-2 / Part B — DoorKey-style benchmark"
"${PY}" experiments/stage2_paper/part_b_benchmark.py
S3=$?

say "Overall summary"
status_word() { [[ "$1" -eq 0 ]] && echo PASS || echo FAIL; }
printf '  Stage-1                   : %s\n' "$(status_word $S1)"
printf '  Stage-2 / Part A figures  : %s\n' "$(status_word $S2)"
printf '  Stage-2 / Part B benchmark: %s\n' "$(status_word $S3)"
printf '\nArtifacts:\n'
printf '  experiments/stage1_unified/results.json\n'
printf '  experiments/stage2_paper/results_part_a.json\n'
printf '  experiments/stage2_paper/results_part_b.json\n'
printf '  experiments/stage2_paper/figures/fig1_gap_vs_dw.pdf\n'
printf '  experiments/stage2_paper/figures/fig2_collapse.pdf\n'
printf '  experiments/stage2_paper/figures/fig3_lambda_phase.pdf\n'
printf '  experiments/stage2_paper/figures/fig4_resource_graph.pdf\n'
printf '  experiments/stage2_paper/figures/table1_properties_scaling.csv\n'
printf '  experiments/stage2_paper/benchmark/benchmark_returns.pdf\n'
printf '  experiments/stage2_paper/benchmark/benchmark_h_sweep.pdf\n'
printf '  experiments/stage2_paper/benchmark/benchmark_results.csv\n'

if [[ $S1 -eq 0 && $S2 -eq 0 && $S3 -eq 0 ]]; then
  printf '\nAll three stages PASS.\n'
  exit 0
else
  printf '\nAt least one stage FAILED — see the failed module above. Do not retune.\n'
  exit 1
fi

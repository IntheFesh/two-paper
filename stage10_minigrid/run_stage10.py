"""
stage10_minigrid/run_stage10.py
============================================

Stage-10 driver -- run the MRC mechanism + margin-preservation checks on a
NATIVE MiniGrid environment and write results + figure.

This is the primary-track experiment that answers the reviewer concern
"does the mechanism depend on the author's hand-written corridors?": it
runs the SAME unified evaluation as Stage-9 (exact-model sanity, four
controls reward_only / mrc-learned / oracle_mrc / full_dp, separation /
recovery / collapse, and the margin-preservation theorem) but on an
environment built on the real Farama MiniGrid engine, with the exact
reachability enumerated offline from the real simulator.

Pre-registered PASS/FAIL (locked before any run):
  PASS iff on the native MiniGrid twin:
    - separation  : learned mrc > reward_only on the irreversible twin
                    (charge_load >= 0.5) for the large majority of seeds;
    - recovery    : observed lambda* matches the margin-consistent
                    threshold and pi_MRC(s_0) = a_safe at lambda = 1;
    - collapse    : reversible twin mrc == reward_only (collapse <= 0.30);
    - margin      : 0 margin-preservation violations.
  PARTIAL / FAIL reported honestly otherwise, with the failure mode.

Cheat-check: every learned-planner rollout passes through
run_closed_loop, which asserts the planner reads no true-env dynamics
during a probe choose(S0).  The MiniGrid simulator is used ONLY for the
one-time offline graph enumeration in build_twin, never by the test-time
planner.

Runtime: pure CPU, ~2 minutes.  No GPU.
"""

import json
import os
import sys
from typing import Any, Dict

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE9_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "stage9_embodied_family"))
for _p in (_STAGE9_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage9_common import evaluate_environment, to_jsonable  # noqa: E402
import stage10_minigrid_env as s10  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def write_figure(res: Dict[str, Any], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_seed = res["mechanism_per_seed"]
    margin_rows = res["margin_rows"]
    recovery = res["recovery"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) four controls x two twins (mean returns).
    ax = axes[0]
    controls = ["reward_only", "mrc_learned", "oracle_mrc", "full_dp"]
    irr = [np.mean([r[f"R_{c}_irr"] for r in per_seed]) for c in controls]
    rev = [np.mean([r[f"R_{c}_rev"] for r in per_seed]) for c in controls]
    x = np.arange(len(controls))
    ax.bar(x - 0.2, irr, width=0.4, label="irreversible twin")
    ax.bar(x + 0.2, rev, width=0.4, label="reversible twin")
    ax.set_xticks(x)
    ax.set_xticklabels(["reward\nonly", "mrc\nlearned", "oracle\nmrc",
                         "full\ndp"], fontsize=8)
    ax.set_ylabel("mean return")
    ax.set_title("(a) native MiniGrid -- four controls x two twins")
    ax.legend(fontsize=8)

    # (b) separation / collapse per seed.
    ax = axes[1]
    charge = [r["charge_load_ratio"] for r in per_seed]
    collapse = [r["collapse_ratio"] for r in per_seed]
    sd = np.arange(len(per_seed))
    ax.bar(sd - 0.2, charge, width=0.4, label="charge_load (irr)",
            color="seagreen")
    ax.bar(sd + 0.2, collapse, width=0.4, label="collapse (rev)",
            color="firebrick")
    ax.axhline(0.5, color="seagreen", ls="--", lw=0.8)
    ax.axhline(0.3, color="firebrick", ls="--", lw=0.8)
    ax.set_xticks(sd)
    ax.set_xticklabels([f"s{r['seed']}" for r in per_seed])
    ax.set_ylabel("ratio")
    ax.set_title("(b) separation & collapse per seed")
    ax.legend(fontsize=8)

    # (c) margin phase diagram (learned).
    ax = axes[2]
    decoy = [(r["reward_margin"], r["cost_gap"]) for r in margin_rows
              if not r["flipped"]]
    safe = [(r["reward_margin"], r["cost_gap"]) for r in margin_rows
             if r["flipped"]]
    if decoy:
        xs, ys = zip(*decoy)
        ax.scatter(xs, ys, c="firebrick", s=14, alpha=0.6, label="decoy")
    if safe:
        xs, ys = zip(*safe)
        ax.scatter(xs, ys, c="steelblue", s=14, alpha=0.6, label="safe")
    allx = [r["reward_margin"] for r in margin_rows]
    line = np.linspace(min([0] + allx), max([0.01] + allx), 50)
    ax.plot(line, line, "k--", lw=1, label="cost_gap = margin")
    ax.set_xlabel("reward margin")
    ax.set_ylabel("cost gap (lambda * dD_w_hat)")
    viol = res["margin"]["n_violations"]
    ax.set_title(f"(c) margin theorem ({viol} violations)")
    ax.legend(fontsize=8)

    fig.suptitle("Stage-10 native MiniGrid -- MRC mechanism + margin theorem",
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> bool:
    import time
    t0 = time.time()
    print("=" * 78)
    print("Stage-10 -- MRC mechanism on a NATIVE MiniGrid environment")
    print("=" * 78)
    print("Engine: Farama minigrid 3.1.0 (real turn/forward dynamics, real "
          "Lava/Goal/Grid).")
    print("Exact reachability enumerated offline by BFS-stepping the real "
          "simulator;\nthe test-time planner uses only the learned world "
          "model's D_w_hat (CountedMDP guard).")

    res = evaluate_environment(s10.SPEC, seeds=SEEDS)

    pdf = os.path.join(_THIS_DIR, "results", "stage10_minigrid.pdf")
    write_figure(res, pdf)
    print(f"\nFigure -> {pdf}")

    v = res["verdict"]
    verdict = "PASS" if v["env_pass"] else (
        "PARTIAL" if (v["separation_ok"] or v["collapse_ok"]) else "FAIL")

    print("\n" + "=" * 78)
    print(f"Stage-10 verdict: {verdict}")
    print(f"  environment: {res['name']} ({res['irreversibility_type']})")
    print(f"  exact sanity : {'OK' if v['sanity_pass'] else 'X'}  "
          f"(D_w irr={res['exact_sanity']['Dw_irr']:.4f}, "
          f"rev={res['exact_sanity']['Dw_rev']:.4f}, "
          f"states irr={res['exact_sanity']['n_states_irr']}, "
          f"rev={res['exact_sanity']['n_states_rev']})")
    print(f"  separation   : {'OK' if v['separation_ok'] else 'X'}  "
          f"({v['n_separation_pass']}/{v['n_seeds']} seeds, "
          f"charge_load mean {v['mean_charge_load']:.2f})")
    print(f"  recovery     : {'OK' if v['recovery_ok'] else 'X'}  "
          f"(lambda* {res['recovery'].get('lam_star')} vs "
          f"lam_min_hat {res['recovery'].get('lam_min_hat'):.4f}, "
          f"pi(lam=1)={res['recovery'].get('policy_at_lam_1')})")
    print(f"  collapse     : {'OK' if v['collapse_ok'] else 'X'}  "
          f"({v['n_collapse_pass']}/{v['n_seeds']} seeds, "
          f"max collapse {v['max_collapse_ratio']:.3f})")
    print(f"  margin       : {'OK' if v['margin_ok'] else 'X'}  "
          f"({res['margin']['n_violations']} violations / "
          f"{res['margin']['n_rows']} rows)")
    if verdict == "PASS":
        print("The MRC mechanism (separation / recovery / collapse) and the "
              "margin theorem\nhold on a native MiniGrid environment -- the "
              "result does not depend on hand-written\ngrids.")
    dt = time.time() - t0
    print(f"  wall time: {dt:.1f} s (CPU only, no GPU)")

    payload = {
        "verdict": verdict,
        "wall_time_s": dt,
        "seeds": SEEDS,
        "engine": "Farama minigrid 3.1.0 + gymnasium",
        "enumeration": ("Exact reachable graph enumerated offline by "
                         "BFS-stepping a ground-truth copy of the real "
                         "MiniGrid simulator; state = (agent_pos, agent_dir, "
                         "decoy_taken, breadcrumb_taken)."),
        "cheat_check": ("run_closed_loop asserts the planner reads no "
                         "true-env dynamics during a probe choose(S0); the "
                         "MiniGrid simulator is used only for the offline "
                         "enumeration, never by the test-time planner."),
        "result": to_jsonable(res),
    }
    out = os.path.join(_THIS_DIR, "results", "stage10_results.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Results -> {out}")
    return verdict == "PASS"


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

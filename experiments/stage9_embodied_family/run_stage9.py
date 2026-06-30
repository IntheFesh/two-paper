"""
experiments/stage9_embodied_family/run_stage9.py
=================================================

Stage-9 driver -- run the unified MRC mechanism evaluation across the
three-environment embodied family and emit a family-level verdict.

The family is designed to answer the reviewer question "is the MRC
mechanism just a single-corridor artifact?".  It comprises three genuine
2D embodied gridworlds whose irreversibility STRUCTURES differ
fundamentally:

  env1  DoorKey-Lava        absorbing-state      (agent enters a trap)
  env2  Sokoban-barrier     environment-state    (box seals a passage)
  env3  Resource-depletion  monotone-resource    (non-renewable fuel)

For each environment we verify, with the SAME framework and the SAME
exact destroyed_mass:
  - separation  (mrc(learned D_w) > reward_only; charge_load >= 0.5)
  - recovery    (lambda* matches the margin-consistent threshold; safe at 1)
  - collapse    (reversible twin: mrc == reward_only; collapse_ratio <= 0.3)
  - margin      (pi_MRC flips at s_0 <=> cost gap crosses reward margin)
and we report four controls (reward_only / mrc-learned / oracle_mrc /
full_dp) on both twins.  Test-time planners read only the learned world
model's D_w_hat; a CountedMDP cheat-check guards every rollout.

PASS/FAIL (pre-registered)
  Family PASS iff ALL THREE environments pass separation + recovery +
                  collapse, AND margin theorem has 0 violations in each.
  PARTIAL     iff some (but not all) environments pass -- report which
                  irreversibility structure fails (still a valid finding).
  FAIL        iff most environments fail.

Runtime: pure CPU, a few minutes total.  No GPU.
"""

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from stage9_common import evaluate_environment, to_jsonable  # noqa: E402
import env1_doorkey_lava as env1  # noqa: E402
import env2_sokoban_barrier as env2  # noqa: E402
import env3_resource_depletion as env3  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
ENVS = [env1.SPEC, env2.SPEC, env3.SPEC]


def write_env_figure(env_result: Dict[str, Any], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = env_result["name"]
    per_seed = env_result["mechanism_per_seed"]
    margin_rows = env_result["margin_rows"]
    recovery = env_result["recovery"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) Mechanism: four controls on both twins (mean returns).
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
    ax.set_title(f"(a) {name}\nfour controls x two twins")
    ax.legend(fontsize=8)

    # (b) Separation / collapse per seed.
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

    # (c) Margin phase diagram (learned).
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
    viol = env_result["margin"]["n_violations"]
    ax.set_title(f"(c) margin theorem ({viol} violations)")
    ax.legend(fontsize=8)

    fig.suptitle(f"Stage-9 {name} -- {env_result['irreversibility_type']}",
                  fontsize=11)
    import matplotlib.pyplot as _plt
    _plt.tight_layout()
    _plt.savefig(out_path)
    _plt.close(fig)


def main() -> bool:
    import time
    t0 = time.time()
    print("=" * 78)
    print("Stage-9 -- embodied environment family (MRC mechanism generality)")
    print("=" * 78)
    print("Vehicle: genuine self-built 2D embodied gridworlds (real cells, "
          "N/S/E/W movement,\nwalls, grid observations).  MiniGrid 3.1.0 was "
          "installed and verified to import\nand run, but exact destroyed_mass "
          "requires full transition-graph enumeration, so\nthe family is built "
          "as enumerable genuine grids under one uniform reward-on-edge\n"
          "convention with three structurally distinct irreversibility types.")

    results: List[Dict[str, Any]] = []
    for spec in ENVS:
        res = evaluate_environment(spec, seeds=SEEDS)
        results.append(res)
        pdf = os.path.join(_THIS_DIR, "results", f"{spec.name}.pdf")
        write_env_figure(res, pdf)
        print(f"  figure -> {pdf}")

    # Family verdict.
    env_pass = [r["verdict"]["env_pass"] for r in results]
    margin_ok = [r["verdict"]["margin_ok"] for r in results]
    n_pass = sum(env_pass)
    all_margin_clean = all(margin_ok)
    if n_pass == len(results) and all_margin_clean:
        family_verdict = "PASS"
    elif n_pass == 0:
        family_verdict = "FAIL"
    else:
        family_verdict = "PARTIAL"

    print("\n" + "=" * 78)
    print("Family verdict table")
    print("=" * 78)
    print(f"{'environment':<26} {'irrev. structure':<34} {'verdict':<8}")
    print("-" * 78)
    for r in results:
        v = r["verdict"]
        status = "PASS" if v["env_pass"] else "FAIL"
        print(f"{r['name']:<26} {r['irreversibility_type'][:33]:<34} "
              f"{status:<8}")
        print(f"    separation {v['n_separation_pass']}/{v['n_seeds']} "
              f"(charge_load mean {v['mean_charge_load']:.2f}), "
              f"collapse {v['n_collapse_pass']}/{v['n_seeds']} "
              f"(ratio max {v['max_collapse_ratio']:.3f}), "
              f"recovery {'OK' if v['recovery_ok'] else 'X'}, "
              f"margin {'OK' if v['margin_ok'] else 'X'} "
              f"({r['margin']['n_violations']} viol)")
    print("-" * 78)
    print(f"FAMILY VERDICT: {family_verdict}  "
          f"({n_pass}/{len(results)} environments pass; "
          f"margin clean in all: {all_margin_clean})")
    if family_verdict == "PASS":
        print("The MRC mechanism (separation / recovery / collapse) and the "
              "margin theorem\nhold across three structurally distinct "
              "irreversibility types -- evidence that the\nmechanism is "
              "general, not a single-corridor artifact.")

    dt = time.time() - t0
    print(f"\nWall time: {dt:.1f} s (CPU only, no GPU)")

    payload = {
        "family_verdict": family_verdict,
        "n_pass": n_pass,
        "n_envs": len(results),
        "margin_clean_all": all_margin_clean,
        "seeds": SEEDS,
        "wall_time_s": dt,
        "vehicle": ("Genuine self-built 2D embodied gridworlds; MiniGrid "
                     "3.1.0 installed and verified, but exact destroyed_mass "
                     "needs full transition-graph enumeration, so the family "
                     "uses enumerable genuine grids with three structurally "
                     "distinct irreversibility types."),
        "cheat_check": ("Every closed-loop rollout for reward_only and "
                         "mrc-learned runs through run_closed_loop, which "
                         "asserts the planner reads no true-env dynamics "
                         "during a probe choose(S0); any violation raises "
                         "AssertionError before results are written."),
        "environments": [to_jsonable(r) for r in results],
    }
    out = os.path.join(_THIS_DIR, "results", "stage9_results.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Results -> {out}")

    return family_verdict == "PASS"


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

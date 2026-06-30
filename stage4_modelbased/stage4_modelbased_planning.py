"""
stage4_modelbased/stage4_modelbased_planning.py
============================================================

Stage-4 MRC validation -- exact-dynamics model-based decision-time planning
on a 2D-embodied LavaCorridor twin (matched reversible / irreversible).

This is Kill-Gate 1 for the new model-based / world-model embodied direction.
Before training a learned world model, verify that the MRC three properties
(separation / recovery / collapse) survive the move from Stage-1's pure
deterministic MDP setting to a model-based-planning setting:

  - Agent holds an EXACT dynamics model of a 2D gridworld twin.
  - At every decision step the agent runs an H-step rollout *over the model*
    and picks an action via either:
        reward_only :  argmax_a Q^reward_H(s, a)
        mrc         :  argmax_a [Q^reward_H(s, a) - lambda * D_w(s, a)]
    where D_w(s, a) is computed from the model's own reachability (no
    learned regressor, no policy-gradient shaping).
  - The chosen action is then committed in the real (matched) env.

This is precisely model-based decision-time planning -- the same shape as
MPC / MPPI / lookahead-with-value -- and is the cleanest embodied reduction
of Stage-1's framework that can sit immediately before a learned world model.

What is intentionally NOT here (kill-gate boundary):
  - No learned dynamics or learned D_w regressor (Stage-5+ territory).
  - No model-free RL / reward shaping (Phase-1's failure mode).
  - No real-robot, no MuJoCo / Isaac sim, no 3D navigation.

Reuse of Stage-1 primitives
---------------------------
We IMPORT destroyed_mass, value_h, policy_obl, policy_mrc, rollout_value
verbatim from stage1_unified_validation.  These are LITERALLY the same
Python objects -- no redefinition, no re-derivation of D_w, no change to
the reward-on-edge convention.  An assertion below confirms the imports
are the originals.  This guarantees any PASS/FAIL change between Stage-1
and Stage-4 comes from the model-based-planning embodiment layer, not
from a definitional drift.

Environment: 2D LavaCorridor twin
---------------------------------
Cells are labeled by integer (x, y) coordinates on a 2D grid.

  - s_0 = (0, 0)                       : start.
  - lava = (1, 0)                      : decoy/trap, east of start.
        irreversible : lava has NO actions (absorbing).
        reversible   : lava has action "recover" -> (0, 1) with reward 0
                       (the agent steps north out of lava onto the safe
                       corridor; mirrors Stage-1's "recover" -> c_1 exactly,
                       so r_d is collected at most once).
  - corridor = (0, 1), (0, 2), ..., (0, m+k)  : safe corridor going north.
        From each corridor cell only the "fwd" action is available
        (one-way; same as Stage-1 corridor).
        Target cells are (0, m), (0, m+1), ..., (0, m+k-1).
        Reward r_g sits on the OUTGOING "fwd" edge of every target
        (Stage-1 reward-on-edge convention; do not change).
  - "absorb"                           : terminal sink after (0, m+k).

Topology rationale: this is a 2D-coordinate embedding of Stage-1's corridor.
We deliberately hold topology identical so that the variable under test is
ONLY the model-based-planning embodiment layer (closed-loop step / observe /
plan-in-model / commit), not the MDP structure.  Adding orthogonal off-
corridor branches would not change what the planner sees -- it would only
test navigation, which Stage-1 already verified analytically.

Pre-registered PASS/FAIL conditions (LOCKED before any run)
-----------------------------------------------------------
  V_sep PASS iff on irreversible twin, sweeping k in {1..6} (D_w grows in k):
    - pointwise gap == max(0, D_w - r_d) to 1e-12 (with lambda = 1);
    - gap is non-decreasing in D_w (monotone in k);
    - at least one k yields strictly positive gap (separation actually fires).
  V_rec PASS iff on irreversible twin (k = 3):
    - observed switch point lambda* matches theory lambda_min = r_d / D_w
      to one grid step (0.001);
    - pi_mrc(s_0) == "a_safe" at lambda = 1;
    - separation precondition lambda_min < 1 holds.
  V_col PASS iff on matched reversible twin (k = 3):
    - D_w(s_0, a_decoy) == 0 exactly (machine equality);
    - V_mrc - V_obl == 0 to 1e-12 (gap collapses to machine precision);
    - irreversible-twin sanity: gap == D_w - r_d to 1e-12 also holds.

Any FAIL must be reported as-is.  Do NOT retune to mask a failure.

Runtime
-------
~5 seconds on a single CPU core (CPython, numpy only).  Hard ceiling 10
minutes per spec; if not finished in 20 minutes there is a bug.
"""

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List

import numpy as np

# --------------------------------------------------------------------
# Stage-1 primitive import.  These are the ONE-AND-ONLY definitions of
# destroyed_mass and the lookahead planners; we reuse them verbatim.
# --------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE1_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage1_unified"))
if _STAGE1_DIR not in sys.path:
    sys.path.insert(0, _STAGE1_DIR)

from stage1_unified_validation import (  # noqa: E402
    MDP,
    destroyed_mass,
    policy_obl,
    policy_mrc,
    rollout_value,
)

# Hard guarantee: these MUST be the Stage-1 originals (no shadowing).
assert destroyed_mass.__module__ == "stage1_unified_validation"
assert policy_obl.__module__ == "stage1_unified_validation"
assert policy_mrc.__module__ == "stage1_unified_validation"


# ====================================================================
# 2D LavaCorridor twin
# ====================================================================

S0 = (0, 0)
LAVA = (1, 0)


def build_lava_gridworld(
    k: int, m: int, r_d: float, r_g: float, gamma: float, mode: str,
) -> MDP:
    """Construct the 2D LavaCorridor twin MDP.  See module docstring."""
    assert mode in ("irreversible", "reversible")
    assert k >= 1 and m >= 1

    corridor = [(0, y) for y in range(1, m + k + 1)]
    states: List[Any] = [S0, LAVA] + corridor + ["absorb"]
    actions: Dict[Any, List[str]] = {s: [] for s in states}
    f: Dict[Any, Any] = {}
    r: Dict[Any, float] = {}

    # s_0: a_decoy = "east into lava";  a_safe = "north onto corridor".
    actions[S0] = ["a_decoy", "a_safe"]
    f[(S0, "a_decoy")] = LAVA
    r[(S0, "a_decoy")] = r_d
    f[(S0, "a_safe")] = (0, 1)
    r[(S0, "a_safe")] = 0.0

    # Corridor: one-way "fwd" (== north).  Reward-on-edge: r_g on the
    # outgoing edge of every target cell.
    target_y_set = set(range(m, m + k))
    for idx, cell in enumerate(corridor):
        nxt = corridor[idx + 1] if idx + 1 < len(corridor) else "absorb"
        actions[cell] = ["fwd"]
        f[(cell, "fwd")] = nxt
        r[(cell, "fwd")] = r_g if cell[1] in target_y_set else 0.0

    # Lava: absorbing in irreversible mode; "recover" -> (0, 1) in reversible.
    if mode == "reversible":
        actions[LAVA] = ["recover"]
        f[(LAVA, "recover")] = (0, 1)
        r[(LAVA, "recover")] = 0.0

    targets = {(0, y) for y in target_y_set}
    target_weights = {t: r_g for t in targets}

    return MDP(
        states=states, actions=actions, f=f, r=r,
        targets=targets, target_weights=target_weights, gamma=gamma,
    )


# ====================================================================
# Model-based decision-time planning over the exact dynamics model.
#
# policy_obl(mdp, s, H)        runs an H-step lookahead over mdp's
# transition function f and picks argmax Q^reward_H.  This IS
# model-based decision-time planning over an exact dynamics model
# (the same shape as MPC / MPPI / lookahead-with-value, just with
# a finite deterministic lookahead).
#
# policy_mrc(mdp, s, H, lam)   adds the per-step charge -lam * D_w(s, a)
# to the planning objective.  D_w is computed from the model's own
# reachability (destroyed_mass).
#
# Closed-loop embodied rollout: rollout_value(env, start, choose_action)
# steps the env one action at a time, asking choose_action(s) at every s.
# For Stage-4 the env IS the same exact-dynamics model (perfect oracle);
# the closed-loop structure is what Stage-5 will keep when the model
# becomes a learned world model and env stays ground-truth.
# ====================================================================

# Shared parameters for the corridor twin (mirror Stage-1 V1 defaults
# so the predictions transfer cleanly).  See Stage-1 module docstring
# for the H = m choice rationale.
DEFAULTS = dict(m=4, H=4, r_d=1.0, r_g=1.0, gamma=0.9)
LAMBDA = 1.0           # Kill-gate spec: lambda = 1 must already suffice
                       # whenever D_w > r_d.  (V_sep / V_col both use this.)


def _planner_obl(mdp: MDP, H: int) -> Callable[[Any], str]:
    return lambda s: policy_obl(mdp, s, H)


def _planner_mrc(mdp: MDP, H: int, lam: float) -> Callable[[Any], str]:
    return lambda s: policy_mrc(mdp, s, H, lam)


# ====================================================================
# V_sep -- separation:  on the irreversible twin the MRC planner avoids
# the decoy and reaches the targets; gap tracks destroyed mass.
# ====================================================================


def v_sep() -> Dict[str, Any]:
    m = DEFAULTS["m"]; H = DEFAULTS["H"]
    r_d = DEFAULTS["r_d"]; r_g = DEFAULTS["r_g"]; gamma = DEFAULTS["gamma"]
    lam = LAMBDA

    ks = list(range(1, 7))
    rows = []
    for k in ks:
        mdp = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                   mode="irreversible")
        Dw = destroyed_mass(mdp, S0, "a_decoy")
        lam_min = (r_d / Dw) if Dw > 0 else float("inf")

        v_obl = rollout_value(mdp, S0, _planner_obl(mdp, H))
        v_mrc = rollout_value(mdp, S0, _planner_mrc(mdp, H, lam))
        gap = v_mrc - v_obl
        # Closed form: at lambda = 1, MRC switches to safe iff D_w > r_d.
        #   D_w <= r_d  ->  MRC stays on decoy, gap = 0.
        #   D_w  > r_d  ->  MRC takes safe and collects sum_{g} gamma^d(s0,g) r_g
        #                   == D_w; reward_only collects r_d; gap = D_w - r_d.
        gap_pred = max(0.0, Dw - r_d)
        rows.append({
            "k": int(k), "D_w": float(Dw),
            "lam_min_theory": float(lam_min) if Dw > 0 else None,
            "V_obl": float(v_obl), "V_mrc": float(v_mrc),
            "gap_observed": float(gap), "gap_predicted": float(gap_pred),
            "pointwise_match": bool(abs(gap - gap_pred) < 1e-12),
        })

    gaps = [row["gap_observed"] for row in rows]
    monotone = all(gaps[i + 1] >= gaps[i] - 1e-12 for i in range(len(gaps) - 1))
    pointwise_ok = all(row["pointwise_match"] for row in rows)
    separation_fires = any(row["gap_observed"] > 1e-9 for row in rows)

    passed = bool(pointwise_ok and monotone and separation_fires)

    print("\n--- V_sep: separation on irreversible twin"
          f"  (lambda = {lam}, H = {H}, m = {m}) ---")
    print(f"{'k':>3} {'D_w':>10} {'lam_min':>9} {'V_obl':>10} {'V_mrc':>10} "
          f"{'gap obs':>10} {'gap pred':>10} {'match':>6}")
    for row in rows:
        lm = "n/a" if row["lam_min_theory"] is None else f"{row['lam_min_theory']:.4f}"
        print(f"{row['k']:>3} {row['D_w']:>10.6f} {lm:>9} "
              f"{row['V_obl']:>10.6f} {row['V_mrc']:>10.6f} "
              f"{row['gap_observed']:>10.6f} {row['gap_predicted']:>10.6f} "
              f"{str(row['pointwise_match']):>6}")
    print(f"Monotone in D_w (gap non-decreasing in k): {monotone}")
    print(f"Pointwise gap == max(0, D_w - r_d) to 1e-12: {pointwise_ok}")
    print(f"Separation fires for some k: {separation_fires}")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")

    return {
        "name": "V_sep: separation -- gap tracks destroyed mass",
        "lambda": lam, "H": H, "m": m,
        "rows": rows,
        "monotone_in_Dw": monotone,
        "pointwise_match": pointwise_ok,
        "separation_fires": separation_fires,
        "passed": passed,
    }


# ====================================================================
# V_rec -- recovery:  switching point of pi_mrc at s_0 as lambda varies
# matches theory lambda_min = r_d / D_w.
# ====================================================================


def v_rec() -> Dict[str, Any]:
    m = DEFAULTS["m"]; H = DEFAULTS["H"]
    r_d = DEFAULTS["r_d"]; r_g = DEFAULTS["r_g"]; gamma = DEFAULTS["gamma"]
    k = 3

    mdp = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                               mode="irreversible")
    Dw = destroyed_mass(mdp, S0, "a_decoy")
    lam_min = r_d / Dw

    lambdas = np.linspace(0.0, 1.5, 1501)   # grid step 0.001
    actions_at_s0 = [policy_mrc(mdp, S0, H, float(lam)) for lam in lambdas]
    switch_idx = next(
        (i for i, a in enumerate(actions_at_s0) if a == "a_safe"), None
    )
    lam_star = float(lambdas[switch_idx]) if switch_idx is not None else None
    grid_step = float(lambdas[1] - lambdas[0])

    at_lam_1 = policy_mrc(mdp, S0, H, 1.0)

    switch_ok = (
        lam_star is not None and abs(lam_star - lam_min) <= grid_step + 1e-9
    )
    lam1_ok = (at_lam_1 == "a_safe")
    sep_pre = (lam_min < 1.0)
    passed = bool(switch_ok and lam1_ok and sep_pre)

    print(f"\n--- V_rec: lambda phase transition on irreversible twin (k = {k}) ---")
    print(f"D_w(s_0, a_decoy)  = {Dw:.6f}")
    print(f"lambda_min theory  = r_d / D_w = {lam_min:.6f}"
          f"   (precondition < 1: {sep_pre})")
    print(f"lambda* observed   = {lam_star}"
          f"   (|lambda* - lambda_min| = {abs(lam_star - lam_min):.6f},"
          f" grid step = {grid_step:.4f})")
    print(f"pi_mrc(s_0) at lambda = 1: '{at_lam_1}'"
          f"   (expect 'a_safe': {lam1_ok})")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")

    return {
        "name": "V_rec: lambda phase transition",
        "k": k, "D_w": float(Dw),
        "lam_min_theory": float(lam_min),
        "lam_star_observed": lam_star,
        "grid_step": grid_step,
        "policy_at_lam_1": at_lam_1,
        "switch_within_grid_step": switch_ok,
        "lam1_picks_safe": lam1_ok,
        "separation_precondition": sep_pre,
        "passed": passed,
    }


# ====================================================================
# V_col -- collapse:  on the matched reversible twin D_w == 0 exactly and
# the MRC vs reward_only gap collapses to machine precision.  This is the
# causal-identification test: it confirms the gap is driven by D_w (the
# destroyed-mass charge), not by some other property of MRC.
# ====================================================================


def v_col() -> Dict[str, Any]:
    m = DEFAULTS["m"]; H = DEFAULTS["H"]
    r_d = DEFAULTS["r_d"]; r_g = DEFAULTS["r_g"]; gamma = DEFAULTS["gamma"]
    k = 3
    lam = LAMBDA

    mdp_irr = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                   mode="irreversible")
    mdp_rev = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                   mode="reversible")

    Dw_irr = destroyed_mass(mdp_irr, S0, "a_decoy")
    Dw_rev = destroyed_mass(mdp_rev, S0, "a_decoy")

    v_obl_irr = rollout_value(mdp_irr, S0, _planner_obl(mdp_irr, H))
    v_mrc_irr = rollout_value(mdp_irr, S0, _planner_mrc(mdp_irr, H, lam))
    v_obl_rev = rollout_value(mdp_rev, S0, _planner_obl(mdp_rev, H))
    v_mrc_rev = rollout_value(mdp_rev, S0, _planner_mrc(mdp_rev, H, lam))

    gap_irr = v_mrc_irr - v_obl_irr
    gap_rev = v_mrc_rev - v_obl_rev
    pred_irr = max(0.0, Dw_irr - r_d)

    irr_match = abs(gap_irr - pred_irr) < 1e-12
    rev_dw_zero = (Dw_rev == 0.0)
    rev_gap_zero = abs(gap_rev) < 1e-12
    passed = bool(irr_match and rev_dw_zero and rev_gap_zero)

    print(f"\n--- V_col: matched-twin collapse  (lambda = {lam}, k = {k}) ---")
    print(f"Irreversible: D_w = {Dw_irr:.6f}, V_obl = {v_obl_irr:.6f}, "
          f"V_mrc = {v_mrc_irr:.6f}, gap = {gap_irr:.6f} "
          f"(predicted {pred_irr:.6f}, match: {irr_match})")
    print(f"Reversible  : D_w = {Dw_rev:.6f} (expect exactly 0: {rev_dw_zero})")
    print(f"Reversible  : V_obl = {v_obl_rev:.6f}, V_mrc = {v_mrc_rev:.6f},"
          f" gap = {gap_rev:.2e}  (expect 0 to 1e-12: {rev_gap_zero})")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")

    return {
        "name": "V_col: matched-twin collapse (causal identification)",
        "k": k, "lambda": lam,
        "irreversible": {
            "D_w": float(Dw_irr),
            "V_obl": float(v_obl_irr), "V_mrc": float(v_mrc_irr),
            "gap_observed": float(gap_irr),
            "gap_predicted": float(pred_irr),
            "match": irr_match,
        },
        "reversible": {
            "D_w": float(Dw_rev),
            "V_obl": float(v_obl_rev), "V_mrc": float(v_mrc_rev),
            "gap_observed": float(gap_rev),
            "D_w_zero": rev_dw_zero,
            "gap_zero": rev_gap_zero,
        },
        "passed": passed,
    }


# ====================================================================
# Main
# ====================================================================


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_to_jsonable(x) for x in obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    return str(obj)


def main() -> bool:
    t0 = time.time()
    print("=" * 72)
    print("Stage-4 Kill-Gate 1 -- exact-dynamics model-based planning on a")
    print("matched reversible/irreversible 2D LavaCorridor embodied twin")
    print("=" * 72)
    print(f"Defaults: {DEFAULTS}")
    print(f"Lambda (V_sep / V_col): {LAMBDA}")
    print("Stage-1 reuse: destroyed_mass, policy_obl, policy_mrc, rollout_value")
    print("(imported verbatim from stage1_unified_validation; no redefinition)")

    results: Dict[str, Dict[str, Any]] = {}
    results["V_sep"] = v_sep()
    results["V_rec"] = v_rec()
    results["V_col"] = v_col()

    print()
    print("=" * 72)
    print("Stage-4 verdict table (pre-registered PASS/FAIL conditions)")
    print("=" * 72)
    print(f"{'Module':<8}  {'Status':<6}  Description")
    print("-" * 72)
    all_pass = True
    for key, res in results.items():
        status = "PASS" if res["passed"] else "FAIL"
        if not res["passed"]:
            all_pass = False
        print(f"{key:<8}  {status:<6}  {res['name']}")
    print("-" * 72)

    dt = time.time() - t0
    if all_pass:
        print("Overall: ALL PASS -- exact-dynamics model-based planning preserves")
        print("the MRC three properties on the embodied LavaCorridor twin.")
        print("Kill-Gate 1 cleared; the D_w object is native to the model-based")
        print("planning embodiment.  Proceed to Stage 5 (learned world model).")
    else:
        print("Overall: FAIL -- model-based planning does NOT preserve MRC on")
        print("the embodied twin.  Do NOT train a learned world model; first")
        print("diagnose the failed assertion(s) above.")
    print(f"Wall time: {dt:.2f} s")

    out_dir = _THIS_DIR
    out_path = os.path.join(out_dir, "stage4_results.json")
    payload = {
        "overall_pass": all_pass,
        "wall_time_seconds": dt,
        "defaults": DEFAULTS,
        "lambda": LAMBDA,
        "environment": "2D LavaCorridor twin (Stage-1 corridor embedded in 2D)",
        "stage1_imports": ["destroyed_mass", "policy_obl", "policy_mrc",
                           "rollout_value", "MDP"],
        "modules": {k: _to_jsonable(v) for k, v in results.items()},
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nResults written to {out_path}")
    return all_pass


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

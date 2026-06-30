"""
stage8_aaai/block2_margin_theorem.py
==================================================

Stage-8 Block 2 -- margin theorem validation.

Reviewer's request: lift the "decision-point error matters" finding from
descriptive plot to a precise condition supported by the margin theorem
of Q_reward and lambda * D_w_hat:

    pi_MRC(s_0) flips from a_decoy to a_safe
       <=>   lambda * (D_w_hat(s_0, a_decoy) - D_w_hat(s_0, a_safe))
              > Q_reward(s_0, a_decoy) - Q_reward(s_0, a_safe)

This block validates this margin condition empirically on two paths:
  Part A -- exact D_w (Stage 4 LavaCorridor): scan k in {1..6} and lambda
            in [0, 1.5].  By construction, mrc with exact D_w obeys the
            margin condition; we verify with a phase-diagram fit.
  Part B -- learned D_w_hat (Stage 6 perturbed WM): scan recover_corrupt_p
            and seeds; for each (rcp, seed, lambda) compute the cost gap,
            the reward margin, and the actual mrc decision, then test that
            the decision flips at exactly the line cost_gap == reward_margin.

Output: phase diagrams (PDF) + per-row CSV-like JSON.

Pre-registered PASS / FAIL
--------------------------
  PASS iff in BOTH part A and part B, every recorded (cost_gap,
  reward_margin, action) row is consistent with the margin condition --
  no decoy point with cost_gap > margin, no safe point with
  cost_gap < margin (up to floating-point tolerance of 1e-9).

  FAIL otherwise (would mean the margin theorem is not the right
  characterisation in the empirical setting).

Runtime: < 2 minutes CPU.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE1_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage1_unified"))
_STAGE4_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage4_modelbased"))
_STAGE5_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage5_learned_wm"))
_STAGE6_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage6_noisy_wm"))
for _p in (_STAGE1_DIR, _STAGE4_DIR, _STAGE5_DIR, _STAGE6_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1_unified_validation import (  # noqa: E402
    MDP, destroyed_mass, q_reward_h, policy_mrc,
)
from stage4_modelbased_planning import build_lava_gridworld, S0  # noqa: E402
from stage5_learned_wm import build_mdp_hat  # noqa: E402
from stage6_noisy_wm import train_world_model_perturbed  # noqa: E402

assert destroyed_mass.__module__ == "stage1_unified_validation"


# ====================================================================
# Constants
# ====================================================================

DEFAULTS = dict(m=4, H=4, r_d=1.0, r_g=1.0, gamma=0.9)
LAMBDAS_PART_A = np.linspace(0.0, 1.5, 31)            # coarse grid
LAMBDAS_PART_B = np.linspace(0.0, 1.5, 31)
K_LEVELS_PART_A = [1, 2, 3, 4, 5, 6]
RCP_LEVELS_PART_B = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
SEEDS_PART_B = [0, 1, 2]
TOL = 1e-9


# ====================================================================
# Part A: exact D_w on Stage 4 LavaCorridor twin
# ====================================================================

def run_part_a() -> List[Dict[str, Any]]:
    print("[Part A] Exact D_w on Stage-4 LavaCorridor")
    m = DEFAULTS["m"]; H = DEFAULTS["H"]
    r_d = DEFAULTS["r_d"]; r_g = DEFAULTS["r_g"]; gamma = DEFAULTS["gamma"]
    rows: List[Dict[str, Any]] = []
    for k in K_LEVELS_PART_A:
        mdp = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                    mode="irreversible")
        Dw_decoy = destroyed_mass(mdp, S0, "a_decoy")
        Dw_safe  = destroyed_mass(mdp, S0, "a_safe")
        Q_decoy  = q_reward_h(mdp, S0, "a_decoy", H)
        Q_safe   = q_reward_h(mdp, S0, "a_safe",  H)
        for lam in LAMBDAS_PART_A:
            a = policy_mrc(mdp, S0, H, float(lam))
            cost_gap = float(lam) * (Dw_decoy - Dw_safe)
            margin   = Q_decoy - Q_safe
            flip     = (a == "a_safe")
            rows.append({
                "k": int(k), "lam": float(lam),
                "Dw_decoy": float(Dw_decoy), "Dw_safe": float(Dw_safe),
                "Q_decoy": float(Q_decoy), "Q_safe": float(Q_safe),
                "cost_gap": float(cost_gap),
                "reward_margin": float(margin),
                "flip": bool(flip),
                "action": a,
            })
    return rows


# ====================================================================
# Part B: learned D_w_hat on Stage 6 perturbed WM
# ====================================================================

def _train_wm_cached(rcp: float, seed: int, epochs: int = 800):
    """Train a Stage-6 perturbed WM with no label/obs noise -- only the
    decision-state directional corruption (recover_corrupt_p)."""
    m = DEFAULTS["m"]; r_d = DEFAULTS["r_d"]
    r_g = DEFAULTS["r_g"]; gamma = DEFAULTS["gamma"]
    k = 3
    mdp_rev = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                    mode="reversible")
    wm, _ = train_world_model_perturbed(
        mdp_rev, epochs=epochs, label_noise_p=0.0, obs_noise_std=0.0,
        hidden=32, latent=16, seed=seed, recover_corrupt_p=rcp)
    mdp_hat, _diag = build_mdp_hat(mdp_rev, wm)
    return mdp_rev, mdp_hat


def run_part_b() -> List[Dict[str, Any]]:
    print("[Part B] Learned D_w_hat on Stage-6 perturbed WM")
    H = DEFAULTS["H"]
    rows: List[Dict[str, Any]] = []
    for rcp in RCP_LEVELS_PART_B:
        for seed in SEEDS_PART_B:
            mdp_rev, mdp_hat = _train_wm_cached(rcp, seed)
            Dw_decoy = destroyed_mass(mdp_hat, S0, "a_decoy")
            Dw_safe  = destroyed_mass(mdp_hat, S0, "a_safe")
            Q_decoy  = q_reward_h(mdp_hat, S0, "a_decoy", H)
            Q_safe   = q_reward_h(mdp_hat, S0, "a_safe",  H)
            for lam in LAMBDAS_PART_B:
                a = policy_mrc(mdp_hat, S0, H, float(lam))
                cost_gap = float(lam) * (Dw_decoy - Dw_safe)
                margin   = Q_decoy - Q_safe
                flip     = (a == "a_safe")
                rows.append({
                    "rcp": float(rcp), "seed": int(seed),
                    "lam": float(lam),
                    "Dw_hat_decoy": float(Dw_decoy),
                    "Dw_hat_safe":  float(Dw_safe),
                    "Q_decoy": float(Q_decoy), "Q_safe": float(Q_safe),
                    "cost_gap": float(cost_gap),
                    "reward_margin": float(margin),
                    "flip": bool(flip),
                    "action": a,
                })
            print(f"  rcp={rcp:.2f} seed={seed}: D_decoy={Dw_decoy:.4f} "
                  f"Q_decoy={Q_decoy:.4f} Q_safe={Q_safe:.4f}")
    return rows


# ====================================================================
# Verdict
# ====================================================================

def check_margin_condition(rows: List[Dict[str, Any]],
                            label: str) -> Dict[str, Any]:
    """For each row, the theorem predicts:
        flip == (cost_gap > reward_margin)
       with ties resolved by the tie-break (alphabetical: a_decoy first).
    """
    violations: List[Dict[str, Any]] = []
    n = 0
    for r in rows:
        cg = r["cost_gap"]; rm = r["reward_margin"]
        actually_flipped = r["flip"]
        predicted_flipped = (cg > rm + TOL)
        if predicted_flipped != actually_flipped:
            # Allow the boundary case (cost_gap == reward_margin) to go
            # either way under tie-break -- mark as borderline not violation.
            if abs(cg - rm) <= 1e-9:
                continue
            violations.append({**r, "predicted_flip": predicted_flipped})
        n += 1
    return {
        "label": label,
        "n_rows": n,
        "n_violations": len(violations),
        "violation_rate": (len(violations) / n) if n else 0.0,
        "violations_sample": violations[:5],
        "passed": len(violations) == 0,
    }


def write_figure(part_a_rows: List[Dict[str, Any]],
                  part_b_rows: List[Dict[str, Any]],
                  out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Part A.
    ax = axes[0]
    decoy_pts = [(r["reward_margin"], r["cost_gap"]) for r in part_a_rows
                  if not r["flip"]]
    safe_pts  = [(r["reward_margin"], r["cost_gap"]) for r in part_a_rows
                  if r["flip"]]
    if decoy_pts:
        xs, ys = zip(*decoy_pts)
        ax.scatter(xs, ys, c="firebrick", s=18, alpha=0.7, label="decoy")
    if safe_pts:
        xs, ys = zip(*safe_pts)
        ax.scatter(xs, ys, c="steelblue", s=18, alpha=0.7, label="safe")
    xline = np.linspace(min([0] + [r["reward_margin"] for r in part_a_rows]),
                         max([1] + [r["reward_margin"] for r in part_a_rows]),
                         50)
    ax.plot(xline, xline, "k--", lw=1, label="cost_gap = margin")
    ax.set_xlabel("reward margin (Q_decoy - Q_safe)")
    ax.set_ylabel("cost gap (lambda * (D_w_decoy - D_w_safe))")
    ax.set_title("(a) Part A -- exact D_w on Stage-4 corridor")
    ax.legend(fontsize=9)

    # Part B.
    ax = axes[1]
    decoy_pts = [(r["reward_margin"], r["cost_gap"]) for r in part_b_rows
                  if not r["flip"]]
    safe_pts  = [(r["reward_margin"], r["cost_gap"]) for r in part_b_rows
                  if r["flip"]]
    if decoy_pts:
        xs, ys = zip(*decoy_pts)
        ax.scatter(xs, ys, c="firebrick", s=14, alpha=0.5, label="decoy")
    if safe_pts:
        xs, ys = zip(*safe_pts)
        ax.scatter(xs, ys, c="steelblue", s=14, alpha=0.5, label="safe")
    all_x = [r["reward_margin"] for r in part_b_rows]
    xline = np.linspace(min([0] + all_x), max([1] + all_x), 50)
    ax.plot(xline, xline, "k--", lw=1, label="cost_gap = margin")
    ax.set_xlabel("reward margin (Q_decoy - Q_safe)")
    ax.set_ylabel("cost gap (lambda * (D_w_hat_decoy - D_w_hat_safe))")
    ax.set_title("(b) Part B -- learned D_w_hat on Stage-6 perturbed WM")
    ax.legend(fontsize=9)

    fig.suptitle("Stage-8 Block 2 -- margin theorem phase diagram")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


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
    print("=" * 78)
    print("Stage-8 Block 2 -- margin theorem validation")
    print("=" * 78)

    a_rows = run_part_a()
    b_rows = run_part_b()
    a_check = check_margin_condition(a_rows, "Part A (exact)")
    b_check = check_margin_condition(b_rows, "Part B (learned)")

    print("\n[Verification]")
    print(f"  Part A: {a_check['n_rows']} rows, "
          f"{a_check['n_violations']} violations  PASS={a_check['passed']}")
    print(f"  Part B: {b_check['n_rows']} rows, "
          f"{b_check['n_violations']} violations  PASS={b_check['passed']}")

    pdf_path = os.path.join(_THIS_DIR, "results", "block2_phase_diagram.pdf")
    write_figure(a_rows, b_rows, pdf_path)
    print(f"Figure: {pdf_path}")

    overall_pass = a_check["passed"] and b_check["passed"]
    print("\n" + "=" * 78)
    print(f"Block 2 verdict: {'PASS' if overall_pass else 'FAIL'}")
    if not overall_pass:
        if a_check["violations_sample"]:
            print("  Part A sample violations:")
            for v in a_check["violations_sample"]:
                print(f"    {v}")
        if b_check["violations_sample"]:
            print("  Part B sample violations:")
            for v in b_check["violations_sample"]:
                print(f"    {v}")
    print("=" * 78)

    dt = time.time() - t0
    payload = {
        "block": "block2_margin_theorem",
        "verdict": "PASS" if overall_pass else "FAIL",
        "wall_time_s": dt,
        "defaults": DEFAULTS,
        "k_levels_part_a": K_LEVELS_PART_A,
        "rcp_levels_part_b": RCP_LEVELS_PART_B,
        "seeds_part_b": SEEDS_PART_B,
        "lambdas_part_a": LAMBDAS_PART_A.tolist(),
        "lambdas_part_b": LAMBDAS_PART_B.tolist(),
        "part_a_check": a_check,
        "part_b_check": b_check,
        "part_a_rows": a_rows,
        "part_b_rows": b_rows,
    }
    out_path = os.path.join(_THIS_DIR, "results", "block2_results.json")
    with open(out_path, "w") as fh:
        json.dump(_to_jsonable(payload), fh, indent=2)
    print(f"Results: {out_path}")
    print(f"Wall time: {dt:.1f} s")
    return overall_pass


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

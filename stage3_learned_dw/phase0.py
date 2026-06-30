"""
phase0.py
=========

Stage-3 Phase 0 driver: FA kill-gate for learned-D_w.

Runtime: ~30 seconds on a single CPU core (well under the 10-minute budget).

What it tests
-------------
Whether the contraction-aware action score -lambda * D_w_hat remains
load-bearing when D_w is APPROXIMATED by a small MLP regressor (instead
of computed exactly), AND whether the reversible-twin collapse still
holds in the approximate setting (collapse = causal identification that
the gain is due to D_w, not some side-effect of the estimator).

This is the only non-trivial question Phase 0 answers; everything else
(estimator architecture, training schedule, instance topology) is
deliberately minimal so the result is a clean signal on the question.

Topology choice (why corridor, not free 2D grid)
------------------------------------------------
Under the spec's lambda=1 immediate-charge formulation with a finite-
horizon planner, the rollout after the s_0 decision must be FORCED for
the test to isolate the D_w-approximation question -- otherwise a 2D
agent that correctly refuses the decoy will fail downstream navigation
and the metric collapses. Stage-1 already proved this on the corridor;
Phase 0 keeps that exact MDP and ONLY swaps the exact destroyed_mass
for a learned estimator so the moving variable is the approximation
alone. Phase 1 lifts this into pixel-based deep RL on MiniGrid where the
agent learns its own value function and the navigation question gets
answered jointly. Per-instance parameters (k, m, r_d, r_g) are
RANDOMISED in Phase 0 so the regressor must generalise across instance
geometry; if it could only memorise one instance, the FA story would
be trivial and the result would not transfer to MiniGrid.

Pre-registered PASS/FAIL (LOCKED before any run)
------------------------------------------------
(a) Estimator accuracy on held-out instances:
    - median relative error on positive-D_w samples  <  0.20
    - rank AUC (positive vs zero)                    >= 0.95
(b) Charge load-bearing on irreversible twin:
    - mean_return(oracle_mrc, irreversible)  >  mean_return(reward_only, irreversible)
      (oracle gap must exist; else no decoy => trivial instance set)
    - mean_return(learned_mrc, irreversible) >  mean_return(reward_only, irreversible)
    - learned gap >= 0.50 * oracle gap
      (the learned charge closes at least half of the oracle gap)
(c) Collapse on reversible twin (causal identification):
    - residual = mean_return(learned_mrc, reversible)
                 - mean_return(reward_only, reversible)
    - irr_gain = mean_return(learned_mrc, irreversible)
                 - mean_return(reward_only, irreversible)
    - |residual| <= 0.20 * irr_gain
      (learned gain collapses to a small fraction on reversible twin)

Phase 0 PASSES iff (a) AND (b) AND (c). Any FAIL is reported as-is; do
NOT retune to mask. A FAIL means the mechanism does not survive
approximation in the cheapest setting, so Phase 1 would just amplify it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable, Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_STAGE1 = os.path.normpath(os.path.join(_HERE, "..", "stage1_unified"))
if _STAGE1 not in sys.path:
    sys.path.insert(0, _STAGE1)

from corridor_instance import (  # noqa: E402
    InstanceParams,
    build_mdp_from_params,
    collect_samples,
    destroyed_mass,
    featurise,
    horizon_for_params,
    make_random_instance,
    samples_to_tensors,
)
from regressor import (  # noqa: E402
    DwMLP,
    regression_metrics,
    train_dw_regressor,
)
from stage1_unified_validation import (  # noqa: E402
    policy_obl,
    q_reward_h,
    rollout_value,
)

# Pre-registered constants -- do NOT tune these.
GAMMA = 0.9
LAMBDA = 1.0       # paper's lambda = 1 corollary; not a tuning knob
N_TRAIN_SEEDS = 200
N_VAL_SEEDS = 40
N_EVAL_SEEDS = 100
TRAIN_SEED_BASE = 0
VAL_SEED_BASE = 5_000
EVAL_SEED_BASE = 10_000

PASS_THRESHOLDS = dict(
    estimator_max_median_rel_err=0.20,
    estimator_min_rank_auc=0.95,
    learned_gap_closure_fraction=0.50,
    collapse_residual_fraction=0.20,
)


# --------------------------------------------------------------------- #
# Estimator wrapper: maps (mdp, state, action) -> D_w-hat                #
# --------------------------------------------------------------------- #


class LearnedDwEstimator:
    """Drop-in replacement for the oracle destroyed_mass(mdp, s, a).

    Bound to a specific instance's params + mode so the planner can score
    actions at any state without re-extracting fixed features. The
    network is queried only on states/actions the MDP exposes; terminal
    states are handled by the MDP having no actions there.
    """

    def __init__(self, model: DwMLP, params: InstanceParams, mode: str):
        import torch
        self._torch = torch
        self.model = model
        self.params = params
        self.mode = mode
        self._cache: Dict = {}

    def __call__(self, mdp, state, action: str) -> float:
        key = (state, action)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        x = featurise(self.params, state, action, self.mode)
        with self._torch.no_grad():
            v = float(self.model(self._torch.from_numpy(x).unsqueeze(0)).item())
        v = max(0.0, v)
        self._cache[key] = v
        return v


# --------------------------------------------------------------------- #
# Policy wrapper that swaps the D_w source                              #
# --------------------------------------------------------------------- #


def policy_mrc_with(estimator: Callable, mdp, s, H: int, lam: float) -> str:
    """Identical to Stage-1's policy_mrc except the D_w source is the
    `estimator` callable. The reward-side Q is exact in both cases, so
    the only difference between oracle_mrc and learned_mrc here is the
    D_w plug-in -- which is exactly what we want to isolate.
    """
    acts = sorted(mdp.actions[s])
    best, best_a = -float("inf"), None
    for a in acts:
        score = q_reward_h(mdp, s, a, H) - lam * float(estimator(mdp, s, a))
        if score > best:
            best, best_a = score, a
    return best_a


# --------------------------------------------------------------------- #
# Per-instance evaluation                                                #
# --------------------------------------------------------------------- #


def evaluate_instance(params: InstanceParams, learned_model) -> Dict:
    H = horizon_for_params(params)
    out: Dict = {"twins": {}, "horizon": H, "params": dict(
        k=params.k, m=params.m, r_d=params.r_d, r_g=params.r_g, gamma=params.gamma,
    )}
    for mode in ("irreversible", "reversible"):
        mdp = build_mdp_from_params(params, mode=mode)

        r_ro = rollout_value(mdp, "s0", lambda s: policy_obl(mdp, s, H))

        r_or = rollout_value(
            mdp, "s0",
            lambda s: policy_mrc_with(destroyed_mass, mdp, s, H, LAMBDA),
        )

        learned_est = LearnedDwEstimator(learned_model, params, mode)
        r_lr = rollout_value(
            mdp, "s0",
            lambda s: policy_mrc_with(learned_est, mdp, s, H, LAMBDA),
        )

        out["twins"][mode] = {
            "return_reward_only": r_ro,
            "return_oracle_mrc": r_or,
            "return_learned_mrc": r_lr,
        }
    return out


# --------------------------------------------------------------------- #
# Main                                                                   #
# --------------------------------------------------------------------- #


def main() -> bool:
    print("=" * 72)
    print("Stage-3 Phase 0 -- FA kill-gate for learned D_w")
    print("=" * 72)
    t_total = time.time()

    # ---------- 1. Collect training data ----------
    print("\n[1/4] collecting training data from random corridor instances ...")
    t0 = time.time()
    train_seeds = list(range(TRAIN_SEED_BASE, TRAIN_SEED_BASE + N_TRAIN_SEEDS))
    val_seeds = list(range(VAL_SEED_BASE, VAL_SEED_BASE + N_VAL_SEEDS))
    train_samples = collect_samples(train_seeds, gamma=GAMMA)
    val_samples = collect_samples(val_seeds, gamma=GAMMA)
    Xt, yt = samples_to_tensors(train_samples)
    Xv, yv = samples_to_tensors(val_samples)
    print(f"  train pairs = {Xt.shape[0]}, val pairs = {Xv.shape[0]}, "
          f"in_dim = {Xt.shape[1]}, collection time = {time.time()-t0:.1f}s")

    # ---------- 2. Train the regressor ----------
    print("\n[2/4] training tiny MLP regressor (D_w-hat) ...")
    t0 = time.time()
    model, val_metrics = train_dw_regressor(
        Xt, yt, Xv, yv, hidden=64, epochs=80, lr=3e-3, seed=0, verbose=True,
    )
    train_time = time.time() - t0
    print(f"  training time = {train_time:.1f}s")

    # ---------- 3. Evaluate on held-out instances ----------
    print(f"\n[3/4] evaluating 3 agents x 2 twins on {N_EVAL_SEEDS} held-out instances ...")
    t0 = time.time()
    eval_seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + N_EVAL_SEEDS))
    per_instance: List[Dict] = []
    held_out_y_true: List[float] = []
    held_out_y_pred: List[float] = []
    for seed in eval_seeds:
        params = make_random_instance(seed=seed, gamma=GAMMA)
        res = evaluate_instance(params, model)
        res["seed"] = int(seed)
        per_instance.append(res)
        # Record D_w-hat vs oracle for the held-out estimator metric.
        for mode in ("irreversible", "reversible"):
            mdp = build_mdp_from_params(params, mode=mode)
            est = LearnedDwEstimator(model, params, mode)
            for s in mdp.states:
                for a in mdp.actions.get(s, []):
                    held_out_y_true.append(float(destroyed_mass(mdp, s, a)))
                    held_out_y_pred.append(float(est(mdp, s, a)))
    eval_time = time.time() - t0
    held_out_y_true_np = np.array(held_out_y_true)
    held_out_y_pred_np = np.array(held_out_y_pred)
    held_out_metrics = regression_metrics(held_out_y_pred_np, held_out_y_true_np)
    print(f"  eval time = {eval_time:.1f}s")
    print(f"  held-out estimator metrics = {held_out_metrics}")

    # ---------- 4. Aggregate returns ----------
    rets = {mode: {"reward_only": [], "oracle_mrc": [], "learned_mrc": []}
            for mode in ("irreversible", "reversible")}
    for rec in per_instance:
        for mode, vals in rec["twins"].items():
            rets[mode]["reward_only"].append(vals["return_reward_only"])
            rets[mode]["oracle_mrc"].append(vals["return_oracle_mrc"])
            rets[mode]["learned_mrc"].append(vals["return_learned_mrc"])

    def stats(name, mode):
        arr = np.array(rets[mode][name])
        return dict(mean=float(arr.mean()), std=float(arr.std()), n=int(arr.size))

    summary = {mode: {k: stats(k, mode)
                      for k in ("reward_only", "oracle_mrc", "learned_mrc")}
               for mode in ("irreversible", "reversible")}

    irr, rev = summary["irreversible"], summary["reversible"]
    oracle_gap_irr = irr["oracle_mrc"]["mean"] - irr["reward_only"]["mean"]
    learned_gap_irr = irr["learned_mrc"]["mean"] - irr["reward_only"]["mean"]
    oracle_gap_rev = rev["oracle_mrc"]["mean"] - rev["reward_only"]["mean"]
    learned_gap_rev = rev["learned_mrc"]["mean"] - rev["reward_only"]["mean"]

    # ---------- 5. Apply pre-registered PASS conditions ----------
    cond_a_med = held_out_metrics["median_rel_err_on_positives"] \
        < PASS_THRESHOLDS["estimator_max_median_rel_err"]
    cond_a_auc = held_out_metrics["rank_auc_positive_vs_zero"] \
        >= PASS_THRESHOLDS["estimator_min_rank_auc"]
    cond_a = bool(cond_a_med and cond_a_auc)

    if oracle_gap_irr > 1e-9:
        gap_closure = learned_gap_irr / oracle_gap_irr
    else:
        gap_closure = 0.0
    cond_b = bool(
        oracle_gap_irr > 1e-9
        and learned_gap_irr > 1e-9
        and gap_closure >= PASS_THRESHOLDS["learned_gap_closure_fraction"]
    )

    if learned_gap_irr > 1e-9:
        collapse_ratio = abs(learned_gap_rev) / learned_gap_irr
    else:
        collapse_ratio = float("inf")
    cond_c = bool(collapse_ratio <= PASS_THRESHOLDS["collapse_residual_fraction"])

    overall = cond_a and cond_b and cond_c

    # ---------- 6. Print verdict ----------
    print("\n[4/4] verdict on pre-registered PASS conditions")
    print("-" * 72)
    print("(a) Estimator quality on held-out instances:")
    print(f"    median rel err on positives = {held_out_metrics['median_rel_err_on_positives']:.4f}"
          f"  (threshold < {PASS_THRESHOLDS['estimator_max_median_rel_err']})  "
          f"=> {'PASS' if cond_a_med else 'FAIL'}")
    print(f"    rank AUC (positive vs zero) = {held_out_metrics['rank_auc_positive_vs_zero']:.4f}"
          f"  (threshold >= {PASS_THRESHOLDS['estimator_min_rank_auc']})  "
          f"=> {'PASS' if cond_a_auc else 'FAIL'}")
    print(f"    (a) overall: {'PASS' if cond_a else 'FAIL'}")

    print(f"\n(b) Charge load-bearing on irreversible twin "
          f"(mean returns over {N_EVAL_SEEDS} instances):")
    print(f"    reward_only  = {irr['reward_only']['mean']:.4f}  "
          f"(std {irr['reward_only']['std']:.4f})")
    print(f"    oracle_mrc   = {irr['oracle_mrc']['mean']:.4f}  "
          f"(std {irr['oracle_mrc']['std']:.4f})")
    print(f"    learned_mrc  = {irr['learned_mrc']['mean']:.4f}  "
          f"(std {irr['learned_mrc']['std']:.4f})")
    print(f"    oracle gap   = {oracle_gap_irr:+.4f}")
    print(f"    learned gap  = {learned_gap_irr:+.4f}  "
          f"({100*gap_closure:.1f}% of oracle gap; "
          f"threshold >= {100*PASS_THRESHOLDS['learned_gap_closure_fraction']:.0f}%)")
    print(f"    (b) overall: {'PASS' if cond_b else 'FAIL'}")

    print("\n(c) Reversible-twin collapse (causal identification):")
    print(f"    reward_only  = {rev['reward_only']['mean']:.4f}  "
          f"(std {rev['reward_only']['std']:.4f})")
    print(f"    oracle_mrc   = {rev['oracle_mrc']['mean']:.4f}  "
          f"(std {rev['oracle_mrc']['std']:.4f})")
    print(f"    learned_mrc  = {rev['learned_mrc']['mean']:.4f}  "
          f"(std {rev['learned_mrc']['std']:.4f})")
    print(f"    oracle  residual = {oracle_gap_rev:+.4f}  (expect ~0)")
    print(f"    learned residual = {learned_gap_rev:+.4f}  "
          f"(collapse ratio |residual|/irr_gain = {collapse_ratio:.4f}; "
          f"threshold <= {PASS_THRESHOLDS['collapse_residual_fraction']})")
    print(f"    (c) overall: {'PASS' if cond_c else 'FAIL'}")

    print("\n" + "=" * 72)
    print(f"Phase 0 verdict: {'PASS' if overall else 'FAIL'}")
    print("=" * 72)
    if not overall:
        print("\nDo NOT proceed to Phase 1. Diagnose the failed condition above.\n"
              "Failed conditions are reported as-is and must not be retuned away.\n"
              "Failure modes:")
        if not cond_a:
            print("  - estimator did not meet held-out accuracy threshold.")
        if not cond_b:
            print("  - learned charge did not close enough of the oracle gap on")
            print("    the irreversible twin -- approximation breaks the mechanism.")
        if not cond_c:
            print("  - learned charge still helps on the reversible twin -- the")
            print("    advantage is NOT attributable to D_w, so causal identification")
            print("    fails (the estimator is doing something other than estimating D_w).")
    print(f"\nTotal runtime: {time.time()-t_total:.1f}s")

    # ---------- 7. Persist results ----------
    out_path = os.path.join(_HERE, "phase0_results.json")
    payload = {
        "passed": bool(overall),
        "conditions": {
            "(a)_estimator_quality": bool(cond_a),
            "(b)_charge_load_bearing": bool(cond_b),
            "(c)_reversible_collapse": bool(cond_c),
        },
        "thresholds": PASS_THRESHOLDS,
        "constants": dict(
            gamma=GAMMA, lam=LAMBDA,
            n_train_seeds=N_TRAIN_SEEDS, n_val_seeds=N_VAL_SEEDS,
            n_eval_seeds=N_EVAL_SEEDS,
        ),
        "train_metrics_val": val_metrics,
        "held_out_metrics": held_out_metrics,
        "returns_summary": summary,
        "gaps": dict(
            oracle_gap_irr=oracle_gap_irr,
            learned_gap_irr=learned_gap_irr,
            oracle_gap_rev=oracle_gap_rev,
            learned_gap_rev=learned_gap_rev,
            gap_closure_ratio=gap_closure,
            collapse_ratio=collapse_ratio,
        ),
        "train_time_sec": train_time,
        "eval_time_sec": eval_time,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\nResults written to {out_path}")

    return overall


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

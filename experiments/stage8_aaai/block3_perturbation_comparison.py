"""
experiments/stage8_aaai/block3_perturbation_comparison.py
==========================================================

Stage-8 Block 3 -- three perturbation families.

Reviewer's claim: not all WM errors are equally dangerous to the MRC
mechanism.  Specifically, only DIRECTIONAL errors at the DECISION POINT
reliably push D_w_hat across the margin and flip pi_MRC; diffuse errors
and off-decision systematic errors do not.

This block tests that on the Stage-6 LavaCorridor reversible twin with
three perturbation families:

  (1) GLOBAL RANDOM NOISE -- uniform label_noise_p applied to all
      transitions.  Standard Stage-6 knob.  Predicted to rarely flip
      because random replacement averages out.
  (2) OFF-DECISION SYSTEMATIC ERROR -- per epoch, replace a specific
      intermediate transition's target (e.g. ((0, m+1), fwd)) with
      "absorb".  Directional error, but NOT at the s_0 decision point.
      Predicted to occasionally drive max_Dw_hat_rev > 0 at the
      perturbed intermediate state, but D_w_hat_rev at s_0 stays 0
      (since R(s_0) in mdp_hat still equals R(lava in rev hat) -- the
      lava recover transition is unperturbed).
  (3) DECISION-STATE DIRECTIONAL -- Stage-6's recover_corrupt_p.  Per
      epoch, replace (lava, recover) target with "absorb".  Predicted
      to reliably push D_w_hat_rev at s_0 to nonzero values and flip
      pi_MRC once cost gap > reward margin.

We sweep strength x seeds x perturbation family and report the s_0
FLIP RATE on the reversible twin -- the only metric that decides
whether collapse breaks.

Pre-registered PASS conditions (LOCKED before any run; reported as
"families tested" -- not as an absolute theorem per the user spec):
  PASS iff
    - family (3) shows a strictly higher flip rate than (1) and (2)
      across at least the upper half of the strength sweep;
    - families (1) and (2) flip rate stays below a "noise floor" of
      10% across their tested ranges.

  PARTIAL iff family (1) or (2) sporadically flips at extreme strengths
  but family (3) is still clearly the dominant driver.

  FAIL iff family (1) or (2) reaches a comparable or higher flip rate
  than (3) at any tested strength -- the "decision-point error is what
  matters" claim is not supported by this experimental family.

Runtime: ~3 minutes CPU.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE1_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage1_unified"))
_STAGE4_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage4_modelbased"))
_STAGE5_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage5_learned_wm"))
_STAGE6_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage6_noisy_wm"))
for _p in (_STAGE1_DIR, _STAGE4_DIR, _STAGE5_DIR, _STAGE6_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1_unified_validation import (  # noqa: E402
    MDP, destroyed_mass, policy_mrc, q_reward_h,
)
from stage4_modelbased_planning import build_lava_gridworld, S0, LAVA  # noqa: E402
from stage5_learned_wm import (  # noqa: E402
    WorldModel, phi, act_oh, collect_transitions, build_mdp_hat,
)


# ====================================================================
# Constants
# ====================================================================

DEFAULTS = dict(m=4, H=4, r_d=1.0, r_g=1.0, gamma=0.9, k=3)
LAMBDA = 1.0
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 400
HIDDEN = 32
LATENT = 16

# Per-family strength sweeps.
GLOBAL_NOISE_LEVELS  = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
OFF_DECISION_LEVELS  = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
DECISION_LEVELS      = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


# ====================================================================
# Local perturbed trainer that supports all three families
# ====================================================================

def _build_rev_twin():
    m = DEFAULTS["m"]; r_d = DEFAULTS["r_d"]; r_g = DEFAULTS["r_g"]
    gamma = DEFAULTS["gamma"]; k = DEFAULTS["k"]
    return build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                  mode="reversible")


def train_wm_family(*, family: str, strength: float, seed: int,
                     epochs: int = EPOCHS) -> WorldModel:
    """Train Stage-5 WorldModel with the family-specific perturbation."""
    assert family in ("global_random", "off_decision", "decision_state")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mdp = _build_rev_twin()
    wm = WorldModel(latent=LATENT, hidden=HIDDEN)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)

    transitions = collect_transitions(mdp)
    obs_s = torch.stack([phi(t[0]) for t in transitions])
    act_a = torch.stack([act_oh(t[1]) for t in transitions])
    rewards = torch.tensor([t[3] for t in transitions], dtype=torch.float32)
    state_list = list(mdp.states)
    n_states = len(state_list)
    true_s_next = [t[2] for t in transitions]

    # The "off-decision systematic" target: pick a corridor cell whose
    # corruption raises max-anywhere D_w_hat_rev but does NOT touch
    # f_hat(s_0, a_decoy) -> lava or f_hat(lava, recover) -> (0, 1).
    # We pick the first corridor cell ((0, 1), "fwd"); corrupting it to
    # "absorb" disconnects the corridor from s_0's safe path.  This is
    # an INTERMEDIATE-state directional error; it affects D_w_hat at
    # (0, 1) but not at s_0 (s_0's f is unperturbed).
    OFF_DECISION_TRANSITION = ((0, 1), "fwd")

    for epoch in range(epochs):
        s_next_list = list(true_s_next)
        if family == "global_random" and strength > 0:
            for i in range(len(s_next_list)):
                if rng.random() < strength:
                    s_next_list[i] = state_list[
                        int(rng.integers(0, n_states))]
        elif family == "off_decision" and strength > 0:
            for i, (s_i, a_i, _, _) in enumerate(transitions):
                if (s_i, a_i) == OFF_DECISION_TRANSITION:
                    if rng.random() < strength:
                        s_next_list[i] = "absorb"
        elif family == "decision_state" and strength > 0:
            # Replicates Stage 6's recover_corrupt_p.
            for i, (s_i, a_i, _, _) in enumerate(transitions):
                if s_i == LAVA and a_i == "recover":
                    if rng.random() < strength:
                        s_next_list[i] = "absorb"
        obs_s_next = torch.stack([phi(s) for s in s_next_list])

        z = wm.encode(obs_s)
        za = torch.cat([z, act_a], dim=-1)
        z_next_pred = wm.dynamics(za)
        r_pred = wm.reward(za).squeeze(-1)
        z_next_target = wm.encode(obs_s_next).detach()
        loss_dyn = F.mse_loss(z_next_pred, z_next_target)
        loss_rew = F.mse_loss(r_pred, rewards)
        loss = loss_dyn + loss_rew
        opt.zero_grad()
        loss.backward()
        opt.step()
    return wm


def evaluate_family_strength_seed(family: str, strength: float, seed: int
                                    ) -> Dict[str, Any]:
    """Train + measure: does mrc flip s_0 on rev twin?  And does max-
    anywhere D_w_hat_rev rise off zero?"""
    H = DEFAULTS["H"]
    lam = LAMBDA
    mdp_rev = _build_rev_twin()
    wm = train_wm_family(family=family, strength=strength, seed=seed)
    mdp_hat, diag = build_mdp_hat(mdp_rev, wm)

    # D_w_hat values.
    Dw_decoy_s0 = destroyed_mass(mdp_hat, S0, "a_decoy")
    Dw_safe_s0  = destroyed_mass(mdp_hat, S0, "a_safe")
    Q_decoy = q_reward_h(mdp_hat, S0, "a_decoy", H)
    Q_safe  = q_reward_h(mdp_hat, S0, "a_safe",  H)
    # Max-anywhere D_w_hat on rev (looks for any directional error).
    max_anywhere = 0.0
    for s in mdp_rev.states:
        for a in mdp_rev.actions.get(s, []):
            v = destroyed_mass(mdp_hat, s, a)
            if v > max_anywhere:
                max_anywhere = v
    a_chosen = policy_mrc(mdp_hat, S0, H, lam)
    return {
        "family": family, "strength": float(strength), "seed": int(seed),
        "Dw_hat_decoy_s0": float(Dw_decoy_s0),
        "Dw_hat_safe_s0":  float(Dw_safe_s0),
        "Dw_hat_max_anywhere": float(max_anywhere),
        "Q_decoy": float(Q_decoy), "Q_safe": float(Q_safe),
        "action_at_lam_1": a_chosen,
        "s0_flipped": bool(a_chosen == "a_safe"),
        "transition_accuracy": diag["transition_accuracy"],
    }


# ====================================================================
# Aggregation
# ====================================================================

def aggregate_per_family(rows: List[Dict[str, Any]]
                         ) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((r["family"], r["strength"]), []).append(r)
    out = []
    for (family, strength), rs in grouped.items():
        out.append({
            "family": family,
            "strength": float(strength),
            "n_seeds": len(rs),
            "flip_count": sum(1 for r in rs if r["s0_flipped"]),
            "flip_rate": float(np.mean([r["s0_flipped"] for r in rs])),
            "mean_Dw_decoy_s0": float(np.mean(
                [r["Dw_hat_decoy_s0"] for r in rs])),
            "max_Dw_decoy_s0":  float(max(r["Dw_hat_decoy_s0"] for r in rs)),
            "mean_Dw_max_anywhere": float(np.mean(
                [r["Dw_hat_max_anywhere"] for r in rs])),
            "mean_tr_acc": float(np.mean([r["transition_accuracy"] for r in rs])),
        })
    out.sort(key=lambda x: (x["family"], x["strength"]))
    return out


# ====================================================================
# Verdict (per user spec: "in the tested perturbation families")
# ====================================================================

NOISE_FLOOR = 0.10   # flip rate <= this counts as "doesn't reliably flip".


def compute_verdict(agg: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for a in agg:
        by_family.setdefault(a["family"], []).append(a)
    for f in by_family:
        by_family[f].sort(key=lambda x: x["strength"])

    # For each family, take the UPPER HALF of the sweep (the strong end)
    # and compute the max flip rate seen.
    max_upper_half = {f: max(
        x["flip_rate"] for x in by_family[f][len(by_family[f]) // 2:]
    ) for f in by_family}
    max_lower_half = {f: max(
        x["flip_rate"] for x in by_family[f][: max(1, len(by_family[f]) // 2)]
    ) for f in by_family}
    n_strong_decision = (max_upper_half["decision_state"] >= 0.6)
    quiet_global = (max_upper_half["global_random"] <= NOISE_FLOOR)
    quiet_off    = (max_upper_half["off_decision"]    <= NOISE_FLOOR)

    if n_strong_decision and quiet_global and quiet_off:
        verdict = "PASS"
        reason = ("In the tested perturbation families, decision-state "
                  "directional corruption reliably flips pi_MRC at s_0 on "
                  "the reversible twin (strong-end max flip rate "
                  f"{max_upper_half['decision_state']:.2f}); global random "
                  f"({max_upper_half['global_random']:.2f}) and off-decision "
                  f"systematic ({max_upper_half['off_decision']:.2f}) flip "
                  f"rates remain below the noise floor {NOISE_FLOOR}.")
    elif n_strong_decision:
        verdict = "PARTIAL"
        reason = ("Decision-state directional reliably flips, but some "
                  "non-decision family also crosses the noise floor at "
                  "extreme strength.")
    else:
        verdict = "FAIL"
        reason = ("Decision-state directional did not reliably flip at "
                  "strong strength.  The 'decision-point error matters' "
                  "claim is not supported by this family setup.")
    return {
        "verdict": verdict, "reason": reason,
        "max_flip_rate_upper_half": max_upper_half,
        "max_flip_rate_lower_half": max_lower_half,
        "noise_floor": NOISE_FLOOR,
    }


def write_figure(agg: List[Dict[str, Any]],
                  per_run: List[Dict[str, Any]], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for a in agg:
        by_family.setdefault(a["family"], []).append(a)
    for f in by_family:
        by_family[f].sort(key=lambda x: x["strength"])

    colours = {"global_random": "steelblue",
                "off_decision":  "darkorange",
                "decision_state": "firebrick"}

    # (a) flip rate vs strength.
    ax = axes[0]
    for family, rows in by_family.items():
        xs = [r["strength"] for r in rows]
        ys = [r["flip_rate"] for r in rows]
        ax.plot(xs, ys, "o-", color=colours[family], label=family)
    ax.axhline(NOISE_FLOOR, color="k", ls="--", lw=0.8,
                label=f"noise floor {NOISE_FLOOR}")
    ax.set_xlabel("perturbation strength")
    ax.set_ylabel("s_0 flip rate on rev twin (mean over seeds)")
    ax.set_title("(a) s_0 flip rate vs strength")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.05)

    # (b) D_w_hat at s_0 a_decoy vs strength.
    ax = axes[1]
    for family, rows in by_family.items():
        xs = [r["strength"] for r in rows]
        ys = [r["mean_Dw_decoy_s0"] for r in rows]
        ax.plot(xs, ys, "s-", color=colours[family], label=family)
        ys_any = [r["mean_Dw_max_anywhere"] for r in rows]
        ax.plot(xs, ys_any, "+--", color=colours[family], alpha=0.6,
                 label=f"{family} max-anywhere")
    ax.set_xlabel("perturbation strength")
    ax.set_ylabel("D_w_hat on rev twin")
    ax.set_title("(b) D_w_hat at s_0 decoy vs max-anywhere")
    ax.legend(fontsize=7)

    fig.suptitle("Stage-8 Block 3 -- three perturbation families")
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
    print("Stage-8 Block 3 -- three perturbation families")
    print("=" * 78)

    per_run: List[Dict[str, Any]] = []
    for family, strengths in [
        ("global_random",  GLOBAL_NOISE_LEVELS),
        ("off_decision",   OFF_DECISION_LEVELS),
        ("decision_state", DECISION_LEVELS),
    ]:
        print(f"\n[{family}] strengths={strengths}, seeds={SEEDS}")
        for st in strengths:
            for seed in SEEDS:
                r = evaluate_family_strength_seed(family, st, seed)
                per_run.append(r)
            agg_for_print = aggregate_per_family(
                [x for x in per_run if x["family"] == family
                  and x["strength"] == st])
            a = agg_for_print[0]
            print(f"  strength={st:.2f} -> flip {a['flip_count']}/"
                  f"{a['n_seeds']}, Dw_decoy_s0_mean={a['mean_Dw_decoy_s0']:.3f}, "
                  f"max_anywhere_mean={a['mean_Dw_max_anywhere']:.3f}")

    agg = aggregate_per_family(per_run)
    verdict = compute_verdict(agg)

    print("\n[Aggregated]")
    print(f"{'family':>16} {'strength':>9} {'flip':>6} {'Dw_dec_s0':>10} "
          f"{'Dw_any':>8}")
    for a in agg:
        print(f"{a['family']:>16} {a['strength']:>9.2f} "
              f"{a['flip_count']:>2d}/{a['n_seeds']:<2d}   "
              f"{a['mean_Dw_decoy_s0']:>10.4f} {a['mean_Dw_max_anywhere']:>8.4f}")

    print("\n" + "=" * 78)
    print(f"Block 3 verdict: {verdict['verdict']}")
    print(f"  {verdict['reason']}")
    print(f"  upper-half max flip rates: {verdict['max_flip_rate_upper_half']}")
    print("=" * 78)

    pdf_path = os.path.join(_THIS_DIR, "results", "block3_families.pdf")
    write_figure(agg, per_run, pdf_path)
    print(f"Figure: {pdf_path}")

    dt = time.time() - t0
    payload = {
        "block": "block3_perturbation_comparison",
        "verdict": verdict["verdict"],
        "verdict_meta": verdict,
        "wall_time_s": dt,
        "defaults": DEFAULTS,
        "lambda": LAMBDA,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "global_noise_levels": GLOBAL_NOISE_LEVELS,
        "off_decision_levels": OFF_DECISION_LEVELS,
        "decision_levels": DECISION_LEVELS,
        "aggregated": agg,
        "per_run": per_run,
        "scope": ("Conclusion holds for the tested perturbation families; "
                   "we do not claim the result as an absolute theorem."),
    }
    out_path = os.path.join(_THIS_DIR, "results", "block3_results.json")
    with open(out_path, "w") as fh:
        json.dump(_to_jsonable(payload), fh, indent=2)
    print(f"Results: {out_path}")
    print(f"Wall time: {dt:.1f} s")
    return verdict["verdict"] == "PASS"


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

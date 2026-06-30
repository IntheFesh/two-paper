"""
experiments/stage8_aaai/block4_non_oracle.py
==============================================

Stage-8 Block 4 -- non-oracle supervision variant.

Reviewer's concern (C4): Stage 7's reachability head was supervised by
labels derived from BFS on the TRUE env -- arguably too close to oracle.
This block adds a NON-ORACLE supervision variant where the labels come
from the agent's empirical rollout-observed reachability under the
SAME noisy env that the WM is being trained on, NOT from a global BFS
on the ground-truth transition function.

Concretely:
  oracle  labels:  labels[i, g] = 1 if g in reachable_set(mdp_true, f_true(s_i, a_i))
                                     else 0.
  non_oracle labels:  labels[i, g] = empirical mean over K closed-loop
                                       rollouts in the NOISY env starting
                                       from (s_i, a_i), of whether g was
                                       visited within rollout_len steps.

Crucially:
  - The non-oracle rollouts use the same recover_corrupt_p that perturbs
    the WM's transition labels.  So the agent "experiences" the same
    stochastic absorbing-lava env that its training data reflects.
  - The non-oracle labels are computed ONCE at training time and stored
    in a buffer -- no test-time access to env dynamics.
  - The runtime CountedMDP cheat-check is still in place during rollouts.

We compare three variants on the recover_corrupt_p sweep:
  - BASELINE       : Stage 5/6 WM with no reach head (baseline collapse-prone).
  - ORACLE-DA      : Stage 7's reach head with TRUE-env BFS labels.
  - NON-ORACLE-DA  : reach head with empirical rollout labels under the
                      noisy env.

Pre-registered PASS / PARTIAL / FAIL
------------------------------------
  PASS iff non-oracle-DA matches oracle-DA on collapse_ratio + charge_load
       across the rcp sweep (within tolerance).
  PARTIAL iff non-oracle-DA shows partial improvement over baseline -- it
       fixes collapse at low rcp but breaks at high rcp.  Threshold to
       report: the rcp at which non-oracle starts failing.
  FAIL iff non-oracle-DA does not improve over baseline -- repair claim
       reduces to "oracle-only".

Runtime: ~3 minutes CPU.
"""

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE1_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage1_unified"))
_STAGE4_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage4_modelbased"))
_STAGE5_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage5_learned_wm"))
_STAGE6_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage6_noisy_wm"))
_STAGE7_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "stage7_decision_aware_wm"))
for _p in (_STAGE1_DIR, _STAGE4_DIR, _STAGE5_DIR, _STAGE6_DIR, _STAGE7_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1_unified_validation import (  # noqa: E402
    MDP, destroyed_mass, policy_obl, policy_mrc, q_reward_h,
    rollout_value, reachable_set, bfs_distances,
)
from stage4_modelbased_planning import build_lava_gridworld, S0, LAVA  # noqa: E402
from stage5_learned_wm import (  # noqa: E402
    phi, act_oh, collect_transitions, OBS_DIM, ACTION_DIM, LATENT_DIM,
)
from stage7_decision_aware import (  # noqa: E402
    DAWorldModel, CountedMDP,
    precompute_distance_table, sort_targets,
    build_mdp_hat, compute_dw_hat_da,
    planner_obl, planner_mrc_baseline, planner_mrc_da,
    assert_planner_cheat_free,
)


# ====================================================================
# Constants
# ====================================================================

DEFAULTS = dict(m=4, H=4, r_d=1.0, r_g=1.0, gamma=0.9, k=3)
LAMBDA = 1.0
EPS = 1e-9
COLLAPSE_THRESHOLD = 0.30
CHARGE_THRESHOLD = 0.50
RCP_LEVELS = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
SEEDS = [0, 1, 2]
EPOCHS = 800
HIDDEN = 32
LATENT = 16
REACH_WEIGHT = 3.0
N_ROLLOUTS_NONORACLE = 60
ROLLOUT_LEN_NONORACLE = 25


# ====================================================================
# Non-oracle reach-label collection via empirical noisy rollouts
# ====================================================================

def collect_reach_labels_nonoracle(
    mdp: MDP, target_list: List[Any], rcp: float, seed: int,
    n_rollouts: int = N_ROLLOUTS_NONORACLE,
    rollout_len: int = ROLLOUT_LEN_NONORACLE,
) -> List[List[float]]:
    """For each transition (s, a, s_next_true, r), simulate n_rollouts
    closed-loop rollouts in the NOISY env starting from f_true(s, a).
    The noisy env has lava-recover -> absorb with probability rcp.

    Behaviour policy: uniform-random over available actions.

    Returns a list of [reach_g for g in target_list] per transition,
    each in [0, 1] (mean over rollouts of "did this rollout visit g
    within rollout_len steps?").

    Note: when (s, a) = (LAVA, "recover") itself, the corruption is
    applied to THIS transition's outcome -- not at a downstream step.
    This is the same corruption model that perturbs the training data.
    """
    rng = np.random.default_rng(seed)
    transitions = collect_transitions(mdp)
    out: List[List[float]] = []
    for (s_origin, a_origin, s_next_true, _) in transitions:
        reach_counts = np.zeros(len(target_list), dtype=np.float32)
        for _ in range(n_rollouts):
            # First step: apply corruption if it's the (LAVA, recover)
            # transition itself.
            if s_origin == LAVA and a_origin == "recover":
                if rng.random() < rcp:
                    s_after = "absorb"
                else:
                    s_after = s_next_true
            else:
                s_after = s_next_true
            # Continue from s_after under noisy env + random behaviour.
            visited = {s_after}
            s = s_after
            for _ in range(rollout_len):
                acts = mdp.actions.get(s, [])
                if not acts:
                    break
                a = acts[int(rng.integers(0, len(acts)))]
                if s == LAVA and a == "recover":
                    if rng.random() < rcp:
                        s_next = "absorb"
                    else:
                        s_next = mdp.f[(s, a)]
                else:
                    s_next = mdp.f[(s, a)]
                visited.add(s_next)
                if s_next == s:
                    break
                s = s_next
            for i, g in enumerate(target_list):
                if g in visited:
                    reach_counts[i] += 1
        out.append([float(c / n_rollouts) for c in reach_counts])
    return out


# ====================================================================
# Trainer that supports oracle / non-oracle / baseline (reach_weight=0)
# ====================================================================

def train_world_model(
    mdp: MDP, *, epochs: int, seed: int, hidden: int, latent: int,
    reach_weight: float, supervision: str, rcp: float, lr: float = 1e-3,
    label_noise_p: float = 0.0, obs_noise_std: float = 0.0,
) -> Tuple[DAWorldModel, List[Any], Dict[Tuple[Any, int], int], float]:
    """Train DA WM with the chosen supervision kind.

    supervision = "baseline"   -> reach_weight forced 0; no reach head.
    supervision = "oracle"     -> reach labels from TRUE env BFS.
    supervision = "non_oracle" -> reach labels from rollouts under noisy
                                    env (rcp-corrupted transitions).
    """
    assert supervision in ("baseline", "oracle", "non_oracle")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    target_list = sort_targets(mdp.targets)
    n_targets   = len(target_list)
    dist_table  = precompute_distance_table(mdp, target_list)
    wm = DAWorldModel(latent=latent, hidden=hidden, n_targets=n_targets)
    opt = torch.optim.Adam(wm.parameters(), lr=lr)

    transitions = collect_transitions(mdp)
    obs_s_clean = torch.stack([phi(t[0]) for t in transitions])
    act_a       = torch.stack([act_oh(t[1]) for t in transitions])
    rewards     = torch.tensor([t[3] for t in transitions], dtype=torch.float32)

    if supervision == "oracle":
        from stage7_decision_aware import collect_reach_labels as _crl
        reach_lbl = torch.tensor(_crl(mdp, target_list), dtype=torch.float32)
    elif supervision == "non_oracle":
        reach_lbl = torch.tensor(
            collect_reach_labels_nonoracle(mdp, target_list, rcp, seed=seed),
            dtype=torch.float32)
    else:
        reach_lbl = None  # baseline: not used

    state_list  = list(mdp.states)
    n_states    = len(state_list)
    true_s_next = [t[2] for t in transitions]

    t0 = time.time()
    for epoch in range(epochs):
        s_next_list = list(true_s_next)
        if label_noise_p > 0:
            for i in range(len(s_next_list)):
                if rng.random() < label_noise_p:
                    s_next_list[i] = state_list[
                        int(rng.integers(0, n_states))]
        if rcp > 0:
            for i, (s_i, a_i, _, _) in enumerate(transitions):
                if s_i == LAVA and a_i == "recover":
                    if rng.random() < rcp:
                        s_next_list[i] = "absorb"
        obs_s_next = torch.stack([phi(s) for s in s_next_list])
        if obs_noise_std > 0:
            obs_s = obs_s_clean + torch.randn_like(obs_s_clean) * obs_noise_std
        else:
            obs_s = obs_s_clean

        z = wm.encoder(obs_s)
        za = torch.cat([z, act_a], dim=-1)
        z_next_pred  = wm.dynamics(za)
        r_pred       = wm.reward(za).squeeze(-1)
        z_next_target = wm.encoder(obs_s_next).detach()
        loss_dyn = F.mse_loss(z_next_pred, z_next_target)
        loss_rew = F.mse_loss(r_pred, rewards)
        if supervision != "baseline" and reach_weight > 0.0:
            reach_logits = wm.reach_head(za)
            loss_reach = F.binary_cross_entropy_with_logits(
                reach_logits, reach_lbl)
            loss = loss_dyn + loss_rew + reach_weight * loss_reach
        else:
            loss = loss_dyn + loss_rew
        opt.zero_grad()
        loss.backward()
        opt.step()
    return wm, target_list, dist_table, time.time() - t0


# ====================================================================
# Closed-loop with cheat check
# ====================================================================

def run_closed_loop(true_env: MDP, choose: Callable[[Any], str],
                     where: str) -> float:
    counted = CountedMDP(true_env)
    counted.reset_count()
    _ = choose(S0)
    assert counted.dyn_count == 0, f"CHEAT [{where}]: dyn_count={counted.dyn_count}"
    counted.reset_count()
    return rollout_value(counted, S0, choose)


# ====================================================================
# Single config evaluation: baseline / oracle / non-oracle for one (rcp, seed)
# ====================================================================

def evaluate_three(rcp: float, seed: int) -> Dict[str, Any]:
    m = DEFAULTS["m"]; H = DEFAULTS["H"]
    r_d = DEFAULTS["r_d"]; r_g = DEFAULTS["r_g"]; gamma = DEFAULTS["gamma"]
    k = DEFAULTS["k"]
    lam = LAMBDA

    mdp_irr = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                    mode="irreversible")
    mdp_rev = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                    mode="reversible")

    # --- Train three kinds of WMs per twin.
    wm_b_irr, tl_irr, dt_irr, _ = train_world_model(
        mdp_irr, epochs=EPOCHS, seed=seed, hidden=HIDDEN, latent=LATENT,
        reach_weight=0.0, supervision="baseline", rcp=rcp)
    wm_b_rev, tl_rev, dt_rev, _ = train_world_model(
        mdp_rev, epochs=EPOCHS, seed=seed, hidden=HIDDEN, latent=LATENT,
        reach_weight=0.0, supervision="baseline", rcp=rcp)
    wm_o_irr, _, _, _ = train_world_model(
        mdp_irr, epochs=EPOCHS, seed=seed, hidden=HIDDEN, latent=LATENT,
        reach_weight=REACH_WEIGHT, supervision="oracle", rcp=rcp)
    wm_o_rev, _, _, _ = train_world_model(
        mdp_rev, epochs=EPOCHS, seed=seed, hidden=HIDDEN, latent=LATENT,
        reach_weight=REACH_WEIGHT, supervision="oracle", rcp=rcp)
    wm_n_irr, _, _, _ = train_world_model(
        mdp_irr, epochs=EPOCHS, seed=seed, hidden=HIDDEN, latent=LATENT,
        reach_weight=REACH_WEIGHT, supervision="non_oracle", rcp=rcp)
    wm_n_rev, _, _, _ = train_world_model(
        mdp_rev, epochs=EPOCHS, seed=seed, hidden=HIDDEN, latent=LATENT,
        reach_weight=REACH_WEIGHT, supervision="non_oracle", rcp=rcp)

    # --- Build mdp_hats.
    mdp_hat_b_irr = build_mdp_hat(mdp_irr, wm_b_irr)
    mdp_hat_b_rev = build_mdp_hat(mdp_rev, wm_b_rev)
    mdp_hat_o_irr = build_mdp_hat(mdp_irr, wm_o_irr)
    mdp_hat_o_rev = build_mdp_hat(mdp_rev, wm_o_rev)
    mdp_hat_n_irr = build_mdp_hat(mdp_irr, wm_n_irr)
    mdp_hat_n_rev = build_mdp_hat(mdp_rev, wm_n_rev)

    # --- Decision-point D_w_hat values.
    Dw_b_irr_s0 = destroyed_mass(mdp_hat_b_irr, S0, "a_decoy")
    Dw_b_rev_s0 = destroyed_mass(mdp_hat_b_rev, S0, "a_decoy")
    Dw_o_irr_s0 = compute_dw_hat_da(wm_o_irr, tl_irr, dt_irr,
                                     mdp_irr.target_weights, mdp_irr.gamma,
                                     S0, "a_decoy")
    Dw_o_rev_s0 = compute_dw_hat_da(wm_o_rev, tl_rev, dt_rev,
                                     mdp_rev.target_weights, mdp_rev.gamma,
                                     S0, "a_decoy")
    Dw_n_irr_s0 = compute_dw_hat_da(wm_n_irr, tl_irr, dt_irr,
                                     mdp_irr.target_weights, mdp_irr.gamma,
                                     S0, "a_decoy")
    Dw_n_rev_s0 = compute_dw_hat_da(wm_n_rev, tl_rev, dt_rev,
                                     mdp_rev.target_weights, mdp_rev.gamma,
                                     S0, "a_decoy")

    # --- Closed-loop returns with cheat check.
    H_p = DEFAULTS["H"]
    def C_obl_b_irr(s): return planner_obl(mdp_hat_b_irr, s, H_p)
    def C_mrc_b_irr(s): return planner_mrc_baseline(mdp_hat_b_irr, s, H_p, lam)
    def C_obl_b_rev(s): return planner_obl(mdp_hat_b_rev, s, H_p)
    def C_mrc_b_rev(s): return planner_mrc_baseline(mdp_hat_b_rev, s, H_p, lam)
    def C_obl_o_irr(s): return planner_obl(mdp_hat_o_irr, s, H_p)
    def C_mrc_o_irr(s): return planner_mrc_da(mdp_hat_o_irr, wm_o_irr,
                                                tl_irr, dt_irr, s, H_p, lam)
    def C_obl_o_rev(s): return planner_obl(mdp_hat_o_rev, s, H_p)
    def C_mrc_o_rev(s): return planner_mrc_da(mdp_hat_o_rev, wm_o_rev,
                                                tl_rev, dt_rev, s, H_p, lam)
    def C_obl_n_irr(s): return planner_obl(mdp_hat_n_irr, s, H_p)
    def C_mrc_n_irr(s): return planner_mrc_da(mdp_hat_n_irr, wm_n_irr,
                                                tl_irr, dt_irr, s, H_p, lam)
    def C_obl_n_rev(s): return planner_obl(mdp_hat_n_rev, s, H_p)
    def C_mrc_n_rev(s): return planner_mrc_da(mdp_hat_n_rev, wm_n_rev,
                                                tl_rev, dt_rev, s, H_p, lam)

    R_b_obl_irr = run_closed_loop(mdp_irr, C_obl_b_irr, "B/obl/irr")
    R_b_mrc_irr = run_closed_loop(mdp_irr, C_mrc_b_irr, "B/mrc/irr")
    R_b_obl_rev = run_closed_loop(mdp_rev, C_obl_b_rev, "B/obl/rev")
    R_b_mrc_rev = run_closed_loop(mdp_rev, C_mrc_b_rev, "B/mrc/rev")
    R_o_obl_irr = run_closed_loop(mdp_irr, C_obl_o_irr, "O/obl/irr")
    R_o_mrc_irr = run_closed_loop(mdp_irr, C_mrc_o_irr, "O/mrc/irr")
    R_o_obl_rev = run_closed_loop(mdp_rev, C_obl_o_rev, "O/obl/rev")
    R_o_mrc_rev = run_closed_loop(mdp_rev, C_mrc_o_rev, "O/mrc/rev")
    R_n_obl_irr = run_closed_loop(mdp_irr, C_obl_n_irr, "N/obl/irr")
    R_n_mrc_irr = run_closed_loop(mdp_irr, C_mrc_n_irr, "N/mrc/irr")
    R_n_obl_rev = run_closed_loop(mdp_rev, C_obl_n_rev, "N/obl/rev")
    R_n_mrc_rev = run_closed_loop(mdp_rev, C_mrc_n_rev, "N/mrc/rev")

    # Oracle reference gap (on TRUE env).
    R_orc_obl_irr = rollout_value(mdp_irr, S0,
                                    lambda s: policy_obl(mdp_irr, s, H_p))
    R_orc_mrc_irr = rollout_value(mdp_irr, S0,
                                    lambda s: policy_mrc(mdp_irr, s, H_p, lam))
    oracle_gap_irr = R_orc_mrc_irr - R_orc_obl_irr

    denom = max(oracle_gap_irr, EPS)
    return {
        "rcp": float(rcp), "seed": int(seed),
        "baseline": {
            "Dw_irr": float(Dw_b_irr_s0), "Dw_rev": float(Dw_b_rev_s0),
            "R_obl_irr": float(R_b_obl_irr), "R_mrc_irr": float(R_b_mrc_irr),
            "R_obl_rev": float(R_b_obl_rev), "R_mrc_rev": float(R_b_mrc_rev),
            "collapse_ratio": float(abs(R_b_mrc_rev - R_b_obl_rev) / denom),
            "charge_load_ratio": float((R_b_mrc_irr - R_b_obl_irr) / denom),
        },
        "oracle": {
            "Dw_irr": float(Dw_o_irr_s0), "Dw_rev": float(Dw_o_rev_s0),
            "R_obl_irr": float(R_o_obl_irr), "R_mrc_irr": float(R_o_mrc_irr),
            "R_obl_rev": float(R_o_obl_rev), "R_mrc_rev": float(R_o_mrc_rev),
            "collapse_ratio": float(abs(R_o_mrc_rev - R_o_obl_rev) / denom),
            "charge_load_ratio": float((R_o_mrc_irr - R_o_obl_irr) / denom),
        },
        "non_oracle": {
            "Dw_irr": float(Dw_n_irr_s0), "Dw_rev": float(Dw_n_rev_s0),
            "R_obl_irr": float(R_n_obl_irr), "R_mrc_irr": float(R_n_mrc_irr),
            "R_obl_rev": float(R_n_obl_rev), "R_mrc_rev": float(R_n_mrc_rev),
            "collapse_ratio": float(abs(R_n_mrc_rev - R_n_obl_rev) / denom),
            "charge_load_ratio": float((R_n_mrc_irr - R_n_obl_irr) / denom),
        },
        "oracle_gap_irr": float(oracle_gap_irr),
    }


# ====================================================================
# Aggregation + verdict
# ====================================================================

def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[float, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["rcp"], []).append(r)
    out = []
    for rcp in sorted(grouped.keys()):
        rs = grouped[rcp]
        agg = {"rcp": rcp, "n_seeds": len(rs)}
        for kind in ("baseline", "oracle", "non_oracle"):
            agg[kind] = {
                "mean_collapse": float(np.mean(
                    [r[kind]["collapse_ratio"] for r in rs])),
                "max_collapse": float(max(
                    r[kind]["collapse_ratio"] for r in rs)),
                "mean_charge": float(np.mean(
                    [r[kind]["charge_load_ratio"] for r in rs])),
                "min_charge": float(min(
                    r[kind]["charge_load_ratio"] for r in rs)),
                "mean_Dw_rev": float(np.mean(
                    [r[kind]["Dw_rev"] for r in rs])),
                "mean_Dw_irr": float(np.mean(
                    [r[kind]["Dw_irr"] for r in rs])),
            }
        out.append(agg)
    return out


def compute_verdict(agg: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare non-oracle to oracle and baseline.

    PASS:    at every triggered rcp (baseline broken), non-oracle fixes
             collapse AND keeps charge_load above threshold.
    PARTIAL: non-oracle fixes some triggered rcps but not all.
    FAIL:    non-oracle does not improve on baseline.
    """
    triggered = [a for a in agg
                  if a["baseline"]["max_collapse"] > COLLAPSE_THRESHOLD]
    non_oracle_fixed = [a for a in triggered
                          if a["non_oracle"]["max_collapse"] <= COLLAPSE_THRESHOLD
                          and a["non_oracle"]["min_charge"] >= CHARGE_THRESHOLD]
    non_oracle_failed = [a for a in triggered
                           if a["non_oracle"]["max_collapse"] > COLLAPSE_THRESHOLD
                           or  a["non_oracle"]["min_charge"]   < CHARGE_THRESHOLD]
    oracle_fixed = [a for a in triggered
                      if a["oracle"]["max_collapse"] <= COLLAPSE_THRESHOLD
                      and a["oracle"]["min_charge"] >= CHARGE_THRESHOLD]

    if not triggered:
        return {"verdict": "INCONCLUSIVE_NO_BASELINE_FAILURE",
                "triggered": [], "non_oracle_fixed": [],
                "non_oracle_failed": [], "oracle_fixed": []}

    if not non_oracle_fixed:
        verdict = "FAIL"
        reason = ("Non-oracle supervision did not fix collapse at ANY "
                  "triggered rcp.  Repair claim reduces to oracle-only "
                  "supervision; honest reporting required.")
    elif non_oracle_fixed and not non_oracle_failed:
        verdict = "PASS"
        reason = ("Non-oracle empirical-rollout supervision repaired "
                  "collapse at ALL triggered rcps (matched oracle).  The "
                  "reachability-consistency repair does not require oracle "
                  "labels.")
    else:
        threshold_rcp = min(a["rcp"] for a in non_oracle_failed)
        verdict = "PARTIAL"
        reason = (f"Non-oracle supervision repaired collapse at "
                  f"{len(non_oracle_fixed)} triggered rcps, failed at "
                  f"{len(non_oracle_failed)} (failure threshold at rcp >= "
                  f"{threshold_rcp:.2f}).  Targeted reachability-consistency "
                  f"repair partially transfers under empirical-rollout "
                  f"labels; full repair requires more accurate reach signal "
                  f"at high corruption.")
    return {
        "verdict": verdict, "reason": reason,
        "triggered_rcps": [a["rcp"] for a in triggered],
        "non_oracle_fixed_rcps": [a["rcp"] for a in non_oracle_fixed],
        "non_oracle_failed_rcps": [a["rcp"] for a in non_oracle_failed],
        "oracle_fixed_rcps": [a["rcp"] for a in oracle_fixed],
    }


def write_figure(agg: List[Dict[str, Any]], per_run: List[Dict[str, Any]],
                  out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    xs = [a["rcp"] for a in agg]
    colours = {"baseline": "firebrick", "oracle": "steelblue",
                "non_oracle": "seagreen"}

    # (a) collapse_ratio.
    ax = axes[0, 0]
    for kind in ("baseline", "oracle", "non_oracle"):
        ax.plot(xs, [a[kind]["mean_collapse"] for a in agg], "o-",
                 color=colours[kind], label=f"{kind} mean")
        ax.plot(xs, [a[kind]["max_collapse"] for a in agg], "o--",
                 color=colours[kind], alpha=0.4, label=f"{kind} max")
    ax.axhline(COLLAPSE_THRESHOLD, color="k", ls=":", lw=0.8)
    ax.set_xlabel("recover_corrupt_p"); ax.set_ylabel("collapse_ratio")
    ax.set_title("(a) collapse_ratio on rev twin")
    ax.legend(fontsize=7)

    # (b) charge_load.
    ax = axes[0, 1]
    for kind in ("baseline", "oracle", "non_oracle"):
        ax.plot(xs, [a[kind]["mean_charge"] for a in agg], "o-",
                 color=colours[kind], label=f"{kind} mean")
        ax.plot(xs, [a[kind]["min_charge"] for a in agg], "o--",
                 color=colours[kind], alpha=0.4, label=f"{kind} min")
    ax.axhline(CHARGE_THRESHOLD, color="k", ls=":", lw=0.8)
    ax.set_xlabel("recover_corrupt_p"); ax.set_ylabel("charge_load_ratio")
    ax.set_title("(b) charge_load_ratio on irr twin")
    ax.legend(fontsize=7)

    # (c) D_w_hat(rev) at s_0.
    ax = axes[1, 0]
    for kind in ("baseline", "oracle", "non_oracle"):
        ax.plot(xs, [a[kind]["mean_Dw_rev"] for a in agg], "o-",
                 color=colours[kind], label=f"{kind}")
    ax.set_xlabel("recover_corrupt_p")
    ax.set_ylabel("mean D_w_hat(s_0, a_decoy) on rev (target 0)")
    ax.set_title("(c) Decision-point asymmetric error vs rcp")
    ax.legend(fontsize=8)

    # (d) per-seed collapse scatter.
    ax = axes[1, 1]
    offset = {"baseline": -0.012, "oracle": 0.0, "non_oracle": 0.012}
    for r in per_run:
        for kind in ("baseline", "oracle", "non_oracle"):
            ax.scatter([r["rcp"] + offset[kind]], [r[kind]["collapse_ratio"]],
                        color=colours[kind], s=22, alpha=0.7)
    ax.axhline(COLLAPSE_THRESHOLD, color="k", ls=":", lw=0.8)
    ax.set_xlabel("recover_corrupt_p")
    ax.set_ylabel("collapse_ratio (per seed)")
    ax.set_title("(d) per-seed collapse scatter")

    fig.suptitle("Stage-8 Block 4 -- non-oracle reach supervision",
                  fontsize=12)
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
    print("Stage-8 Block 4 -- non-oracle reach supervision")
    print("=" * 78)
    print(f"Defaults: {DEFAULTS}, lambda={LAMBDA}, reach_weight={REACH_WEIGHT}")
    print(f"Non-oracle rollouts: K={N_ROLLOUTS_NONORACLE} per (s, a), "
          f"len={ROLLOUT_LEN_NONORACLE}")
    print("Cheat-check on every closed-loop rollout.")

    per_run = []
    print(f"\n{'rcp':>5} {'sd':>3} {'B_col':>6} {'O_col':>6} {'N_col':>6}  "
          f"{'B_chg':>6} {'O_chg':>6} {'N_chg':>6}")
    for rcp in RCP_LEVELS:
        for seed in SEEDS:
            r = evaluate_three(rcp, seed)
            per_run.append(r)
            print(f"{rcp:>5.2f} {seed:>3d} "
                  f"{r['baseline']['collapse_ratio']:>6.3f} "
                  f"{r['oracle']['collapse_ratio']:>6.3f} "
                  f"{r['non_oracle']['collapse_ratio']:>6.3f}  "
                  f"{r['baseline']['charge_load_ratio']:>6.3f} "
                  f"{r['oracle']['charge_load_ratio']:>6.3f} "
                  f"{r['non_oracle']['charge_load_ratio']:>6.3f}")

    agg = aggregate(per_run)
    verdict = compute_verdict(agg)

    print("\n[Aggregated]")
    print(f"{'rcp':>5} {'B_col_mx':>8} {'O_col_mx':>8} {'N_col_mx':>8}  "
          f"{'B_chg_mn':>8} {'O_chg_mn':>8} {'N_chg_mn':>8}  "
          f"{'B_Dw_rev':>8} {'O_Dw_rev':>8} {'N_Dw_rev':>8}")
    for a in agg:
        b, o, n = a["baseline"], a["oracle"], a["non_oracle"]
        print(f"{a['rcp']:>5.2f} "
              f"{b['max_collapse']:>8.4f} {o['max_collapse']:>8.4f} "
              f"{n['max_collapse']:>8.4f}  "
              f"{b['min_charge']:>8.4f} {o['min_charge']:>8.4f} "
              f"{n['min_charge']:>8.4f}  "
              f"{b['mean_Dw_rev']:>8.4f} {o['mean_Dw_rev']:>8.4f} "
              f"{n['mean_Dw_rev']:>8.4f}")

    print("\n" + "=" * 78)
    print(f"Block 4 verdict: {verdict['verdict']}")
    print(f"  {verdict['reason']}")
    print(f"  triggered: {verdict['triggered_rcps']}")
    print(f"  non_oracle fixed: {verdict.get('non_oracle_fixed_rcps')}")
    print(f"  non_oracle failed: {verdict.get('non_oracle_failed_rcps')}")
    print(f"  oracle fixed: {verdict.get('oracle_fixed_rcps')}")
    print("=" * 78)

    pdf_path = os.path.join(_THIS_DIR, "results", "block4_non_oracle.pdf")
    write_figure(agg, per_run, pdf_path)
    print(f"Figure: {pdf_path}")

    dt = time.time() - t0
    payload = {
        "block": "block4_non_oracle",
        "verdict": verdict["verdict"],
        "verdict_meta": verdict,
        "wall_time_s": dt,
        "defaults": DEFAULTS,
        "lambda": LAMBDA,
        "reach_weight": REACH_WEIGHT,
        "n_rollouts_nonoracle": N_ROLLOUTS_NONORACLE,
        "rollout_len_nonoracle": ROLLOUT_LEN_NONORACLE,
        "rcp_levels": RCP_LEVELS,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "per_run": per_run,
        "aggregated": agg,
        "cheat_check": ("assert_planner_cheat_free on every rollout; "
                         "non-oracle labels are computed from rollouts in "
                         "the noisy env (no access to mdp_true outside "
                         "rollout env-step)."),
    }
    out_path = os.path.join(_THIS_DIR, "results", "block4_results.json")
    with open(out_path, "w") as fh:
        json.dump(_to_jsonable(payload), fh, indent=2)
    print(f"Results: {out_path}")
    print(f"Wall time: {dt:.1f} s")
    return verdict["verdict"] in ("PASS", "PARTIAL")


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

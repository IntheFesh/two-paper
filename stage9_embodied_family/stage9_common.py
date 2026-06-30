"""
stage9_embodied_family/stage9_common.py
====================================================

Stage-9 shared framework -- genuine 2D embodied gridworlds + unified
MRC mechanism evaluation.

WHAT THIS PROVIDES
------------------
  1. Genuine 2D grid movement primitives (N/S/E/W on an H x W cell grid,
     walls, agent moves between cells).  Each environment in this family
     builds its own deterministic MDP by enumerating reachable
     (agent_cell, extra_state) tuples -- real grid coordinates, real
     movement, NOT a labelled-station list.
  2. A uniform evaluation reused across the whole environment family:
       - exact_model_sanity : verify with the EXACT destroyed_mass that
                              the twin is correctly built (oblivious -> trap,
                              mrc -> safe, D_w>0 irr / D_w=0 rev,
                              separation / recovery / margin) BEFORE any
                              learned WM is involved.
       - mechanism_eval     : train a Stage-7 decision-aware world model
                              per twin, run FOUR controls
                              (reward_only / mrc(learned D_w) /
                               oracle_mrc(exact D_w) / full_dp(optimal))
                              and measure separation / recovery / collapse.
       - margin_eval        : validate pi_MRC flips at s_0 <=> cost gap
                              crosses the reward margin (Stage-8 Block 2
                              extended to each environment).
  3. CountedMDP cheat-check on every closed-loop rollout: the test-time
     planner reads only the learned WM's D_w_hat, never the true env's
     transition function or reward.

REUSE (no rewrite of conventions)
---------------------------------
  - destroyed_mass, policy_obl, policy_mrc, q_reward_h, value_h,
    rollout_value, reachable_set, bfs_distances, MDP : Stage-1 verbatim.
  - DAWorldModel, CountedMDP, sort_targets, precompute_distance_table :
    Stage-7 verbatim (imported; asserted).
  The decision-aware D_w_hat readout and the DA training loss are the
  Stage-7 formulation, generalised to accept a per-environment grid
  observation phi and action vocabulary (the architecture and loss are
  unchanged; only the observation encoder input differs per grid).

Runtime: pure CPU.  Small grids (tens to a few hundred states), tiny
MLP world models.  Each environment evaluates in well under a minute.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------
# Reuse imports.
# --------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE1_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage1_unified"))
_STAGE5_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage5_learned_wm"))
_STAGE7_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "stage7_decision_aware_wm"))
for _p in (_STAGE1_DIR, _STAGE5_DIR, _STAGE7_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1_unified_validation import (  # noqa: E402
    MDP, destroyed_mass, policy_obl, policy_mrc,
    q_reward_h, value_h, rollout_value,
    reachable_set, bfs_distances,
)
from stage7_decision_aware import (  # noqa: E402
    DAWorldModel, CountedMDP, precompute_distance_table,
)

assert destroyed_mass.__module__ == "stage1_unified_validation"
assert policy_mrc.__module__ == "stage1_unified_validation"
assert DAWorldModel.__module__ == "stage7_decision_aware"
assert CountedMDP.__module__ == "stage7_decision_aware"


# ====================================================================
# Genuine 2D grid movement primitives
# ====================================================================

# Cardinal moves on the grid: name -> (dx, dy).  y grows downward (row).
MOVES: Dict[str, Tuple[int, int]] = {
    "N": (0, -1), "S": (0, +1), "E": (+1, 0), "W": (-1, 0),
}


def parse_layout(layout: List[str]) -> Dict[Tuple[int, int], str]:
    """Parse an ASCII grid into {(x, y): char}.  Rows are y, columns x."""
    cells: Dict[Tuple[int, int], str] = {}
    for y, row in enumerate(layout):
        for x, ch in enumerate(row):
            cells[(x, y)] = ch
    return cells


def neighbours(cell: Tuple[int, int]) -> Dict[str, Tuple[int, int]]:
    x, y = cell
    return {name: (x + dx, y + dy) for name, (dx, dy) in MOVES.items()}


# ====================================================================
# Environment specification interface
# ====================================================================

@dataclass
class EnvSpec:
    """Everything the unified evaluation needs about one environment."""
    name: str
    description: str
    build_twin: Callable[[str], MDP]      # mode -> MDP
    S0: Any
    phi: Callable[[Any], torch.Tensor]
    act_oh: Callable[[str], torch.Tensor]
    action_vocab: List[str]
    obs_dim: int
    act_dim: int
    H: int
    a_decoy: str                          # the irreversible-leading action at S0
    a_safe: str                           # the safe action at S0
    r_d: float
    r_g: float
    gamma: float
    irreversibility_type: str             # human description of the structure
    train_epochs: int = 800               # WM training budget for this env


# ====================================================================
# Generic transition / label collection (parameterised by phi/act_oh)
# ====================================================================

def collect_transitions(mdp: MDP) -> List[Tuple[Any, str, Any, float]]:
    out = []
    for s in mdp.states:
        for a in mdp.actions.get(s, []):
            out.append((s, a, mdp.f[(s, a)], mdp.r[(s, a)]))
    return out


def target_order(mdp: MDP) -> List[Any]:
    """Deterministic target ordering (by repr) shared across baseline/DA."""
    return sorted(mdp.targets, key=repr)


def collect_reach_labels(mdp: MDP, target_list: List[Any]
                          ) -> List[List[float]]:
    """For each transition (s, a, s_next, r): label_g = 1 if g reachable
    from s_next in the TRUE env, else 0.  Training-time supervision
    (semantically: did the agent eventually reach g after taking a)."""
    transitions = collect_transitions(mdp)
    out: List[List[float]] = []
    for (s, a, s_next, r) in transitions:
        R = reachable_set(mdp, s_next)
        out.append([1.0 if g in R else 0.0 for g in target_list])
    return out


# ====================================================================
# Decision-aware world model training (Stage-7 formulation, generic phi)
# ====================================================================

def train_da_world_model(
    mdp: MDP, spec: EnvSpec, *, epochs: int, seed: int,
    hidden: int = 32, latent: int = 16, reach_weight: float = 3.0,
    lr: float = 1e-3,
) -> Tuple[DAWorldModel, List[Any], Dict[Tuple[Any, int], int], float]:
    """Train a Stage-7 DAWorldModel on `mdp` using the environment's
    grid observation `spec.phi` and action encoding `spec.act_oh`.

    reach_weight = 0  -> baseline WM (dyn + reward only; reach head unused).
    reach_weight > 0  -> decision-aware (adds reachability-consistency BCE).
    """
    import time
    # Full determinism: torch CPU reductions can vary run-to-run with
    # multiple threads, and any stray numpy draw would leak across seeds.
    # Pin all of it so per-seed training is reproducible regardless of
    # process hash seed or prior seeds run in the same process.
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    target_list = target_order(mdp)
    n_targets = len(target_list)
    dist_table = precompute_distance_table(mdp, target_list)
    wm = DAWorldModel(obs_dim=spec.obs_dim, act_dim=spec.act_dim,
                       latent=latent, hidden=hidden, n_targets=n_targets)
    opt = torch.optim.Adam(wm.parameters(), lr=lr)

    transitions = collect_transitions(mdp)
    obs_s = torch.stack([spec.phi(t[0]) for t in transitions])
    act_a = torch.stack([spec.act_oh(t[1]) for t in transitions])
    rewards = torch.tensor([t[3] for t in transitions], dtype=torch.float32)
    obs_s_next = torch.stack([spec.phi(t[2]) for t in transitions])
    reach_lbl = torch.tensor(collect_reach_labels(mdp, target_list),
                              dtype=torch.float32)

    t0 = time.time()
    for _ in range(epochs):
        z = wm.encoder(obs_s)
        za = torch.cat([z, act_a], dim=-1)
        z_next_pred = wm.dynamics(za)
        r_pred = wm.reward(za).squeeze(-1)
        z_next_target = wm.encoder(obs_s_next).detach()
        loss_dyn = F.mse_loss(z_next_pred, z_next_target)
        loss_rew = F.mse_loss(r_pred, rewards)
        if reach_weight > 0.0:
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


def build_mdp_hat(mdp_struct: MDP, wm: DAWorldModel, spec: EnvSpec) -> MDP:
    """Nearest-neighbour decode of the learned latent dynamics into a
    discrete predicted transition graph (Stage-5/7 readout, generic phi).
    Reads ONLY task-spec attributes of mdp_struct (states, actions,
    targets, weights, gamma); never mdp_struct.f / .r."""
    wm.eval()
    with torch.no_grad():
        z_table = {s: wm.encoder(spec.phi(s)) for s in mdp_struct.states}
        z_stack = torch.stack(list(z_table.values()))
        state_keys = list(z_table.keys())
        f_hat: Dict[Any, Any] = {}
        r_hat: Dict[Any, float] = {}
        for s in mdp_struct.states:
            z = z_table[s]
            for a in mdp_struct.actions.get(s, []):
                za = torch.cat([z, spec.act_oh(a)], dim=-1)
                zhat = wm.dynamics(za)
                r_pred = wm.reward(za).squeeze(-1)
                d = torch.norm(z_stack - zhat.unsqueeze(0), dim=-1)
                idx = int(torch.argmin(d).item())
                f_hat[(s, a)] = state_keys[idx]
                r_hat[(s, a)] = float(r_pred.item())
    return MDP(
        states=list(mdp_struct.states),
        actions={s: list(mdp_struct.actions.get(s, []))
                  for s in mdp_struct.states},
        f=f_hat, r=r_hat,
        targets=set(mdp_struct.targets),
        target_weights=dict(mdp_struct.target_weights),
        gamma=mdp_struct.gamma,
    )


def compute_dw_hat_da(wm: DAWorldModel, spec: EnvSpec,
                       target_list: List[Any],
                       dist_table: Dict[Tuple[Any, int], int],
                       target_weights: Dict[Any, float],
                       gamma: float, s: Any, a: str) -> float:
    """Decision-aware D_w_hat (Stage-7 formula, generic phi):
         sum_g (1 - reach_pred(s, a, g)) * gamma^{dist_table[s, g]} * u(g).
    reach_pred from the learned reach head; distances from the training-
    time table; weights/gamma from task spec.  No test-time true-env access.
    """
    with torch.no_grad():
        z = wm.encoder(spec.phi(s))
        za = torch.cat([z, spec.act_oh(a)], dim=-1)
        reach_probs = torch.sigmoid(wm.reach_head(za)).numpy()
    total = 0.0
    for i, g in enumerate(target_list):
        d = dist_table.get((s, i))
        if d is None:
            continue
        u = target_weights[g]
        total += (1.0 - float(reach_probs[i])) * (gamma ** d) * u
    return total


# ====================================================================
# Planners
# ====================================================================
# reward_only / mrc-learned use mdp_hat (+ learned reach head); they never
# accept the true env, so the static interface alone forbids peeking.
# oracle_mrc / full_dp are explicit references that may use the true model.

def planner_reward_only(mdp_hat: MDP, s: Any, H: int) -> str:
    return policy_obl(mdp_hat, s, H)


def planner_mrc_learned(mdp_hat: MDP, wm: DAWorldModel, spec: EnvSpec,
                         target_list: List[Any],
                         dist_table: Dict[Tuple[Any, int], int],
                         s: Any, H: int, lam: float) -> str:
    acts = sorted(mdp_hat.actions[s])
    def score(a: str) -> float:
        return q_reward_h(mdp_hat, s, a, H) - lam * compute_dw_hat_da(
            wm, spec, target_list, dist_table,
            mdp_hat.target_weights, mdp_hat.gamma, s, a)
    return max(acts, key=score)


def planner_oracle_mrc(mdp_true: MDP, s: Any, H: int, lam: float) -> str:
    """Reference: exact D_w on the true model (upper-reference planner)."""
    return policy_mrc(mdp_true, s, H, lam)


def value_iteration(mdp: MDP, sweeps: int = 500, tol: float = 1e-12
                     ) -> Dict[Any, float]:
    """Optimal infinite-horizon value of a deterministic MDP."""
    V = {s: 0.0 for s in mdp.states}
    for _ in range(sweeps):
        delta = 0.0
        for s in mdp.states:
            acts = mdp.actions.get(s, [])
            if not acts:
                continue
            best = max(mdp.r[(s, a)] + mdp.gamma * V[mdp.f[(s, a)]]
                        for a in acts)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < tol:
            break
    return V


def full_dp_return(mdp: MDP, start: Any) -> float:
    """Optimal achievable discounted return from `start` (oracle upper
    bound reference; uses the true model, not a test-time planner)."""
    V = value_iteration(mdp)
    return V[start]


# ====================================================================
# Cheat-checked closed-loop rollout
# ====================================================================

def run_closed_loop(true_env: MDP, start: Any,
                     choose: Callable[[Any], str], where: str) -> float:
    """Wrap true_env in CountedMDP, assert the planner reads no env
    dynamics during a probe choose(start), then roll out."""
    counted = CountedMDP(true_env)
    counted.reset_count()
    _ = choose(start)
    n = counted.dyn_count
    assert n == 0, (
        f"CHEAT DETECTED [{where}]: planner read true_env.f/.r {n} times "
        f"during choose().  Test-time planner must use only the learned WM.")
    counted.reset_count()
    return rollout_value(counted, start, choose)


# ====================================================================
# Exact-model sanity (foundation; no learned WM)
# ====================================================================

EPS = 1e-9


def exact_model_sanity(spec: EnvSpec, lam: float = 1.0) -> Dict[str, Any]:
    """Verify the twin is correctly constructed using EXACT destroyed_mass.

    Checks (all on the true model, exact D_w):
      - D_w(S0, a_decoy) > 0 on irreversible twin, == 0 on reversible.
      - oblivious (reward-only, horizon H) picks a_decoy on irreversible.
      - mrc (exact D_w, lambda=1) picks a_safe on irreversible.
      - separation: oracle return(mrc) > return(reward_only) on irreversible.
      - collapse: on reversible twin, return(mrc) == return(reward_only).
      - recovery: observed lambda* switch matches r_d / D_w within grid step.
    """
    H = spec.H
    mdp_irr = spec.build_twin("irreversible")
    mdp_rev = spec.build_twin("reversible")
    S0 = spec.S0

    Dw_irr = destroyed_mass(mdp_irr, S0, spec.a_decoy)
    Dw_rev = destroyed_mass(mdp_rev, S0, spec.a_decoy)

    a_obl_irr = policy_obl(mdp_irr, S0, H)
    a_mrc_irr = policy_mrc(mdp_irr, S0, H, lam)

    R_obl_irr = rollout_value(mdp_irr, S0, lambda s: policy_obl(mdp_irr, s, H))
    R_mrc_irr = rollout_value(mdp_irr, S0,
                               lambda s: policy_mrc(mdp_irr, s, H, lam))
    R_obl_rev = rollout_value(mdp_rev, S0, lambda s: policy_obl(mdp_rev, s, H))
    R_mrc_rev = rollout_value(mdp_rev, S0,
                               lambda s: policy_mrc(mdp_rev, s, H, lam))
    R_fulldp_irr = full_dp_return(mdp_irr, S0)
    R_fulldp_rev = full_dp_return(mdp_rev, S0)

    sep_gap = R_mrc_irr - R_obl_irr
    collapse_gap = abs(R_mrc_rev - R_obl_rev)

    # Recovery threshold (margin-consistent).  The flip a_decoy -> a_safe
    # happens when lambda * (D_w(a_decoy) - D_w(a_safe)) crosses the reward
    # margin Q_reward(a_decoy) - Q_reward(a_safe), i.e. at
    #     lambda_min = (Q_decoy - Q_safe) / (D_decoy - D_safe).
    # The simpler r_d / D_w is only the special case Q_safe = D_safe = 0
    # (Stage-1's single-action corridor); on a genuine bidirectional grid
    # Q_reward(a_safe) is generally nonzero (the myopic planner believes it
    # can backtrack toward the decoy), so the margin form is the correct
    # threshold and ties recovery directly to the margin theorem.
    Q_decoy = q_reward_h(mdp_irr, S0, spec.a_decoy, H)
    Q_safe = q_reward_h(mdp_irr, S0, spec.a_safe, H)
    D_decoy = destroyed_mass(mdp_irr, S0, spec.a_decoy)
    D_safe = destroyed_mass(mdp_irr, S0, spec.a_safe)
    margin = Q_decoy - Q_safe
    cost_slope = D_decoy - D_safe
    lam_min_margin = (margin / cost_slope) if cost_slope > EPS else float("inf")
    lam_min_simple = (spec.r_d / Dw_irr) if Dw_irr > 0 else float("inf")
    lambdas = np.linspace(0.0, 2.0, 2001)
    grid_step = float(lambdas[1] - lambdas[0])
    acts = [policy_mrc(mdp_irr, S0, H, float(l)) for l in lambdas]
    sw = next((i for i, a in enumerate(acts) if a == spec.a_safe), None)
    lam_star = float(lambdas[sw]) if sw is not None else None
    a_at_1 = policy_mrc(mdp_irr, S0, H, 1.0)
    lam_min_theory = lam_min_margin

    checks = {
        "Dw_irr_positive": bool(Dw_irr > EPS),
        "Dw_rev_zero": bool(abs(Dw_rev) <= EPS),
        "oblivious_takes_decoy_irr": bool(a_obl_irr == spec.a_decoy),
        "mrc_takes_safe_irr": bool(a_mrc_irr == spec.a_safe),
        "separation_positive": bool(sep_gap > EPS),
        "collapse_zero": bool(collapse_gap <= 1e-9),
        "recovery_lambda_match": bool(
            lam_star is not None
            and abs(lam_star - lam_min_theory) <= 3 * grid_step + 1e-9),
        "recovery_policy_at_1_safe": bool(a_at_1 == spec.a_safe),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "Dw_irr": float(Dw_irr), "Dw_rev": float(Dw_rev),
        "a_obl_irr": a_obl_irr, "a_mrc_irr": a_mrc_irr,
        "R_obl_irr": float(R_obl_irr), "R_mrc_irr": float(R_mrc_irr),
        "R_obl_rev": float(R_obl_rev), "R_mrc_rev": float(R_mrc_rev),
        "R_fulldp_irr": float(R_fulldp_irr), "R_fulldp_rev": float(R_fulldp_rev),
        "separation_gap": float(sep_gap),
        "collapse_gap": float(collapse_gap),
        "lam_min_theory": float(lam_min_theory),
        "lam_min_margin": float(lam_min_margin),
        "lam_min_simple_rd_over_Dw": float(lam_min_simple),
        "lam_star_observed": lam_star,
        "grid_step": grid_step,
        "n_states_irr": len(mdp_irr.states),
        "n_states_rev": len(mdp_rev.states),
    }


# ====================================================================
# Mechanism evaluation with learned WM (four controls)
# ====================================================================

COLLAPSE_THRESHOLD = 0.30
CHARGE_THRESHOLD = 0.50
DEFAULT_EPOCHS = 800
REACH_WEIGHT = 3.0


def mechanism_eval_one_seed(spec: EnvSpec, seed: int, *,
                             epochs: int = DEFAULT_EPOCHS,
                             lam: float = 1.0) -> Dict[str, Any]:
    """One seed: train DA WM per twin, run four controls, measure
    separation / collapse / charge.  Cheat-checked rollouts for the
    learned planners."""
    H = spec.H
    mdp_irr = spec.build_twin("irreversible")
    mdp_rev = spec.build_twin("reversible")
    S0 = spec.S0

    wm_irr, tl_irr, dt_irr, t_irr = train_da_world_model(
        mdp_irr, spec, epochs=epochs, seed=seed, reach_weight=REACH_WEIGHT)
    wm_rev, tl_rev, dt_rev, t_rev = train_da_world_model(
        mdp_rev, spec, epochs=epochs, seed=seed, reach_weight=REACH_WEIGHT)

    mdp_hat_irr = build_mdp_hat(mdp_irr, wm_irr, spec)
    mdp_hat_rev = build_mdp_hat(mdp_rev, wm_rev, spec)

    # Learned-D_w at the decision point (diagnostic).
    Dw_hat_irr = compute_dw_hat_da(wm_irr, spec, tl_irr, dt_irr,
                                    mdp_irr.target_weights, mdp_irr.gamma,
                                    S0, spec.a_decoy)
    Dw_hat_rev = compute_dw_hat_da(wm_rev, spec, tl_rev, dt_rev,
                                    mdp_rev.target_weights, mdp_rev.gamma,
                                    S0, spec.a_decoy)
    Dw_true_irr = destroyed_mass(mdp_irr, S0, spec.a_decoy)
    Dw_true_rev = destroyed_mass(mdp_rev, S0, spec.a_decoy)

    # --- Closed-loop returns (learned planners are cheat-checked).
    def C_ro_irr(s): return planner_reward_only(mdp_hat_irr, s, H)
    def C_mrc_irr(s): return planner_mrc_learned(mdp_hat_irr, wm_irr, spec,
                                                  tl_irr, dt_irr, s, H, lam)
    def C_ro_rev(s): return planner_reward_only(mdp_hat_rev, s, H)
    def C_mrc_rev(s): return planner_mrc_learned(mdp_hat_rev, wm_rev, spec,
                                                  tl_rev, dt_rev, s, H, lam)

    R_ro_irr = run_closed_loop(mdp_irr, S0, C_ro_irr, f"{spec.name}/ro/irr")
    R_mrc_irr = run_closed_loop(mdp_irr, S0, C_mrc_irr, f"{spec.name}/mrc/irr")
    R_ro_rev = run_closed_loop(mdp_rev, S0, C_ro_rev, f"{spec.name}/ro/rev")
    R_mrc_rev = run_closed_loop(mdp_rev, S0, C_mrc_rev, f"{spec.name}/mrc/rev")

    # --- Reference planners (true model; not cheat-checked by design).
    R_orc_ro_irr = rollout_value(mdp_irr, S0,
                                  lambda s: policy_obl(mdp_irr, s, H))
    R_orc_mrc_irr = rollout_value(mdp_irr, S0,
                                   lambda s: policy_mrc(mdp_irr, s, H, lam))
    R_orc_ro_rev = rollout_value(mdp_rev, S0,
                                  lambda s: policy_obl(mdp_rev, s, H))
    R_orc_mrc_rev = rollout_value(mdp_rev, S0,
                                   lambda s: policy_mrc(mdp_rev, s, H, lam))
    R_fulldp_irr = full_dp_return(mdp_irr, S0)
    R_fulldp_rev = full_dp_return(mdp_rev, S0)

    oracle_gap_irr = R_orc_mrc_irr - R_orc_ro_irr
    denom = max(oracle_gap_irr, EPS)
    charge_load_ratio = (R_mrc_irr - R_ro_irr) / denom
    collapse_ratio = abs(R_mrc_rev - R_ro_rev) / denom

    return {
        "seed": seed,
        "Dw_true_irr": float(Dw_true_irr), "Dw_true_rev": float(Dw_true_rev),
        "Dw_hat_irr": float(Dw_hat_irr), "Dw_hat_rev": float(Dw_hat_rev),
        "R_reward_only_irr": float(R_ro_irr),
        "R_mrc_learned_irr": float(R_mrc_irr),
        "R_oracle_mrc_irr": float(R_orc_mrc_irr),
        "R_full_dp_irr": float(R_fulldp_irr),
        "R_reward_only_rev": float(R_ro_rev),
        "R_mrc_learned_rev": float(R_mrc_rev),
        "R_oracle_mrc_rev": float(R_orc_mrc_rev),
        "R_full_dp_rev": float(R_fulldp_rev),
        "oracle_gap_irr": float(oracle_gap_irr),
        "charge_load_ratio": float(charge_load_ratio),
        "collapse_ratio": float(collapse_ratio),
        "separation_pass": bool(charge_load_ratio >= CHARGE_THRESHOLD),
        "collapse_pass": bool(collapse_ratio <= COLLAPSE_THRESHOLD),
        "train_time_s": float(t_irr + t_rev),
    }


def recovery_eval(spec: EnvSpec, seed: int = 0,
                   epochs: int = DEFAULT_EPOCHS) -> Dict[str, Any]:
    """Lambda sweep on the LEARNED mrc planner; lambda* vs r_d/D_w_hat."""
    H = spec.H
    mdp_irr = spec.build_twin("irreversible")
    S0 = spec.S0
    wm, tl, dt, _ = train_da_world_model(
        mdp_irr, spec, epochs=epochs, seed=seed, reach_weight=REACH_WEIGHT)
    mdp_hat = build_mdp_hat(mdp_irr, wm, spec)
    Dw_decoy = compute_dw_hat_da(wm, spec, tl, dt, mdp_irr.target_weights,
                                  mdp_irr.gamma, S0, spec.a_decoy)
    Dw_safe = compute_dw_hat_da(wm, spec, tl, dt, mdp_irr.target_weights,
                                 mdp_irr.gamma, S0, spec.a_safe)
    if Dw_decoy <= EPS:
        return {"D_w_hat": float(Dw_decoy), "skipped": True,
                "policy_at_lam_1": None}
    # Margin-consistent learned recovery threshold.
    Q_decoy = q_reward_h(mdp_hat, S0, spec.a_decoy, H)
    Q_safe = q_reward_h(mdp_hat, S0, spec.a_safe, H)
    cost_slope = Dw_decoy - Dw_safe
    lam_min_hat = ((Q_decoy - Q_safe) / cost_slope
                    if cost_slope > EPS else float("inf"))
    lam_min_simple = spec.r_d / Dw_decoy
    lambdas = np.linspace(0.0, 2.0, 2001)
    grid = float(lambdas[1] - lambdas[0])
    acts = [planner_mrc_learned(mdp_hat, wm, spec, tl, dt, S0, H, float(l))
            for l in lambdas]
    sw = next((i for i, a in enumerate(acts) if a == spec.a_safe), None)
    lam_star = float(lambdas[sw]) if sw is not None else None
    a_at_1 = planner_mrc_learned(mdp_hat, wm, spec, tl, dt, S0, H, 1.0)
    return {
        "D_w_hat": float(Dw_decoy),
        "D_w_hat_safe": float(Dw_safe),
        "lam_min_hat": float(lam_min_hat),
        "lam_min_simple_rd_over_Dw": float(lam_min_simple),
        "lam_star": lam_star,
        "grid_step": grid,
        "policy_at_lam_1": a_at_1,
        "match_within_5_grid": bool(
            lam_star is not None
            and abs(lam_star - lam_min_hat) <= 5 * grid),
        "skipped": False,
    }


# ====================================================================
# Margin theorem evaluation (extends Stage-8 Block 2 to each env)
# ====================================================================

MARGIN_TOL = 1e-9


def margin_eval(spec: EnvSpec, *, seeds: List[int] = (0, 1, 2),
                 epochs: int = DEFAULT_EPOCHS,
                 lambdas: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """For learned WMs across seeds and a lambda grid, record at s_0:
        cost_gap   = lambda * (D_w_hat(a_decoy) - D_w_hat(a_safe))
        margin     = Q_reward_H(a_decoy) - Q_reward_H(a_safe)
        flipped    = (mrc picks a_safe)
    and verify flipped == (cost_gap > margin) up to tolerance.

    Uses the IRREVERSIBLE twin (where D_w_hat varies); the learned
    Q_reward and D_w_hat are read from mdp_hat / the reach head, so the
    margin condition is tested on genuinely learned quantities.
    """
    if lambdas is None:
        lambdas = np.linspace(0.0, 1.5, 31)
    H = spec.H
    S0 = spec.S0
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        mdp_irr = spec.build_twin("irreversible")
        wm, tl, dt, _ = train_da_world_model(
            mdp_irr, spec, epochs=epochs, seed=seed, reach_weight=REACH_WEIGHT)
        mdp_hat = build_mdp_hat(mdp_irr, wm, spec)
        Dw_decoy = compute_dw_hat_da(wm, spec, tl, dt, mdp_irr.target_weights,
                                      mdp_irr.gamma, S0, spec.a_decoy)
        Dw_safe = compute_dw_hat_da(wm, spec, tl, dt, mdp_irr.target_weights,
                                     mdp_irr.gamma, S0, spec.a_safe)
        Q_decoy = q_reward_h(mdp_hat, S0, spec.a_decoy, H)
        Q_safe = q_reward_h(mdp_hat, S0, spec.a_safe, H)
        margin = Q_decoy - Q_safe
        for lam in lambdas:
            a = planner_mrc_learned(mdp_hat, wm, spec, tl, dt, S0, H, float(lam))
            cost_gap = float(lam) * (Dw_decoy - Dw_safe)
            flipped = (a == spec.a_safe)
            rows.append({
                "seed": int(seed), "lam": float(lam),
                "Dw_hat_decoy": float(Dw_decoy), "Dw_hat_safe": float(Dw_safe),
                "Q_decoy": float(Q_decoy), "Q_safe": float(Q_safe),
                "cost_gap": float(cost_gap), "reward_margin": float(margin),
                "flipped": bool(flipped), "action": a,
            })
    # Verify: flipped == (cost_gap > margin), ignoring the exact boundary.
    violations = []
    n = 0
    for r in rows:
        cg, rm = r["cost_gap"], r["reward_margin"]
        predicted = (cg > rm + MARGIN_TOL)
        if abs(cg - rm) <= 1e-9:
            continue
        if predicted != r["flipped"]:
            violations.append(r)
        n += 1
    return {
        "n_rows": n, "n_violations": len(violations),
        "violations_sample": violations[:5],
        "passed": len(violations) == 0,
        "rows": rows,
    }


# ====================================================================
# Full per-environment evaluation driver
# ====================================================================

def evaluate_environment(spec: EnvSpec, *, seeds: List[int] = (0, 1, 2, 3, 4),
                          epochs: Optional[int] = None) -> Dict[str, Any]:
    """Run exact-model sanity, mechanism eval (multi-seed), recovery, and
    margin theorem for one environment.  Returns a structured dict and a
    per-environment verdict."""
    if epochs is None:
        epochs = spec.train_epochs
    print(f"\n{'='*70}\nEnvironment: {spec.name}  ({spec.irreversibility_type})"
          f"\n{'='*70}")

    sanity = exact_model_sanity(spec)
    print(f"[exact sanity] passed={sanity['passed']}  "
          f"Dw_irr={sanity['Dw_irr']:.4f} Dw_rev={sanity['Dw_rev']:.4f}  "
          f"sep_gap={sanity['separation_gap']:.4f} "
          f"collapse_gap={sanity['collapse_gap']:.2e}")
    if not sanity["passed"]:
        print(f"  FAILED checks: "
              f"{[k for k, v in sanity['checks'].items() if not v]}")

    print(f"[mechanism] training learned WMs over {len(seeds)} seeds "
          f"(epochs={epochs}) ...")
    per_seed = []
    for seed in seeds:
        r = mechanism_eval_one_seed(spec, seed, epochs=epochs)
        per_seed.append(r)
        print(f"  seed {seed}: charge_load={r['charge_load_ratio']:.3f} "
              f"collapse={r['collapse_ratio']:.3f} "
              f"Dw_hat_irr={r['Dw_hat_irr']:.3f} Dw_hat_rev={r['Dw_hat_rev']:.3f} "
              f"sep={'OK' if r['separation_pass'] else 'X'} "
              f"col={'OK' if r['collapse_pass'] else 'X'}")

    recovery = recovery_eval(spec, seed=0, epochs=epochs)
    if recovery.get("skipped"):
        print(f"[recovery] skipped (D_w_hat<=0)")
    else:
        print(f"[recovery] lam_min_hat={recovery['lam_min_hat']:.4f} "
              f"lam*={recovery['lam_star']} "
              f"pi(lam=1)={recovery['policy_at_lam_1']} "
              f"match={recovery['match_within_5_grid']}")

    margin = margin_eval(spec, seeds=list(seeds)[:3], epochs=epochs)
    print(f"[margin] {margin['n_rows']} rows, "
          f"{margin['n_violations']} violations, passed={margin['passed']}")

    # Aggregate.
    charge = [r["charge_load_ratio"] for r in per_seed]
    collapse = [r["collapse_ratio"] for r in per_seed]
    n_sep = sum(r["separation_pass"] for r in per_seed)
    n_col = sum(r["collapse_pass"] for r in per_seed)
    n = len(per_seed)

    separation_ok = (n_sep >= int(np.ceil(0.8 * n)))      # >=80% seeds
    collapse_ok = (n_col >= int(np.ceil(0.8 * n)))
    recovery_ok = (not recovery.get("skipped")
                    and recovery.get("policy_at_lam_1") == spec.a_safe
                    and recovery.get("match_within_5_grid", False))
    margin_ok = margin["passed"]

    env_pass = bool(sanity["passed"] and separation_ok and collapse_ok
                     and recovery_ok and margin_ok)

    verdict = {
        "env_pass": env_pass,
        "sanity_pass": sanity["passed"],
        "separation_ok": separation_ok,
        "collapse_ok": collapse_ok,
        "recovery_ok": recovery_ok,
        "margin_ok": margin_ok,
        "n_seeds": n,
        "n_separation_pass": n_sep,
        "n_collapse_pass": n_col,
        "mean_charge_load": float(np.mean(charge)),
        "min_charge_load": float(np.min(charge)),
        "mean_collapse_ratio": float(np.mean(collapse)),
        "max_collapse_ratio": float(np.max(collapse)),
    }
    print(f"[verdict] env_pass={env_pass}  "
          f"sep={separation_ok}({n_sep}/{n}) col={collapse_ok}({n_col}/{n}) "
          f"rec={recovery_ok} margin={margin_ok}")

    return {
        "name": spec.name,
        "description": spec.description,
        "irreversibility_type": spec.irreversibility_type,
        "config": {"H": spec.H, "r_d": spec.r_d, "r_g": spec.r_g,
                    "gamma": spec.gamma, "a_decoy": spec.a_decoy,
                    "a_safe": spec.a_safe, "obs_dim": spec.obs_dim,
                    "act_dim": spec.act_dim},
        "exact_sanity": sanity,
        "mechanism_per_seed": per_seed,
        "recovery": recovery,
        "margin": {k: v for k, v in margin.items() if k != "rows"},
        "margin_rows": margin["rows"],
        "verdict": verdict,
    }


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return sorted(to_jsonable(x) for x in obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    return str(obj)

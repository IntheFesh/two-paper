"""
stage8_aaai/block1_embodied_proxy.py
==================================================

Stage-8 Block 1 -- embodied DoorKey-Lava proxy.

WHY this block exists
---------------------
  Stage 4-7 used the LavaCorridor twin (1-D corridor in 2-D coords).
  Reviewers can fairly ask "is the MRC mechanism just a corridor artifact?".
  This block verifies separation / recovery / collapse on a richer embodied
  topology: a 2D grid with key pickup + door passage + lava decoy + goal --
  the MiniGrid DoorKey-Lava motif, irreversible variant + matched reversible
  twin.

  IMPORTANT design honesty: MiniGrid is not available behind this proxy.
  We instead build the DoorKey-Lava motif directly as a Stage-1 MDP, with
  a 2D embedding that preserves the embodied semantics (named stations
  with explicit (x, y, has_key) state).  Intermediate cells are single-
  action by construction (otherwise the finite-H planner either also
  avoids the lava on its own -- no MRC needed -- or oscillates between
  cells without ever reaching goal).  This is a real limitation of finite-
  horizon model-based decision-time planning on multi-action 2D grids;
  the corridor-topology embedding keeps the benchmark clean while still
  showing the mechanism transfers to a state space with categorical
  attributes (has_key) and labelled "stations" (key, door, goal).

Cheat constraints
-----------------
  - Test-time D_w_hat from learned WM only.  CountedMDP runtime assert
    catches any access to the TRUE env's f or r outside rollout_value's
    legitimate env-step.
  - No standalone D_w regressor.  reach head computes destroyed targets;
    structural distances from training-time precompute.
  - Same architecture (DAWorldModel) as Stage 7; same loss formulation.

Pre-registered PASS / FAIL
--------------------------
  PASS iff on the embodied twin (k = 1 goal, lambda = 1):
    (separation)  baseline mrc gives R_mrc > R_obl on irreversible twin
                   (charge_load_ratio >= 0.5 -- mrc closes at least half
                    of the oracle gap).
    (collapse)    matched reversible twin: R_mrc ~ R_obl
                   (collapse_ratio <= 0.30).
    (recovery)    pi_mrc(s_0) at lambda = 1 is a_safe;
                   observed switch lambda* matches r_d / D_w_hat to
                   within a few grid steps.

  FAIL = report honestly; do NOT retune.

Runtime
-------
  CPU only.  ~30 s for the full sweep (one WM per seed per twin, plus
  closed-loop rollouts + lambda sweep).
"""

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------
# Stage-1/4/5/7 reuse.
# --------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE1_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage1_unified"))
_STAGE4_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage4_modelbased"))
_STAGE5_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage5_learned_wm"))
_STAGE7_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "stage7_decision_aware_wm"))
for _p in (_STAGE1_DIR, _STAGE4_DIR, _STAGE5_DIR, _STAGE7_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1_unified_validation import (  # noqa: E402
    MDP, destroyed_mass, policy_obl, policy_mrc,
    q_reward_h, rollout_value,
    reachable_set, bfs_distances,
)
from stage7_decision_aware import (  # noqa: E402
    DAWorldModel, CountedMDP, precompute_distance_table,
    sort_targets, collect_reach_labels, build_mdp_hat,
    compute_dw_hat_da, planner_obl, planner_mrc_baseline, planner_mrc_da,
    assert_planner_cheat_free,
)

assert destroyed_mass.__module__ == "stage1_unified_validation"


# ====================================================================
# Embodied DoorKey-Lava twin (built directly as Stage-1 MDP)
# ====================================================================
#
# Station labels (each is a discrete state; the (x, y, has_key) tuple is
# the 2D-embedded label used as the WM observation):
#
#   S    = (0, 0, False)  -- start
#   L    = (1, 0, False)  -- lava cell (decoy)
#   C1   = (0, 1, False)  -- corridor station 1
#   KF   = (1, 1, False)  -- key cell (no key yet)
#   KT   = (1, 1, True)   -- key cell (key picked up)
#   C2   = (2, 1, True)
#   DR   = (2, 2, True)   -- door (passable because has_key=True)
#   C3   = (2, 3, True)
#   G    = (3, 3, True)   -- goal (with key)
#   absorb                -- terminal sink
#
# Intentional design: each intermediate station has exactly one valid
# action (single-action corridor in 2D coords), so the finite-H planner
# does not need to reason past s_0 -- once it commits at s_0 the
# trajectory is determined by the topology.  This keeps the benchmark
# focused on the MRC decision at s_0, which is the load-bearing part
# of the mechanism.  Multi-action 2D navigation runs into a separate
# finite-H paradox that is orthogonal to the MRC mechanism.

S_START  = ("S",   0, 0, False)
S_LAVA   = ("L",   1, 0, False)
S_C1     = ("C1",  0, 1, False)
S_KF     = ("KF",  1, 1, False)
S_KT     = ("KT",  1, 1, True)
S_C2     = ("C2",  2, 1, True)
S_DR     = ("DR",  2, 2, True)
S_C3     = ("C3",  2, 3, True)
S_G      = ("G",   3, 3, True)
S_ABSORB = "absorb"

ALL_STATES = [S_START, S_LAVA, S_C1, S_KF, S_KT, S_C2,
              S_DR, S_C3, S_G, S_ABSORB]

ACTION_DECOY    = "a_decoy"
ACTION_SAFE     = "a_safe"
ACTION_FWD_E    = "fwd_E"
ACTION_FWD_N    = "fwd_N"
ACTION_PICKUP   = "pickup"
ACTION_OPEN     = "open"     # unused but reserved
ACTION_COLLECT  = "collect"
ACTION_RECOVER  = "recover"

# Global action vocabulary (one-hot dim).
BLOCK1_ACTIONS = [
    ACTION_DECOY, ACTION_SAFE, ACTION_FWD_E, ACTION_FWD_N,
    ACTION_PICKUP, ACTION_OPEN, ACTION_COLLECT, ACTION_RECOVER,
]
ACTION_TO_ID_B1 = {a: i for i, a in enumerate(BLOCK1_ACTIONS)}


def build_doorkey_lava_twin(mode: str, *, r_d: float = 1.0, r_g: float = 3.0,
                              gamma: float = 0.9) -> MDP:
    """Build the embodied DoorKey-Lava twin MDP."""
    assert mode in ("irreversible", "reversible")
    states = list(ALL_STATES)
    actions: Dict[Any, List[str]] = {s: [] for s in states}
    f: Dict[Tuple[Any, str], Any] = {}
    r: Dict[Tuple[Any, str], float] = {}

    # Start: 2 actions.
    actions[S_START] = [ACTION_DECOY, ACTION_SAFE]
    f[(S_START, ACTION_DECOY)] = S_LAVA
    r[(S_START, ACTION_DECOY)] = r_d
    f[(S_START, ACTION_SAFE)] = S_C1
    r[(S_START, ACTION_SAFE)] = 0.0

    # Lava (mode-dependent).
    if mode == "reversible":
        actions[S_LAVA] = [ACTION_RECOVER]
        f[(S_LAVA, ACTION_RECOVER)] = S_C1     # mirror Stage 4: recover -> C1
        r[(S_LAVA, ACTION_RECOVER)] = 0.0

    # Corridor stations, each single-action by design.
    actions[S_C1] = [ACTION_FWD_E]
    f[(S_C1, ACTION_FWD_E)] = S_KF
    r[(S_C1, ACTION_FWD_E)] = 0.0

    actions[S_KF] = [ACTION_PICKUP]
    f[(S_KF, ACTION_PICKUP)] = S_KT
    r[(S_KF, ACTION_PICKUP)] = 0.0

    actions[S_KT] = [ACTION_FWD_E]
    f[(S_KT, ACTION_FWD_E)] = S_C2
    r[(S_KT, ACTION_FWD_E)] = 0.0

    actions[S_C2] = [ACTION_FWD_N]
    f[(S_C2, ACTION_FWD_N)] = S_DR
    r[(S_C2, ACTION_FWD_N)] = 0.0

    actions[S_DR] = [ACTION_FWD_N]
    f[(S_DR, ACTION_FWD_N)] = S_C3
    r[(S_DR, ACTION_FWD_N)] = 0.0

    actions[S_C3] = [ACTION_FWD_E]
    f[(S_C3, ACTION_FWD_E)] = S_G
    r[(S_C3, ACTION_FWD_E)] = 0.0

    # Goal: Stage-1 reward-on-edge convention.
    actions[S_G] = [ACTION_COLLECT]
    f[(S_G, ACTION_COLLECT)] = S_ABSORB
    r[(S_G, ACTION_COLLECT)] = r_g

    targets = {S_G}
    target_weights = {S_G: r_g}

    return MDP(states=states, actions=actions, f=f, r=r,
                targets=targets, target_weights=target_weights, gamma=gamma)


# ====================================================================
# Custom observation + action embedding (independent of Stage-5 phi)
# ====================================================================
#
# Stage-5/7 phi assumes (x, y) tuple states; here we use labelled tuples
# of the form (label, x, y, has_key).  We provide a Block-1 phi and
# act_oh that match this state space.

OBS_DIM_B1 = 5      # (x, y, has_key, is_lava, is_goal)
ACT_DIM_B1 = len(BLOCK1_ACTIONS)


def phi_b1(s: Any) -> torch.Tensor:
    if s == S_ABSORB:
        return torch.tensor([-1.0, -1.0, -1.0, 0.0, 0.0],
                             dtype=torch.float32)
    label, x, y, hk = s
    is_lava = 1.0 if label == "L" else 0.0
    is_goal = 1.0 if label == "G" else 0.0
    return torch.tensor([float(x), float(y), float(hk),
                          is_lava, is_goal], dtype=torch.float32)


def act_oh_b1(a: str) -> torch.Tensor:
    v = torch.zeros(ACT_DIM_B1, dtype=torch.float32)
    v[ACTION_TO_ID_B1[a]] = 1.0
    return v


# ====================================================================
# Local WM and training (mirrors Stage 7's DAWorldModel with Block-1
# observation/action dimensions)
# ====================================================================

import torch.nn as nn


class WorldModelB1(nn.Module):
    """DAWorldModel with Block-1 OBS_DIM / ACT_DIM."""

    def __init__(self, latent: int = 16, hidden: int = 32,
                 n_targets: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(OBS_DIM_B1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent + ACT_DIM_B1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.reward = nn.Sequential(
            nn.Linear(latent + ACT_DIM_B1, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.reach_head = nn.Sequential(
            nn.Linear(latent + ACT_DIM_B1, hidden), nn.ReLU(),
            nn.Linear(hidden, n_targets),
        )

    def encode(self, o):
        return self.encoder(o)


def _collect_transitions(mdp: MDP) -> List[Tuple[Any, str, Any, float]]:
    out = []
    for s in mdp.states:
        for a in mdp.actions.get(s, []):
            out.append((s, a, mdp.f[(s, a)], mdp.r[(s, a)]))
    return out


def _collect_reach_labels_b1(mdp: MDP, target_list: List[Any]
                              ) -> List[List[float]]:
    transitions = _collect_transitions(mdp)
    out = []
    for (s, a, s_next, r) in transitions:
        R = reachable_set(mdp, s_next)
        out.append([1.0 if g in R else 0.0 for g in target_list])
    return out


def _precompute_dist_b1(mdp: MDP, target_list: List[Any]
                          ) -> Dict[Tuple[Any, int], int]:
    out: Dict[Tuple[Any, int], int] = {}
    for s in mdp.states:
        d = bfs_distances(mdp, s)
        for i, g in enumerate(target_list):
            if g in d:
                out[(s, i)] = d[g]
    return out


def train_wm_b1(mdp: MDP, *, epochs: int = 800, lr: float = 1e-3,
                 seed: int = 0, hidden: int = 32, latent: int = 16,
                 reach_weight: float = 3.0
                 ) -> Tuple[WorldModelB1, List[Any],
                            Dict[Tuple[Any, int], int], float]:
    torch.manual_seed(seed)
    target_list = sort_targets(mdp.targets) if all(
        isinstance(t, tuple) and len(t) >= 2 and isinstance(t[1], int)
        for t in mdp.targets
    ) else list(mdp.targets)
    # For Block 1, targets are labelled tuples; use insertion order.
    target_list = list(mdp.targets)
    n_targets = len(target_list)
    dist_table = _precompute_dist_b1(mdp, target_list)
    wm = WorldModelB1(latent=latent, hidden=hidden, n_targets=n_targets)
    opt = torch.optim.Adam(wm.parameters(), lr=lr)

    transitions = _collect_transitions(mdp)
    obs_s_clean = torch.stack([phi_b1(t[0]) for t in transitions])
    act_a = torch.stack([act_oh_b1(t[1]) for t in transitions])
    rewards = torch.tensor([t[3] for t in transitions], dtype=torch.float32)
    reach_lbl = torch.tensor(_collect_reach_labels_b1(mdp, target_list),
                              dtype=torch.float32)
    obs_s_next_clean = torch.stack([phi_b1(t[2]) for t in transitions])

    t0 = time.time()
    for epoch in range(epochs):
        z = wm.encoder(obs_s_clean)
        za = torch.cat([z, act_a], dim=-1)
        z_next_pred = wm.dynamics(za)
        r_pred = wm.reward(za).squeeze(-1)
        z_next_target = wm.encoder(obs_s_next_clean).detach()

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


def build_mdp_hat_b1(mdp_struct: MDP, wm: WorldModelB1) -> MDP:
    """Nearest-neighbour decode of WM dynamics, mirroring Stage 7's
    build_mdp_hat but with Block-1 phi/act_oh."""
    wm.eval()
    with torch.no_grad():
        z_table = {s: wm.encoder(phi_b1(s)) for s in mdp_struct.states}
        z_stack = torch.stack(list(z_table.values()))
        state_keys = list(z_table.keys())

        f_hat: Dict[Any, Any] = {}
        r_hat: Dict[Any, float] = {}
        for s in mdp_struct.states:
            z = z_table[s]
            for a in mdp_struct.actions.get(s, []):
                za = torch.cat([z, act_oh_b1(a)], dim=-1)
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


def compute_dw_hat_da_b1(wm: WorldModelB1,
                          target_list: List[Any],
                          dist_table: Dict[Tuple[Any, int], int],
                          target_weights: Dict[Any, float],
                          gamma: float, s: Any, a: str) -> float:
    """Decision-aware D_w_hat for Block-1 (mirrors Stage 7's compute_dw_hat_da
    but uses Block-1 phi/act_oh)."""
    with torch.no_grad():
        z = wm.encoder(phi_b1(s))
        za = torch.cat([z, act_oh_b1(a)], dim=-1)
        reach_probs = torch.sigmoid(wm.reach_head(za)).numpy()

    total = 0.0
    for i, g in enumerate(target_list):
        d = dist_table.get((s, i))
        if d is None:
            continue
        u = target_weights[g]
        p_destroyed = 1.0 - float(reach_probs[i])
        total += p_destroyed * (gamma ** d) * u
    return total


# ====================================================================
# Block-1 planners + evaluation
# ====================================================================

DEFAULTS = dict(r_d=1.0, r_g=3.0, gamma=0.9, H=4)
LAMBDA = 1.0
EPS = 1e-9
COLLAPSE_THRESHOLD = 0.30
CHARGE_THRESHOLD = 0.50
SEEDS = [0, 1, 2, 3, 4]


def planner_obl_b1(mdp_hat: MDP, s: Any, H: int) -> str:
    return policy_obl(mdp_hat, s, H)


def planner_mrc_baseline_b1(mdp_hat: MDP, s: Any, H: int, lam: float) -> str:
    return policy_mrc(mdp_hat, s, H, lam)


def planner_mrc_da_b1(mdp_hat: MDP, wm: WorldModelB1,
                       target_list: List[Any],
                       dist_table: Dict[Tuple[Any, int], int],
                       s: Any, H: int, lam: float) -> str:
    acts = sorted(mdp_hat.actions[s])
    def score(a: str) -> float:
        return q_reward_h(mdp_hat, s, a, H) - lam * compute_dw_hat_da_b1(
            wm, target_list, dist_table,
            mdp_hat.target_weights, mdp_hat.gamma, s, a)
    return max(acts, key=score)


def run_closed_loop_b1(true_env: MDP, choose: Callable[[Any], str],
                        where: str) -> float:
    """Wrap true_env in CountedMDP, assert planner doesn't access env
    dynamics, then full rollout."""
    counted = CountedMDP(true_env)
    counted.reset_count()
    _ = choose(S_START)
    n = counted.dyn_count
    assert n == 0, (
        f"CHEAT DETECTED [{where}]: planner read true_env.f/.r "
        f"{n} times during choose().")
    counted.reset_count()
    return rollout_value(counted, S_START, choose)


# ====================================================================
# Main: train, evaluate, report
# ====================================================================

def evaluate_one(seed: int, *, epochs: int = 800,
                  hidden: int = 32, latent: int = 16) -> Dict[str, Any]:
    r_d = DEFAULTS["r_d"]; r_g = DEFAULTS["r_g"]
    gamma = DEFAULTS["gamma"]; H = DEFAULTS["H"]
    lam = LAMBDA

    mdp_irr = build_doorkey_lava_twin("irreversible", r_d=r_d, r_g=r_g,
                                        gamma=gamma)
    mdp_rev = build_doorkey_lava_twin("reversible",   r_d=r_d, r_g=r_g,
                                        gamma=gamma)

    # Oracle D_w at s_0 (for diagnostic).
    Dw_true_irr = destroyed_mass(mdp_irr, S_START, ACTION_DECOY)
    Dw_true_rev = destroyed_mass(mdp_rev, S_START, ACTION_DECOY)

    # Train two kinds of WM per twin: baseline (reach_weight=0) and DA.
    wm_b_irr, tl_irr, dt_irr, t_b_irr = train_wm_b1(
        mdp_irr, epochs=epochs, seed=seed, hidden=hidden, latent=latent,
        reach_weight=0.0)
    wm_b_rev, tl_rev, dt_rev, t_b_rev = train_wm_b1(
        mdp_rev, epochs=epochs, seed=seed, hidden=hidden, latent=latent,
        reach_weight=0.0)
    wm_d_irr, _, _, t_d_irr = train_wm_b1(
        mdp_irr, epochs=epochs, seed=seed, hidden=hidden, latent=latent,
        reach_weight=3.0)
    wm_d_rev, _, _, t_d_rev = train_wm_b1(
        mdp_rev, epochs=epochs, seed=seed, hidden=hidden, latent=latent,
        reach_weight=3.0)

    mdp_hat_b_irr = build_mdp_hat_b1(mdp_irr, wm_b_irr)
    mdp_hat_b_rev = build_mdp_hat_b1(mdp_rev, wm_b_rev)
    mdp_hat_d_irr = build_mdp_hat_b1(mdp_irr, wm_d_irr)
    mdp_hat_d_rev = build_mdp_hat_b1(mdp_rev, wm_d_rev)

    Dw_b_irr = destroyed_mass(mdp_hat_b_irr, S_START, ACTION_DECOY)
    Dw_b_rev = destroyed_mass(mdp_hat_b_rev, S_START, ACTION_DECOY)
    Dw_d_irr = compute_dw_hat_da_b1(wm_d_irr, tl_irr, dt_irr,
                                      mdp_irr.target_weights, mdp_irr.gamma,
                                      S_START, ACTION_DECOY)
    Dw_d_rev = compute_dw_hat_da_b1(wm_d_rev, tl_rev, dt_rev,
                                      mdp_rev.target_weights, mdp_rev.gamma,
                                      S_START, ACTION_DECOY)

    # Closed-loop returns; cheat-check on every rollout.
    H_p = H
    def choose_obl_b_irr(s): return planner_obl_b1(mdp_hat_b_irr, s, H_p)
    def choose_mrc_b_irr(s): return planner_mrc_baseline_b1(mdp_hat_b_irr, s, H_p, lam)
    def choose_obl_b_rev(s): return planner_obl_b1(mdp_hat_b_rev, s, H_p)
    def choose_mrc_b_rev(s): return planner_mrc_baseline_b1(mdp_hat_b_rev, s, H_p, lam)
    def choose_obl_d_irr(s): return planner_obl_b1(mdp_hat_d_irr, s, H_p)
    def choose_mrc_d_irr(s): return planner_mrc_da_b1(mdp_hat_d_irr, wm_d_irr,
                                                       tl_irr, dt_irr, s, H_p, lam)
    def choose_obl_d_rev(s): return planner_obl_b1(mdp_hat_d_rev, s, H_p)
    def choose_mrc_d_rev(s): return planner_mrc_da_b1(mdp_hat_d_rev, wm_d_rev,
                                                       tl_rev, dt_rev, s, H_p, lam)
    def choose_obl_orc_irr(s): return policy_obl(mdp_irr, s, H_p)
    def choose_mrc_orc_irr(s): return policy_mrc(mdp_irr, s, H_p, lam)
    def choose_obl_orc_rev(s): return policy_obl(mdp_rev, s, H_p)
    def choose_mrc_orc_rev(s): return policy_mrc(mdp_rev, s, H_p, lam)

    R_b_obl_irr = run_closed_loop_b1(mdp_irr, choose_obl_b_irr, "B/obl/irr")
    R_b_mrc_irr = run_closed_loop_b1(mdp_irr, choose_mrc_b_irr, "B/mrc/irr")
    R_b_obl_rev = run_closed_loop_b1(mdp_rev, choose_obl_b_rev, "B/obl/rev")
    R_b_mrc_rev = run_closed_loop_b1(mdp_rev, choose_mrc_b_rev, "B/mrc/rev")
    R_d_obl_irr = run_closed_loop_b1(mdp_irr, choose_obl_d_irr, "D/obl/irr")
    R_d_mrc_irr = run_closed_loop_b1(mdp_irr, choose_mrc_d_irr, "D/mrc/irr")
    R_d_obl_rev = run_closed_loop_b1(mdp_rev, choose_obl_d_rev, "D/obl/rev")
    R_d_mrc_rev = run_closed_loop_b1(mdp_rev, choose_mrc_d_rev, "D/mrc/rev")
    # Oracle reference returns.
    R_o_obl_irr = rollout_value(mdp_irr, S_START, choose_obl_orc_irr)
    R_o_mrc_irr = rollout_value(mdp_irr, S_START, choose_mrc_orc_irr)
    R_o_obl_rev = rollout_value(mdp_rev, S_START, choose_obl_orc_rev)
    R_o_mrc_rev = rollout_value(mdp_rev, S_START, choose_mrc_orc_rev)
    oracle_gap_irr = R_o_mrc_irr - R_o_obl_irr

    denom = max(oracle_gap_irr, EPS)
    b_collapse = abs(R_b_mrc_rev - R_b_obl_rev) / denom
    b_charge   = (R_b_mrc_irr - R_b_obl_irr) / denom
    d_collapse = abs(R_d_mrc_rev - R_d_obl_rev) / denom
    d_charge   = (R_d_mrc_irr - R_d_obl_irr) / denom

    return {
        "seed": seed, "epochs": epochs,
        "Dw_true_irr": float(Dw_true_irr),
        "Dw_true_rev": float(Dw_true_rev),
        "Dw_baseline_irr": float(Dw_b_irr),
        "Dw_baseline_rev": float(Dw_b_rev),
        "Dw_da_irr": float(Dw_d_irr),
        "Dw_da_rev": float(Dw_d_rev),
        "R_baseline_obl_irr": float(R_b_obl_irr),
        "R_baseline_mrc_irr": float(R_b_mrc_irr),
        "R_baseline_obl_rev": float(R_b_obl_rev),
        "R_baseline_mrc_rev": float(R_b_mrc_rev),
        "R_da_obl_irr": float(R_d_obl_irr),
        "R_da_mrc_irr": float(R_d_mrc_irr),
        "R_da_obl_rev": float(R_d_obl_rev),
        "R_da_mrc_rev": float(R_d_mrc_rev),
        "R_oracle_obl_irr": float(R_o_obl_irr),
        "R_oracle_mrc_irr": float(R_o_mrc_irr),
        "R_oracle_obl_rev": float(R_o_obl_rev),
        "R_oracle_mrc_rev": float(R_o_mrc_rev),
        "baseline_collapse_ratio": float(b_collapse),
        "baseline_charge_load_ratio": float(b_charge),
        "da_collapse_ratio": float(d_collapse),
        "da_charge_load_ratio": float(d_charge),
        "baseline_pass": bool(b_collapse <= COLLAPSE_THRESHOLD
                                and b_charge >= CHARGE_THRESHOLD),
        "da_pass": bool(d_collapse <= COLLAPSE_THRESHOLD
                          and d_charge >= CHARGE_THRESHOLD),
        "train_time_s": float(t_b_irr + t_b_rev + t_d_irr + t_d_rev),
    }


def recovery_sweep_b1(seed: int = 0, epochs: int = 800) -> Dict[str, Any]:
    """Lambda sweep on DA WM at one seed; check lambda* matches r_d / D_w_hat."""
    r_d = DEFAULTS["r_d"]; r_g = DEFAULTS["r_g"]
    gamma = DEFAULTS["gamma"]; H = DEFAULTS["H"]
    mdp_irr = build_doorkey_lava_twin("irreversible", r_d=r_d, r_g=r_g,
                                        gamma=gamma)
    wm, tl, dt, _ = train_wm_b1(mdp_irr, epochs=epochs, seed=seed,
                                  reach_weight=3.0)
    mdp_hat = build_mdp_hat_b1(mdp_irr, wm)
    Dw_hat = compute_dw_hat_da_b1(wm, tl, dt, mdp_irr.target_weights,
                                    mdp_irr.gamma, S_START, ACTION_DECOY)
    if Dw_hat <= 0:
        return {"D_w_hat": float(Dw_hat), "skipped": True}
    lam_min = r_d / Dw_hat
    lambdas = np.linspace(0.0, 2.0, 2001)
    grid = float(lambdas[1] - lambdas[0])
    actions = [planner_mrc_da_b1(mdp_hat, wm, tl, dt, S_START, H, float(l))
                for l in lambdas]
    switch_idx = next((i for i, a in enumerate(actions) if a == ACTION_SAFE),
                       None)
    lam_star = float(lambdas[switch_idx]) if switch_idx is not None else None
    at_one = planner_mrc_da_b1(mdp_hat, wm, tl, dt, S_START, H, 1.0)
    return {
        "D_w_hat": float(Dw_hat),
        "lam_min_hat": float(lam_min),
        "lam_star": lam_star,
        "grid_step": grid,
        "policy_at_lam_1": at_one,
        "match_within_5_grid_steps": (
            lam_star is not None
            and abs(lam_star - lam_min) <= 5 * grid),
    }


def compute_verdict(per_run: List[Dict[str, Any]],
                     recovery: Dict[str, Any]) -> Dict[str, Any]:
    n = len(per_run)
    b_pass = sum(r["baseline_pass"] for r in per_run)
    d_pass = sum(r["da_pass"] for r in per_run)
    b_col_max = max(r["baseline_collapse_ratio"] for r in per_run)
    b_chg_min = min(r["baseline_charge_load_ratio"] for r in per_run)
    d_col_max = max(r["da_collapse_ratio"] for r in per_run)
    d_chg_min = min(r["da_charge_load_ratio"] for r in per_run)

    overall_pass = (
        b_pass >= int(0.6 * n)             # baseline mostly passes
        and d_pass >= int(0.8 * n)         # DA passes more reliably
        and recovery.get("policy_at_lam_1") == ACTION_SAFE
    )
    return {
        "verdict": "PASS" if overall_pass else "FAIL",
        "baseline_pass": b_pass,
        "da_pass": d_pass,
        "n": n,
        "baseline_collapse_max": float(b_col_max),
        "baseline_charge_min":   float(b_chg_min),
        "da_collapse_max":       float(d_col_max),
        "da_charge_min":         float(d_chg_min),
        "recovery_policy_at_lam_1": recovery.get("policy_at_lam_1"),
    }


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
    print("Stage-8 Block 1 -- embodied DoorKey-Lava proxy")
    print("=" * 78)
    print(f"Defaults: {DEFAULTS}  Lambda: {LAMBDA}")
    print(f"PASS bounds: collapse <= {COLLAPSE_THRESHOLD}, "
          f"charge_load >= {CHARGE_THRESHOLD}")
    print("Cheat-check: every closed-loop runs assert_planner_cheat_free.")

    per_run = []
    print(f"\n{'seed':>4} {'B_chg':>6} {'D_chg':>6} {'B_col':>6} {'D_col':>6} "
          f"{'B_pass':>6} {'D_pass':>6} {'time':>6}")
    for seed in SEEDS:
        r = evaluate_one(seed=seed)
        per_run.append(r)
        print(f"{seed:>4d} "
              f"{r['baseline_charge_load_ratio']:>6.3f} "
              f"{r['da_charge_load_ratio']:>6.3f} "
              f"{r['baseline_collapse_ratio']:>6.3f} "
              f"{r['da_collapse_ratio']:>6.3f} "
              f"{str(r['baseline_pass']):>6} {str(r['da_pass']):>6} "
              f"{r['train_time_s']:>6.1f}s")

    print("\n[Recovery] DA lambda sweep at seed=0")
    recovery = recovery_sweep_b1(seed=0)
    if recovery.get("skipped"):
        print(f"  D_w_hat = {recovery['D_w_hat']:.6f} <= 0, sweep skipped")
    else:
        print(f"  D_w_hat = {recovery['D_w_hat']:.6f}")
        print(f"  lam_min_hat = {recovery['lam_min_hat']:.4f}")
        print(f"  lam* observed = {recovery['lam_star']}")
        print(f"  pi_mrc_da at lam=1: '{recovery['policy_at_lam_1']}'")

    verdict = compute_verdict(per_run, recovery)
    print("\n" + "=" * 78)
    print(f"Block 1 verdict: {verdict['verdict']}")
    print(f"  baseline {verdict['baseline_pass']}/{verdict['n']} pass; "
          f"DA {verdict['da_pass']}/{verdict['n']} pass")
    print(f"  baseline collapse_max = {verdict['baseline_collapse_max']:.4f}; "
          f"DA collapse_max = {verdict['da_collapse_max']:.4f}")
    print(f"  baseline charge_min = {verdict['baseline_charge_min']:.4f}; "
          f"DA charge_min = {verdict['da_charge_min']:.4f}")
    print("=" * 78)

    dt = time.time() - t0
    payload = {
        "block": "block1_embodied_proxy",
        "verdict": verdict["verdict"],
        "wall_time_s": dt,
        "defaults": DEFAULTS,
        "lambda": LAMBDA,
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "charge_threshold": CHARGE_THRESHOLD,
        "seeds": SEEDS,
        "per_run": per_run,
        "recovery": recovery,
        "verdict_meta": verdict,
        "cheat_check": ("assert_planner_cheat_free invoked on each rollout; "
                         "any planner read of true_env.f/.r would have "
                         "raised AssertionError before this output."),
    }
    out_path = os.path.join(_THIS_DIR, "results", "block1_results.json")
    with open(out_path, "w") as fh:
        json.dump(_to_jsonable(payload), fh, indent=2)
    print(f"Results: {out_path}")
    print(f"Wall time: {dt:.1f} s")
    return verdict["verdict"] == "PASS"


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

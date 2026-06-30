"""
experiments/stage5_learned_wm/stage5_learned_wm.py
====================================================

Stage-5 Kill-Gate 2 -- learned latent world model + MPPI-style
decision-time planning with D_w_hat derived from the world model's own
predicted reachability.

Critical context (READ FIRST -- this fixes Phase-1's failure mode)
-----------------------------------------------------------------
  Phase 1 (Stage-3) tried to use a *standalone* learned MLP regressor
  D_w_hat(s, a) as reward shaping inside a model-free PPO loop, training
  1M steps.  All three seeds failed: the regressor's ~2% error compounded
  non-linearly through long training into a policy-level bias, and the
  reversible-twin collapse no longer held.

  This Stage-5 file ENFORCES three structural differences that exist
  precisely BECAUSE Phase 1 failed.  Removing any of them = Phase 1 redux.

    (1) D_w_hat is NOT a standalone regressor.  It is computed from the
        world model's OWN predicted transition graph: nearest-neighbour
        decoding of the learned latent dynamics produces a predicted
        next-state function f_hat; Stage-1's destroyed_mass is then
        invoked verbatim on the resulting MDP_hat.  D_w_hat error is
        bound to world-model prediction error -- NOT to a separate
        approximation stack stacked on top.

    (2) D_w_hat enters the planning objective at DECISION TIME (per-step
        action scoring on imagined latent rollouts), NOT inside a
        gradient/reward-shaping training loop.  Per-step argmax cannot
        non-linearly compound the error over 1M optimisation steps.

    (3) The matched reversible-twin collapse is treated as a first-class
        diagnostic: if the gap does NOT collapse on the reversible twin,
        Level 0 FAILs and we do NOT continue to Level 1 or "save" the
        result by adding training.  This is the falsification handle
        Phase 1 lacked.

Architecture (minimal TD-MPC2-style world model; CPU; no GPU required)
----------------------------------------------------------------------
  encoder   h_theta : phi(s)  -> z in R^D           (small MLP)
  dynamics  d_theta : (z, a)  -> z'                 (small MLP)
  reward    r_theta : (z, a)  -> R                  (small MLP)

  There is NO separate D_w head.  D_w_hat is a READ-OUT from the world
  model's predicted transition graph, not a learned function.

D_w_hat construction (the critical step)
----------------------------------------
  Step 1 -- encode every known twin state:
            z_j = h_theta(phi(s_j))  for s_j in true MDP.states.
  Step 2 -- predict next-latent under the learned dynamics:
            zhat = d_theta(h_theta(phi(s)), a)
  Step 3 -- decode to a discrete next-state name via nearest neighbour:
            f_hat(s, a) = argmin_{s_j} || zhat - z_j ||_2.
            r_hat(s, a) = r_theta(h_theta(phi(s)), a)         (scalar).
  Step 4 -- assemble MDP_hat = (states, actions, f_hat, r_hat, targets,
            target_weights, gamma).  The target set + weights + gamma
            are part of the task specification (NOT predicted).
  Step 5 -- invoke Stage-1's destroyed_mass(MDP_hat, s, a) verbatim.
            (Asserted at module load that this is the ONE-AND-ONLY
            destroyed_mass function in the project.)

MPPI-style decision-time planner
--------------------------------
  At every decision step the agent runs an H-step lookahead OVER MDP_hat
  and picks an action by argmax of the planning objective:
        reward_only :  Q^reward_H_hat(s, a)
        mrc         :  Q^reward_H_hat(s, a) - lambda * D_w_hat(s, a)
        oracle_mrc  :  Q^reward_H(s, a)     - lambda * D_w(s, a)
                       computed on the TRUE MDP -- upper-reference only.
  The action set per state is small (<=2 in the LavaCorridor twin), so
  the canonical MPPI sample-then-importance-weight loop reduces to exact
  enumeration; we use Stage-1's policy_obl / policy_mrc verbatim on
  MDP_hat (those are H-step lookahead argmax planners -- exactly the
  decision-time MPPI objective under a degenerate sampling distribution
  for a finite small action space).
  The three planners share ALL code; the ONLY difference is whether
  D_w_hat enters the per-action score, and whether the lookahead runs
  over MDP_hat or the true MDP (the latter only for oracle_mrc).

Twin pair
---------
  Reuses experiments/stage4_modelbased/build_lava_gridworld -- the same
  matched reversible/irreversible 2D LavaCorridor twin from Stage 4.
  Topology unchanged so the only variable under test is the learned
  world model + D_w_hat read-out, not the MDP structure.

Level 0 -- minimum training, collapse-only test (LOCKED before any run)
-----------------------------------------------------------------------
  Goal: with the SMALLEST training that gives a "reasonable" world model
        (>= 90% transition accuracy), verify that the matched-twin
        collapse near-survives.

  Pre-registered PASS conditions (both must hold):
    (a) Collapse near-survives on the reversible twin:
          collapse_ratio = |return(mrc, rev) - return(reward_only, rev)|
                            / max(oracle_gap_irr, EPS)
          MUST be <= 0.30.
    (b) Charge stays load-bearing on the irreversible twin:
          charge_load_ratio = (return(mrc, irr) - return(reward_only, irr))
                              / max(oracle_gap_irr, EPS)
          MUST be >= 0.50.

  FAIL = report as-is; do NOT add training to "rescue"; do NOT proceed
  to Level 1.

Level 1 -- only after Level 0 PASS
----------------------------------
  - 5-seed replication of Level 0 (collapse_ratio distribution).
  - Separation sweep: k in {1..5}, gap vs D_w_hat per twin.
  - Recovery sweep:   k = 3, lambda in [0, 1.5], lambda* vs r_d / D_w_hat.
  - Training curve:   epochs in {100, 200, 400, 800, 1600},
                      D_w_hat error and collapse_ratio per epochs setting.
  Writes a PDF figure summarising the four sweeps.

Runtime / cost
--------------
  CPU only; PyTorch tiny model on a ~10-state graph.
  Level 0: < 1 minute wall time.
  Level 1: < 10 minutes wall time (no seed > 5, no sweep > 5 points).
  No GPU.  No long training.  No claim that bigger model = better;
  this is a mechanism-validation test, not a scaling study.
"""

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------
# Stage-1 + Stage-4 reuse (verbatim imports; no redefinition).
# --------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE1_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage1_unified"))
_STAGE4_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage4_modelbased"))
for _p in (_STAGE1_DIR, _STAGE4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1_unified_validation import (  # noqa: E402
    MDP,
    destroyed_mass,
    policy_obl,
    policy_mrc,
    rollout_value,
)
from stage4_modelbased_planning import (  # noqa: E402
    build_lava_gridworld, S0, LAVA,
)

assert destroyed_mass.__module__ == "stage1_unified_validation"
assert policy_obl.__module__ == "stage1_unified_validation"
assert policy_mrc.__module__ == "stage1_unified_validation"


# ====================================================================
# Observation / action encoding for the world model
# ====================================================================

# Global action vocabulary across both twins.  Position in the list IS the
# action's id used for one-hot encoding.  Both twins draw from this set.
GLOBAL_ACTIONS = ["a_decoy", "a_safe", "fwd", "recover"]
ACTION_TO_ID = {a: i for i, a in enumerate(GLOBAL_ACTIONS)}
ACTION_DIM = len(GLOBAL_ACTIONS)

OBS_DIM = 3


def phi(s: Any) -> torch.Tensor:
    """phi(s) -- observation feature vector.

    3-dim (x, y, type_id):
        type_id = 0 (regular cell), 1 (lava), 2 (absorb).
    Small and structured -- the encoder learns the latent.
    """
    if s == "absorb":
        return torch.tensor([0.0, -1.0, 2.0], dtype=torch.float32)
    x, y = s
    if s == LAVA:
        return torch.tensor([float(x), float(y), 1.0], dtype=torch.float32)
    return torch.tensor([float(x), float(y), 0.0], dtype=torch.float32)


def act_oh(a: str) -> torch.Tensor:
    v = torch.zeros(ACTION_DIM, dtype=torch.float32)
    v[ACTION_TO_ID[a]] = 1.0
    return v


# ====================================================================
# Tiny TD-MPC2-style world model
# ====================================================================

LATENT_DIM = 16


class WorldModel(nn.Module):
    """Encoder + latent dynamics + reward head.  No D_w head."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACTION_DIM,
                 latent: int = LATENT_DIM, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.reward = nn.Sequential(
            nn.Linear(latent + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, o: torch.Tensor) -> torch.Tensor:
        return self.encoder(o)

    def predict(self, z: torch.Tensor, a: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        za = torch.cat([z, a], dim=-1)
        return self.dynamics(za), self.reward(za).squeeze(-1)


def collect_transitions(mdp: MDP) -> List[Tuple[Any, str, Any, float]]:
    out = []
    for s in mdp.states:
        for a in mdp.actions.get(s, []):
            out.append((s, a, mdp.f[(s, a)], mdp.r[(s, a)]))
    return out


def train_world_model(mdp: MDP, epochs: int = 800, lr: float = 1e-3,
                      seed: int = 0, verbose: bool = False) -> WorldModel:
    """Minimal training: predict reward + latent-consistency on the
    enumerated transition set.  No rollouts, no policy-gradient anything --
    just plain supervised learning on dynamics + reward.
    """
    torch.manual_seed(seed)
    wm = WorldModel()
    opt = torch.optim.Adam(wm.parameters(), lr=lr)

    transitions = collect_transitions(mdp)
    obs_s = torch.stack([phi(t[0]) for t in transitions])
    act_a = torch.stack([act_oh(t[1]) for t in transitions])
    obs_s_next = torch.stack([phi(t[2]) for t in transitions])
    rew = torch.tensor([t[3] for t in transitions], dtype=torch.float32)

    for epoch in range(epochs):
        z = wm.encode(obs_s)
        z_next_pred, r_pred = wm.predict(z, act_a)
        # Detach target latent (otherwise both sides move -- standard
        # latent-consistency trick).
        z_next_target = wm.encode(obs_s_next).detach()
        loss_dyn = F.mse_loss(z_next_pred, z_next_target)
        loss_rew = F.mse_loss(r_pred, rew)
        loss = loss_dyn + loss_rew
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % 200 == 0:
            print(f"    epoch {epoch+1:4d}: dyn={loss_dyn.item():.6f}  "
                  f"rew={loss_rew.item():.6f}")
    return wm


# ====================================================================
# Build predicted MDP from world model (nearest-neighbour decoding)
# ====================================================================

def build_mdp_hat(true_mdp: MDP, wm: WorldModel
                  ) -> Tuple[MDP, Dict[str, Any]]:
    """Read out a discrete predicted transition graph from the WM.

    See module docstring for the four-step construction.  No standalone
    D_w regressor; the WM's predicted f_hat is fed directly into Stage-1's
    destroyed_mass.
    """
    wm.eval()
    with torch.no_grad():
        z_table = {s: wm.encode(phi(s)) for s in true_mdp.states}
        z_stack = torch.stack(list(z_table.values()))
        state_keys = list(z_table.keys())

        f_hat: Dict[Tuple[Any, str], Any] = {}
        r_hat: Dict[Tuple[Any, str], float] = {}
        per_sa: List[Dict[str, Any]] = []
        correct = 0
        total = 0
        for s in true_mdp.states:
            for a in true_mdp.actions.get(s, []):
                z = z_table[s]
                zhat, r_pred = wm.predict(z, act_oh(a))
                d = torch.norm(z_stack - zhat.unsqueeze(0), dim=-1)
                idx = int(torch.argmin(d).item())
                s_next_hat = state_keys[idx]

                f_hat[(s, a)] = s_next_hat
                r_hat[(s, a)] = float(r_pred.item())
                true_s_next = true_mdp.f[(s, a)]
                ok = (s_next_hat == true_s_next)
                per_sa.append({
                    "s": str(s), "a": a,
                    "true_s_next": str(true_s_next),
                    "pred_s_next": str(s_next_hat),
                    "true_r": float(true_mdp.r[(s, a)]),
                    "pred_r": float(r_pred.item()),
                    "ok": bool(ok),
                })
                total += 1
                correct += int(ok)

    mdp_hat = MDP(
        states=true_mdp.states,
        actions=true_mdp.actions,
        f=f_hat,
        r=r_hat,
        targets=true_mdp.targets,
        target_weights=true_mdp.target_weights,
        gamma=true_mdp.gamma,
    )
    diag = {
        "transition_accuracy": (correct / total) if total else 1.0,
        "transitions_correct": correct,
        "transitions_total": total,
        "per_sa": per_sa,
    }
    return mdp_hat, diag


# ====================================================================
# D_w_hat vs exact D_w diagnostic table
# ====================================================================

def dw_hat_vs_exact(true_mdp: MDP, mdp_hat: MDP) -> List[Dict[str, Any]]:
    rows = []
    for s in true_mdp.states:
        for a in true_mdp.actions.get(s, []):
            d_true = destroyed_mass(true_mdp, s, a)
            d_hat = destroyed_mass(mdp_hat, s, a)
            rows.append({
                "s": str(s), "a": a,
                "D_w_true": float(d_true),
                "D_w_hat": float(d_hat),
                "abs_err": float(abs(d_true - d_hat)),
            })
    return rows


# ====================================================================
# Closed-loop rollout: plan in <planner_mdp>, execute in <true_env>.
# ====================================================================

def run_planner(true_env: MDP, planner_mdp: MDP, H: int, kind: str,
                lam: float = 1.0) -> float:
    """At every state the planner uses planner_mdp (world model OR true
    MDP) to score actions; the chosen action commits in true_env.
    """
    if kind == "reward_only":
        def choose(s):
            return policy_obl(planner_mdp, s, H)
    elif kind == "mrc":
        def choose(s):
            return policy_mrc(planner_mdp, s, H, lam)
    else:
        raise ValueError(f"Unknown planner kind: {kind}")
    return rollout_value(true_env, S0, choose)


# ====================================================================
# Defaults + Level 0
# ====================================================================

DEFAULTS = dict(m=4, H=4, r_d=1.0, r_g=1.0, gamma=0.9, k=3)
LAMBDA = 1.0
EPS = 1e-9
LEVEL0_EPOCHS = 800            # "minimum training for a reasonable WM"
COLLAPSE_THRESHOLD = 0.30      # PASS iff <= 0.30
CHARGE_THRESHOLD = 0.50        # PASS iff >= 0.50


def _print_dw_table(label: str, rows: List[Dict[str, Any]]) -> None:
    max_err = max(r["abs_err"] for r in rows)
    print(f"  {label}: max |D_w_hat - D_w_true| = {max_err:.6e}  "
          f"({sum(1 for r in rows if r['abs_err'] < 1e-9)}/{len(rows)} exact)")


def level_0(seed: int = 0, epochs: int = LEVEL0_EPOCHS, k: int = None,
            verbose: bool = True) -> Dict[str, Any]:
    m = DEFAULTS["m"]; H = DEFAULTS["H"]
    r_d = DEFAULTS["r_d"]; r_g = DEFAULTS["r_g"]; gamma = DEFAULTS["gamma"]
    if k is None:
        k = DEFAULTS["k"]
    lam = LAMBDA

    mdp_irr = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                    mode="irreversible")
    mdp_rev = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                    mode="reversible")

    if verbose:
        print(f"\n[Level 0] Train WM on irreversible twin "
              f"(epochs={epochs}, seed={seed})")
    t0 = time.time()
    wm_irr = train_world_model(mdp_irr, epochs=epochs, seed=seed,
                                verbose=verbose)
    t_irr = time.time() - t0

    if verbose:
        print(f"[Level 0] Train WM on reversible   twin "
              f"(epochs={epochs}, seed={seed})")
    t0 = time.time()
    wm_rev = train_world_model(mdp_rev, epochs=epochs, seed=seed,
                                verbose=verbose)
    t_rev = time.time() - t0

    mdp_irr_hat, diag_irr = build_mdp_hat(mdp_irr, wm_irr)
    mdp_rev_hat, diag_rev = build_mdp_hat(mdp_rev, wm_rev)

    dw_rows_irr = dw_hat_vs_exact(mdp_irr, mdp_irr_hat)
    dw_rows_rev = dw_hat_vs_exact(mdp_rev, mdp_rev_hat)

    Dw_hat_irr_s0 = destroyed_mass(mdp_irr_hat, S0, "a_decoy")
    Dw_true_irr_s0 = destroyed_mass(mdp_irr, S0, "a_decoy")
    Dw_hat_rev_s0 = destroyed_mass(mdp_rev_hat, S0, "a_decoy")
    Dw_true_rev_s0 = destroyed_mass(mdp_rev, S0, "a_decoy")

    # Returns: 4 planners on each twin (learned WM + oracle on the
    # true MDP).  The planner uses mdp_hat for "learned" rows, mdp itself
    # for "oracle" rows.  Execution is always on the true twin.
    R_obl_irr_learned = run_planner(mdp_irr, mdp_irr_hat, H, "reward_only", lam)
    R_mrc_irr_learned = run_planner(mdp_irr, mdp_irr_hat, H, "mrc",        lam)
    R_obl_irr_oracle  = run_planner(mdp_irr, mdp_irr,     H, "reward_only", lam)
    R_mrc_irr_oracle  = run_planner(mdp_irr, mdp_irr,     H, "mrc",        lam)

    R_obl_rev_learned = run_planner(mdp_rev, mdp_rev_hat, H, "reward_only", lam)
    R_mrc_rev_learned = run_planner(mdp_rev, mdp_rev_hat, H, "mrc",        lam)
    R_obl_rev_oracle  = run_planner(mdp_rev, mdp_rev,     H, "reward_only", lam)
    R_mrc_rev_oracle  = run_planner(mdp_rev, mdp_rev,     H, "mrc",        lam)

    learned_gap_irr = R_mrc_irr_learned - R_obl_irr_learned
    learned_gap_rev = R_mrc_rev_learned - R_obl_rev_learned
    oracle_gap_irr  = R_mrc_irr_oracle  - R_obl_irr_oracle
    oracle_gap_rev  = R_mrc_rev_oracle  - R_obl_rev_oracle

    denom = max(oracle_gap_irr, EPS)
    collapse_ratio    = abs(learned_gap_rev) / denom
    charge_load_ratio = learned_gap_irr      / denom

    collapse_ok = (collapse_ratio    <= COLLAPSE_THRESHOLD)
    charge_ok   = (charge_load_ratio >= CHARGE_THRESHOLD)
    passed = bool(collapse_ok and charge_ok)

    if verbose:
        print(f"\n[Level 0] WM transition accuracy: "
              f"irr {diag_irr['transition_accuracy']*100:.1f}%, "
              f"rev {diag_rev['transition_accuracy']*100:.1f}%")
        _print_dw_table("irreversible D_w table", dw_rows_irr)
        _print_dw_table("reversible   D_w table", dw_rows_rev)
        print(f"  D_w(s_0, a_decoy):")
        print(f"    irr   true={Dw_true_irr_s0:.6f}  hat={Dw_hat_irr_s0:.6f}  "
              f"err={abs(Dw_true_irr_s0-Dw_hat_irr_s0):.6e}")
        print(f"    rev   true={Dw_true_rev_s0:.6f}  hat={Dw_hat_rev_s0:.6f}  "
              f"err={abs(Dw_true_rev_s0-Dw_hat_rev_s0):.6e}")

        print(f"\n[Level 0] Returns (irreversible twin, k={k}, lambda={lam}):")
        print(f"  reward_only (learned WM): {R_obl_irr_learned:.6f}")
        print(f"  mrc         (learned WM): {R_mrc_irr_learned:.6f}  "
              f"-> learned gap = {learned_gap_irr:.6f}")
        print(f"  reward_only (oracle)    : {R_obl_irr_oracle:.6f}")
        print(f"  mrc         (oracle)    : {R_mrc_irr_oracle:.6f}  "
              f"-> oracle  gap = {oracle_gap_irr:.6f}")

        print(f"\n[Level 0] Returns (reversible twin, k={k}, lambda={lam}):")
        print(f"  reward_only (learned WM): {R_obl_rev_learned:.6f}")
        print(f"  mrc         (learned WM): {R_mrc_rev_learned:.6f}  "
              f"-> learned gap = {learned_gap_rev:.6f}")
        print(f"  reward_only (oracle)    : {R_obl_rev_oracle:.6f}")
        print(f"  mrc         (oracle)    : {R_mrc_rev_oracle:.6f}  "
              f"-> oracle  gap = {oracle_gap_rev:.6f}")

        print(f"\n[Level 0] VERDICT (k={k}, seed={seed}, lambda={lam}):")
        print(f"  collapse_ratio    (|rev gap| / oracle_gap_irr) "
              f"= {collapse_ratio:.4f}   PASS <= {COLLAPSE_THRESHOLD}: "
              f"{collapse_ok}")
        print(f"  charge_load_ratio (learned_gap_irr / oracle_gap_irr) "
              f"= {charge_load_ratio:.4f}   PASS >= {CHARGE_THRESHOLD}: "
              f"{charge_ok}")
        print(f"  {'PASS' if passed else 'FAIL'}")

    return {
        "k": k, "seed": seed, "epochs": epochs, "lambda": lam,
        "wm_transition_accuracy": {
            "irreversible": diag_irr["transition_accuracy"],
            "reversible":   diag_rev["transition_accuracy"],
        },
        "wm_transitions_per_sa_irr": diag_irr["per_sa"],
        "wm_transitions_per_sa_rev": diag_rev["per_sa"],
        "dw_table_irr": dw_rows_irr,
        "dw_table_rev": dw_rows_rev,
        "dw_s0_decoy": {
            "irreversible": {"true": float(Dw_true_irr_s0),
                              "hat":  float(Dw_hat_irr_s0),
                              "abs_err": float(abs(Dw_true_irr_s0-Dw_hat_irr_s0))},
            "reversible":   {"true": float(Dw_true_rev_s0),
                              "hat":  float(Dw_hat_rev_s0),
                              "abs_err": float(abs(Dw_true_rev_s0-Dw_hat_rev_s0))},
        },
        "returns": {
            "irreversible": {
                "reward_only_learned": float(R_obl_irr_learned),
                "mrc_learned":         float(R_mrc_irr_learned),
                "reward_only_oracle":  float(R_obl_irr_oracle),
                "mrc_oracle":          float(R_mrc_irr_oracle),
            },
            "reversible": {
                "reward_only_learned": float(R_obl_rev_learned),
                "mrc_learned":         float(R_mrc_rev_learned),
                "reward_only_oracle":  float(R_obl_rev_oracle),
                "mrc_oracle":          float(R_mrc_rev_oracle),
            },
        },
        "gaps": {
            "learned_irr": float(learned_gap_irr),
            "learned_rev": float(learned_gap_rev),
            "oracle_irr":  float(oracle_gap_irr),
            "oracle_rev":  float(oracle_gap_rev),
        },
        "collapse_ratio":    float(collapse_ratio),
        "charge_load_ratio": float(charge_load_ratio),
        "collapse_ok":       collapse_ok,
        "charge_ok":         charge_ok,
        "passed":            passed,
        "wall_time": {"train_irr_s": float(t_irr),
                       "train_rev_s": float(t_rev)},
    }


# ====================================================================
# Level 1 -- only if Level 0 PASS
# ====================================================================

def level_1_multi_seed(seeds: List[int] = (0, 1, 2, 3, 4)
                       ) -> Dict[str, Any]:
    """Replicate Level 0 across multiple seeds.  Collect collapse_ratio
    distribution to verify it isn't a single-seed coincidence."""
    print("\n[Level 1] Multi-seed replication (5 seeds)")
    results = []
    for seed in seeds:
        print(f"  seed={seed}:")
        res = level_0(seed=seed, verbose=False)
        print(f"    transition acc irr={res['wm_transition_accuracy']['irreversible']*100:.1f}%  "
              f"rev={res['wm_transition_accuracy']['reversible']*100:.1f}%   "
              f"collapse_ratio={res['collapse_ratio']:.4f}   "
              f"charge_load_ratio={res['charge_load_ratio']:.4f}   "
              f"{'PASS' if res['passed'] else 'FAIL'}")
        results.append(res)
    crs = [r["collapse_ratio"] for r in results]
    clrs = [r["charge_load_ratio"] for r in results]
    passes = [r["passed"] for r in results]
    print(f"  collapse_ratio:    mean={np.mean(crs):.4f}  max={np.max(crs):.4f}  "
          f"all <= {COLLAPSE_THRESHOLD}: {all(c <= COLLAPSE_THRESHOLD for c in crs)}")
    print(f"  charge_load_ratio: mean={np.mean(clrs):.4f}  min={np.min(clrs):.4f}  "
          f"all >= {CHARGE_THRESHOLD}: {all(c >= CHARGE_THRESHOLD for c in clrs)}")
    print(f"  PASS in {sum(passes)}/{len(passes)} seeds")
    return {
        "seeds": list(seeds),
        "per_seed": [
            {"seed": r["seed"],
             "collapse_ratio": r["collapse_ratio"],
             "charge_load_ratio": r["charge_load_ratio"],
             "passed": r["passed"],
             "wm_transition_accuracy": r["wm_transition_accuracy"]}
            for r in results
        ],
        "collapse_ratios": crs,
        "charge_load_ratios": clrs,
        "all_pass": all(passes),
        "n_pass": sum(passes),
        "n_total": len(passes),
    }


def level_1_separation_sweep(ks: List[int] = (1, 2, 3, 4, 5),
                              seed: int = 0,
                              epochs: int = LEVEL0_EPOCHS
                              ) -> Dict[str, Any]:
    """For each k, train WMs on both twins, compute learned/oracle gaps."""
    print("\n[Level 1] Separation sweep over k")
    m = DEFAULTS["m"]; H = DEFAULTS["H"]
    r_d = DEFAULTS["r_d"]; r_g = DEFAULTS["r_g"]; gamma = DEFAULTS["gamma"]
    lam = LAMBDA
    rows = []
    for k in ks:
        mdp_irr = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                        mode="irreversible")
        wm_irr = train_world_model(mdp_irr, epochs=epochs, seed=seed)
        mdp_irr_hat, diag = build_mdp_hat(mdp_irr, wm_irr)
        Dw_true = destroyed_mass(mdp_irr, S0, "a_decoy")
        Dw_hat  = destroyed_mass(mdp_irr_hat, S0, "a_decoy")
        R_obl = run_planner(mdp_irr, mdp_irr_hat, H, "reward_only", lam)
        R_mrc = run_planner(mdp_irr, mdp_irr_hat, H, "mrc",        lam)
        R_obl_o = run_planner(mdp_irr, mdp_irr, H, "reward_only", lam)
        R_mrc_o = run_planner(mdp_irr, mdp_irr, H, "mrc",        lam)
        rows.append({
            "k": k,
            "D_w_true": float(Dw_true), "D_w_hat": float(Dw_hat),
            "Dw_abs_err": float(abs(Dw_true - Dw_hat)),
            "transition_accuracy": diag["transition_accuracy"],
            "learned_gap": float(R_mrc - R_obl),
            "oracle_gap":  float(R_mrc_o - R_obl_o),
            "R_obl_learned": float(R_obl),
            "R_mrc_learned": float(R_mrc),
        })
        print(f"  k={k}: D_w_true={Dw_true:.4f}  D_w_hat={Dw_hat:.4f}  "
              f"learned_gap={R_mrc-R_obl:.4f}  oracle_gap={R_mrc_o-R_obl_o:.4f}")

    # Monotonicity of learned gap in D_w_hat (sort by D_w_hat).
    sorted_rows = sorted(rows, key=lambda r: r["D_w_hat"])
    learned_gaps_sorted = [r["learned_gap"] for r in sorted_rows]
    monotone = all(learned_gaps_sorted[i+1] >= learned_gaps_sorted[i] - 1e-9
                    for i in range(len(learned_gaps_sorted) - 1))
    print(f"  learned_gap monotone non-decreasing in D_w_hat: {monotone}")
    return {"rows": rows, "monotone": monotone}


def level_1_recovery_sweep(k: int = 3, seed: int = 0,
                            epochs: int = LEVEL0_EPOCHS
                            ) -> Dict[str, Any]:
    """Sweep lambda; observed switch lambda* vs r_d / D_w_hat."""
    print("\n[Level 1] Recovery sweep over lambda (k=%d)" % k)
    m = DEFAULTS["m"]; H = DEFAULTS["H"]
    r_d = DEFAULTS["r_d"]; r_g = DEFAULTS["r_g"]; gamma = DEFAULTS["gamma"]
    mdp_irr = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                    mode="irreversible")
    wm = train_world_model(mdp_irr, epochs=epochs, seed=seed)
    mdp_hat, _ = build_mdp_hat(mdp_irr, wm)
    Dw_hat = destroyed_mass(mdp_hat, S0, "a_decoy")
    Dw_true = destroyed_mass(mdp_irr, S0, "a_decoy")
    lam_min_hat  = r_d / Dw_hat  if Dw_hat  > 0 else float("inf")
    lam_min_true = r_d / Dw_true if Dw_true > 0 else float("inf")
    lambdas = np.linspace(0.0, 1.5, 1501)
    actions = [policy_mrc(mdp_hat, S0, H, float(l)) for l in lambdas]
    switch_idx = next((i for i, a in enumerate(actions) if a == "a_safe"),
                       None)
    lam_star = float(lambdas[switch_idx]) if switch_idx is not None else None
    grid_step = float(lambdas[1] - lambdas[0])
    at_one = policy_mrc(mdp_hat, S0, H, 1.0)
    switch_err = (abs(lam_star - lam_min_hat) if lam_star is not None
                   else float("inf"))
    print(f"  D_w_hat = {Dw_hat:.6f}  (true = {Dw_true:.6f})")
    print(f"  lambda_min_hat  (theory r_d/D_w_hat)  = {lam_min_hat:.6f}")
    print(f"  lambda_min_true (theory r_d/D_w_true) = {lam_min_true:.6f}")
    print(f"  lambda* observed                       = {lam_star}")
    print(f"  |lambda* - lambda_min_hat| = {switch_err:.6f}  "
          f"(grid step {grid_step:.4f})")
    print(f"  pi_mrc(s_0) at lambda = 1.0: '{at_one}'")
    return {
        "k": k, "D_w_hat": float(Dw_hat), "D_w_true": float(Dw_true),
        "lam_min_hat": float(lam_min_hat),
        "lam_min_true": float(lam_min_true),
        "lam_star": lam_star, "grid_step": float(grid_step),
        "switch_err_vs_hat":  float(switch_err),
        "policy_at_lam_1": at_one,
    }


def level_1_training_curve(epochs_list: List[int] = (100, 200, 400, 800, 1600),
                             seed: int = 0) -> Dict[str, Any]:
    """For each training-epochs setting, train fresh WMs on both twins
    and report D_w_hat error + collapse_ratio."""
    print("\n[Level 1] Training-curve sweep over epochs")
    rows = []
    for epochs in epochs_list:
        res = level_0(seed=seed, epochs=epochs, verbose=False)
        max_err_irr = max(r["abs_err"] for r in res["dw_table_irr"])
        max_err_rev = max(r["abs_err"] for r in res["dw_table_rev"])
        rows.append({
            "epochs": epochs,
            "max_Dw_abs_err_irr": float(max_err_irr),
            "max_Dw_abs_err_rev": float(max_err_rev),
            "Dw_s0_abs_err_irr": res["dw_s0_decoy"]["irreversible"]["abs_err"],
            "Dw_s0_abs_err_rev": res["dw_s0_decoy"]["reversible"]["abs_err"],
            "transition_acc_irr": res["wm_transition_accuracy"]["irreversible"],
            "transition_acc_rev": res["wm_transition_accuracy"]["reversible"],
            "collapse_ratio": res["collapse_ratio"],
            "charge_load_ratio": res["charge_load_ratio"],
            "passed": res["passed"],
        })
        print(f"  epochs={epochs:5d}: tr_acc irr={res['wm_transition_accuracy']['irreversible']*100:.1f}% "
              f"rev={res['wm_transition_accuracy']['reversible']*100:.1f}%  "
              f"D_w err irr={max_err_irr:.4f} rev={max_err_rev:.4f}  "
              f"collapse_ratio={res['collapse_ratio']:.4f}  "
              f"charge_load={res['charge_load_ratio']:.4f}")
    return {"rows": rows}


# ====================================================================
# Figure
# ====================================================================

def write_figure(level1: Dict[str, Any], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) Multi-seed collapse_ratio
    crs = level1["multi_seed"]["collapse_ratios"]
    clrs = level1["multi_seed"]["charge_load_ratios"]
    seeds = level1["multi_seed"]["seeds"]
    ax = axes[0, 0]
    ax.bar(np.arange(len(crs)) - 0.18, crs, width=0.36, label="collapse_ratio")
    ax.bar(np.arange(len(clrs)) + 0.18, clrs, width=0.36,
            label="charge_load_ratio")
    ax.axhline(COLLAPSE_THRESHOLD, color="r", ls="--", lw=1,
                label=f"collapse PASS <= {COLLAPSE_THRESHOLD}")
    ax.axhline(CHARGE_THRESHOLD,   color="g", ls="--", lw=1,
                label=f"charge    PASS >= {CHARGE_THRESHOLD}")
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("ratio")
    ax.set_title("(a) Multi-seed Level-0 (k=3, lambda=1)")
    ax.legend(loc="best", fontsize=8)

    # (b) Separation sweep
    rows = level1["separation"]["rows"]
    ks = [r["k"] for r in rows]
    learned = [r["learned_gap"] for r in rows]
    oracle  = [r["oracle_gap"]  for r in rows]
    dw_hat  = [r["D_w_hat"]     for r in rows]
    dw_true = [r["D_w_true"]    for r in rows]
    ax = axes[0, 1]
    ax.plot(ks, learned, "o-", label="learned gap (mrc - reward_only)")
    ax.plot(ks, oracle, "x--", label="oracle  gap")
    ax2 = ax.twinx()
    ax2.plot(ks, dw_hat, "s:", color="purple", label="D_w_hat")
    ax2.plot(ks, dw_true, "+--", color="grey",  label="D_w_true")
    ax.set_xlabel("k (number of corridor targets)")
    ax.set_ylabel("return gap")
    ax2.set_ylabel("D_w")
    ax.set_title("(b) Separation sweep on irreversible twin")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)

    # (c) Recovery
    rec = level1["recovery"]
    ax = axes[1, 0]
    ax.axvline(rec["lam_min_hat"], color="purple", ls="--",
                label=f"lambda_min_hat = {rec['lam_min_hat']:.3f}")
    ax.axvline(rec["lam_min_true"], color="grey",  ls=":",
                label=f"lambda_min_true = {rec['lam_min_true']:.3f}")
    if rec["lam_star"] is not None:
        ax.axvline(rec["lam_star"], color="red", ls="-",
                    label=f"lambda* observed = {rec['lam_star']:.3f}")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("lambda")
    ax.set_title("(c) Recovery: switch point at s_0")
    ax.legend(loc="best", fontsize=8)

    # (d) Training-curve sweep
    rows = level1["training_curve"]["rows"]
    eps  = [r["epochs"] for r in rows]
    crs2 = [r["collapse_ratio"] for r in rows]
    cls2 = [r["charge_load_ratio"] for r in rows]
    accs_irr = [r["transition_acc_irr"] for r in rows]
    accs_rev = [r["transition_acc_rev"] for r in rows]
    ax = axes[1, 1]
    ax.plot(eps, crs2, "o-", label="collapse_ratio")
    ax.plot(eps, cls2, "s-", label="charge_load_ratio")
    ax.plot(eps, accs_irr, "x--", label="WM tr.acc irr")
    ax.plot(eps, accs_rev, "+--", label="WM tr.acc rev")
    ax.axhline(COLLAPSE_THRESHOLD, color="r", ls="--", lw=0.8)
    ax.axhline(CHARGE_THRESHOLD,   color="g", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("training epochs (log)")
    ax.set_title("(d) Training-curve sweep")
    ax.legend(loc="best", fontsize=8)

    plt.suptitle("Stage-5 Kill-Gate 2 -- learned WM + D_w_hat (Level 1)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


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
    t_start = time.time()
    print("=" * 78)
    print("Stage-5 Kill-Gate 2 -- learned latent world model + D_w_hat from"
          " predicted reachability")
    print("=" * 78)
    print(f"Defaults: {DEFAULTS}")
    print(f"Lambda: {LAMBDA}   Level-0 epochs: {LEVEL0_EPOCHS}")
    print(f"PASS conditions: collapse_ratio <= {COLLAPSE_THRESHOLD},  "
          f"charge_load_ratio >= {CHARGE_THRESHOLD}")
    print(f"Stage-4 / Stage-1 reuse: build_lava_gridworld, destroyed_mass,"
          f" policy_obl, policy_mrc, rollout_value (verbatim imports)")
    print("D_w_hat read-out: nearest-neighbour decode of latent dynamics -> "
          "MDP_hat -> Stage-1 destroyed_mass(MDP_hat, s, a).  NO standalone"
          " D_w regressor.")

    # Level 0 first.
    l0 = level_0(seed=0, verbose=True)

    if not l0["passed"]:
        print("\n" + "=" * 78)
        print("Level 0 FAIL -- do NOT proceed to Level 1.")
        print("Per pre-registered protocol, do NOT add training to rescue"
              " the result.")
        print(f"  collapse_ratio    = {l0['collapse_ratio']:.4f}"
              f" (PASS <= {COLLAPSE_THRESHOLD}, ok: {l0['collapse_ok']})")
        print(f"  charge_load_ratio = {l0['charge_load_ratio']:.4f}"
              f" (PASS >= {CHARGE_THRESHOLD}, ok: {l0['charge_ok']})")
        print("=" * 78)
        payload = {
            "overall_pass": False,
            "level_0": l0,
            "level_1": None,
            "wall_time_s": time.time() - t_start,
            "defaults": DEFAULTS, "lambda": LAMBDA,
            "level_0_epochs": LEVEL0_EPOCHS,
            "collapse_threshold": COLLAPSE_THRESHOLD,
            "charge_threshold": CHARGE_THRESHOLD,
        }
        out_path = os.path.join(_THIS_DIR, "stage5_results.json")
        with open(out_path, "w") as fh:
            json.dump(_to_jsonable(payload), fh, indent=2)
        print(f"Results written to {out_path}")
        return False

    # Level 1.
    print("\n" + "=" * 78)
    print("Level 0 PASS -- proceeding to Level 1")
    print("=" * 78)
    l1_ms = level_1_multi_seed(seeds=[0, 1, 2, 3, 4])
    l1_sep = level_1_separation_sweep(ks=[1, 2, 3, 4, 5], seed=0)
    l1_rec = level_1_recovery_sweep(k=3, seed=0)
    l1_tc = level_1_training_curve(epochs_list=[100, 200, 400, 800, 1600],
                                    seed=0)

    level_1 = {
        "multi_seed": l1_ms,
        "separation": l1_sep,
        "recovery":  l1_rec,
        "training_curve": l1_tc,
    }

    # Figure.
    pdf_path = os.path.join(_THIS_DIR, "stage5_level1.pdf")
    write_figure(level_1, pdf_path)
    print(f"\nFigure written to {pdf_path}")

    # Final verdict.
    l1_pass = (
        l1_ms["all_pass"]
        and l1_sep["monotone"]
        and l1_rec["lam_star"] is not None
        and abs(l1_rec["lam_star"] - l1_rec["lam_min_hat"]) <= 2 * l1_rec["grid_step"]
        and l1_rec["policy_at_lam_1"] == "a_safe"
    )
    overall_pass = bool(l0["passed"] and l1_pass)

    print("\n" + "=" * 78)
    print("Stage-5 overall verdict")
    print("=" * 78)
    print(f"  Level 0 (k=3, seed=0)              : "
          f"{'PASS' if l0['passed'] else 'FAIL'}")
    print(f"  Level 1 multi-seed (5 seeds)       : "
          f"{l1_ms['n_pass']}/{l1_ms['n_total']} pass  "
          f"({'OK' if l1_ms['all_pass'] else 'NOT OK'})")
    print(f"  Level 1 separation (gap monotone)  : "
          f"{'OK' if l1_sep['monotone'] else 'NOT OK'}")
    print(f"  Level 1 recovery (lambda*)         : "
          f"observed={l1_rec['lam_star']}  vs theory_hat={l1_rec['lam_min_hat']:.4f}"
          f"  ({'OK' if l1_rec['lam_star'] is not None and abs(l1_rec['lam_star']-l1_rec['lam_min_hat'])<=2*l1_rec['grid_step'] else 'NOT OK'})")
    print(f"  Overall                            : "
          f"{'ALL PASS' if overall_pass else 'FAIL'}")
    dt = time.time() - t_start
    print(f"  Wall time: {dt:.1f} s")
    if overall_pass:
        print("Kill-Gate 2 cleared: learned world model + decision-time D_w_hat"
              " preserves MRC three properties on the embodied twin.")
        print("D_w object is native to learned-world-model based planning.")
    else:
        print("Kill-Gate 2 NOT cleared.  See above for failed checks.")

    payload = {
        "overall_pass": overall_pass,
        "level_0": l0,
        "level_1": level_1,
        "wall_time_s": dt,
        "defaults": DEFAULTS, "lambda": LAMBDA,
        "level_0_epochs": LEVEL0_EPOCHS,
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "charge_threshold": CHARGE_THRESHOLD,
    }
    out_path = os.path.join(_THIS_DIR, "stage5_results.json")
    with open(out_path, "w") as fh:
        json.dump(_to_jsonable(payload), fh, indent=2)
    print(f"Results written to {out_path}")
    return overall_pass


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

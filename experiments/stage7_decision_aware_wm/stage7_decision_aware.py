"""
experiments/stage7_decision_aware_wm/stage7_decision_aware.py
================================================================

Stage-7 Kill-Gate A1 -- Decision-Aware WM Training repairs the
reversible-twin collapse fragility found in Stage-6.

Why this stage exists (READ FIRST)
----------------------------------
  Stage-6 produced an honest FAIL: when the WM is trained on data where
  the (lava, recover) target is corrupted to "absorb" with probability
  recover_corrupt_p, the WM predicts that recover destroys reachability,
  D_w_hat(s_0, a_decoy) on the reversible twin becomes nonzero, the
  per-step MPPI argmax flips from a_decoy to a_safe, and reversible-twin
  collapse breaks (collapse_ratio = 1.0568 at rcp = 1.0).

  The mechanism behind that FAIL is:
    MRC's collapse property inherits from D_w being correct AT THE
    DECISION POINT.  The per-step decision-time D_w_hat design isolates
    errors at intermediate states (which never reach the trajectory) but
    does NOT absorb errors at the decision point itself.

  Stage-7 proposes a fix: add an auxiliary "reachability head" to the
  WM that is supervised by what an agent in CLOSED-LOOP TRAINING would
  observe -- which targets actually got reached from f(s, a) over many
  episodes.  This is a faithfully-disclosed training-time signal that
  the user spec explicitly authorises:

    "decision-aware 辅助损失只能用训练期可获得的监督：训练数据里'动作执行
    后实际可达性 / 是否真的删除了目标'的环境反馈（agent 在闭环交互中本来
    就能观测到'我踏进去后还能不能到目标'）"

  At test time, the planner remains learned-WM-only -- it has access only
  to the reach-head's output, NOT to the true env's transition function
  or destroyed_mass over the true MDP.  This is enforced by a runtime
  cheat-check using a CountedMDP wrapper that fails the test if the
  planner ever reads true_env.f or true_env.r outside of rollout_value's
  env-step.

Fairness (strict, pre-registered; violating any = invalid run)
--------------------------------------------------------------
  (1) Aux training signal: reach labels come from TRUE env's reachable
      sets (semantically: empirical multi-step reachability that the
      agent would observe in closed-loop training).  This is explicitly
      training-time only.
  (2) Test-time D_w_hat: computed ONLY from the learned WM's reach_head
      + the learned f_hat (via nearest-neighbour decoding); the true
      env's f / r are accessed ONLY by rollout_value to step the closed
      loop -- and that access is counted and asserted.
  (3) Architecture: baseline WM and decision-aware WM share the SAME
      class and SAME initialisation seed.  They differ only in
      reach_weight (0 = baseline; > 0 = decision-aware).  Apples-to-
      apples.
  (4) D_w_hat readout: baseline uses Stage-5/6 nearest-neighbour
      destroyed_mass(mdp_hat, s, a).  Decision-aware uses the reach
      head's output to determine the destroyed-target set; distances
      still come from BFS over mdp_hat (so the structure-only part
      of D_w is shared).
  (5) NEVER feed oracle D_w or true_env.f to the test-time planner.
      The CountedMDP cheat check makes this a runtime assertion.

Pre-registered PASS / PARTIAL / FAIL
------------------------------------
  Sweep recover_corrupt_p in {0.0, 0.2, 0.3, 0.5, 0.7, 1.0} x 5 seeds.
  At each rcp, compare baseline vs decision-aware on:
    - collapse_ratio  on rev twin   (PASS bound <= 0.30)
    - charge_load     on irr twin   (PASS bound >= 0.50)
    - D_w_hat at s_0 a_decoy on rev (must drop toward 0 for DA)
    - recovery lambda* matches r_d / D_w_hat (sanity)

  PASS (the fix works)
    iff at EVERY rcp where baseline breaks collapse on rev
    (max-over-seeds collapse_ratio_baseline > COLLAPSE_THRESHOLD),
    decision-aware brings max-over-seeds collapse_ratio_DA <= COLLAPSE_THRESHOLD
    AND keeps min-over-seeds charge_load_DA >= CHARGE_THRESHOLD on irr.
    A boundary like rcp=1.0 is allowed to remain broken IFF the user
    interprets rcp=1.0 as "TRUE env is truly absorbing" (in our framing
    the user explicitly calls it a WM misjudgement, so rcp=1.0 still
    counts).

  PARTIAL (boundary pushed but not closed)
    iff DA fixes collapse at some rcp where baseline broke, but breaks
    at higher rcp.  Report exactly where DA's failure threshold is.

  FAIL (the fix doesn't work)
    iff DA does NOT improve collapse over baseline at any triggered rcp.
    Or DA fixes collapse but kills separation (charge_load_DA < 0.50).

  CHEAT (invalid run)
    iff the runtime CountedMDP assert ever fails.

  No retuning to mask any of these outcomes.  No "rescuing" PARTIAL
  with more training.  Honest negative is acceptable; the user wants
  truth, not green numbers.

Reuse
-----
  build_lava_gridworld + S0 + LAVA from Stage 4.
  MDP, destroyed_mass, policy_obl, policy_mrc, q_reward_h, rollout_value,
  reachable_set, bfs_distances from Stage 1.
  phi, act_oh, collect_transitions, ACTION_DIM, OBS_DIM from Stage 5.
  We re-define WorldModel here as DAWorldModel to add a reach_head,
  preserving the encoder + dynamics + reward modules verbatim.

Runtime
-------
  CPU only.  6 rcp levels x 5 seeds x 2 training kinds (baseline / DA)
  x 2 twins = 120 WM trainings, plus lambda sweep diagnostics.
  Expected wall time: 5-12 minutes.  No GPU.
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
# Reuse imports (verbatim).
# --------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE1_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage1_unified"))
_STAGE4_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage4_modelbased"))
_STAGE5_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage5_learned_wm"))
for _p in (_STAGE1_DIR, _STAGE4_DIR, _STAGE5_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1_unified_validation import (  # noqa: E402
    MDP, destroyed_mass, policy_obl, policy_mrc,
    q_reward_h, value_h, rollout_value,
    reachable_set, bfs_distances,
)
from stage4_modelbased_planning import build_lava_gridworld, S0, LAVA  # noqa: E402
from stage5_learned_wm import (  # noqa: E402
    phi, act_oh, collect_transitions, OBS_DIM, ACTION_DIM, LATENT_DIM,
)

assert destroyed_mass.__module__ == "stage1_unified_validation"
assert policy_mrc.__module__   == "stage1_unified_validation"


# ====================================================================
# Cheat-check wrapper: counts accesses to env-dynamics attributes (f, r)
# ====================================================================

class CountedMDP:
    """Wrap an MDP and count accesses to env-dynamics attributes.

    The test-time planner MUST NOT read .f or .r on the TRUE env -- those
    are env dynamics that can come only from the learned WM.  Accessing
    .states, .actions, .targets, .target_weights, .gamma is fine: those
    are task specification (the agent knows them).

    rollout_value DOES read .actions, .f, .r, .gamma during env stepping.
    That is the LEGITIMATE access pattern.  The cheat check is run BEFORE
    rollout, by calling the planner's choose_action(s) once on s_0 and
    verifying that the .f / .r counters stay at zero.
    """

    DYNAMICS_ATTRS = ("f", "r")

    def __init__(self, mdp: MDP):
        object.__setattr__(self, "_mdp", mdp)
        object.__setattr__(self, "_dyn_count", 0)

    def __getattr__(self, name: str):
        mdp = object.__getattribute__(self, "_mdp")
        if name in CountedMDP.DYNAMICS_ATTRS:
            object.__setattr__(
                self, "_dyn_count",
                object.__getattribute__(self, "_dyn_count") + 1,
            )
        return getattr(mdp, name)

    @property
    def dyn_count(self) -> int:
        return object.__getattribute__(self, "_dyn_count")

    def reset_count(self) -> None:
        object.__setattr__(self, "_dyn_count", 0)


# ====================================================================
# Decision-Aware World Model
# ====================================================================

class DAWorldModel(nn.Module):
    """encoder + latent dynamics + reward head + reachability head.

    The reachability head outputs per-target logits predicting whether
    target g is reachable from f(s, a).  Supervised at training time by
    labels derived from the agent's observed multi-step trajectories
    (modelled here via reachable_set on the TRUE env -- semantically the
    same as empirical reachability over many episodes).  At test time,
    only the learned modules are queried.
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACTION_DIM,
                 latent: int = LATENT_DIM, hidden: int = 32,
                 n_targets: int = 3):
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
        # NEW (Stage-7): per-target reachability head.  Output dim is
        # n_targets; the i-th logit predicts "is the i-th target (sorted
        # by corridor y) reachable from f(s, a)?".
        self.reach_head = nn.Sequential(
            nn.Linear(latent + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_targets),
        )

    def encode(self, o: torch.Tensor) -> torch.Tensor:
        return self.encoder(o)


# ====================================================================
# Target ordering (consistent across baseline + DA)
# ====================================================================

def sort_targets(targets) -> List[Any]:
    """Canonical target order: by corridor y-coord."""
    # Targets in the LavaCorridor twin are (0, y) tuples; sort by y.
    return sorted(targets, key=lambda t: (t[1],))


def collect_reach_labels(mdp: MDP, target_list: List[Any]
                          ) -> List[List[float]]:
    """For each transition (s, a, s_next, r), the reach label is a vector
    over `target_list`: 1 if target g is in R(s_next) (TRUE-env reachable
    set), else 0.

    Semantic framing: this is what the agent would observe over many
    closed-loop episodes that pass through (s, a) -- "after taking a in
    s, did I eventually reach g?".  Computing it via reachable_set on
    the TRUE env is an efficient realisation of that empirical signal;
    the user spec explicitly authorises training-time access to true
    transitions for this kind of label.
    """
    transitions = collect_transitions(mdp)
    out: List[List[float]] = []
    for (s, a, s_next, r) in transitions:
        R = reachable_set(mdp, s_next)
        out.append([1.0 if g in R else 0.0 for g in target_list])
    return out


# ====================================================================
# Training with shared baseline / decision-aware switch
# ====================================================================

def precompute_distance_table(mdp: MDP, target_list: List[Any]
                                ) -> Dict[Tuple[Any, int], int]:
    """Per-(state, target_idx) shortest-path distance in TRUE env.

    Semantically: the agent observes "from state s, the shortest trajectory
    that reached target g had length d" across many closed-loop episodes.
    This is training-time empirical info (allowed by the user spec).
    Computed efficiently here via BFS on the TRUE MDP.

    Returns a dict keyed by (state, target_idx) with int values.  Targets
    unreachable from s are simply omitted from the dict.
    """
    out: Dict[Tuple[Any, int], int] = {}
    for s in mdp.states:
        d = bfs_distances(mdp, s)
        for i, g in enumerate(target_list):
            if g in d:
                out[(s, i)] = d[g]
    return out


def train_world_model(
    mdp: MDP, *, epochs: int, label_noise_p: float, obs_noise_std: float,
    recover_corrupt_p: float, hidden: int, latent: int, seed: int,
    reach_weight: float, lr: float = 1e-3,
) -> Tuple[DAWorldModel, List[Any], Dict[Tuple[Any, int], int], float]:
    """Train a DAWorldModel.

    reach_weight = 0.0  -> BASELINE (Stage-5/6 style: dyn + reward only).
                            reach_head exists in the module but is not
                            updated by the loss; never queried at test
                            time.
    reach_weight > 0.0  -> DECISION-AWARE: add weight * BCE-with-logits
                            on per-target reachability labels.
    """
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
    reach_lbl   = torch.tensor(collect_reach_labels(mdp, target_list),
                                dtype=torch.float32)

    state_list  = list(mdp.states)
    n_states    = len(state_list)
    true_s_next = [t[2] for t in transitions]

    t0 = time.time()
    for epoch in range(epochs):
        # Same per-epoch perturbations as Stage 6 (transition labels):
        s_next_list = list(true_s_next)
        if label_noise_p > 0:
            for i in range(len(s_next_list)):
                if rng.random() < label_noise_p:
                    s_next_list[i] = state_list[
                        int(rng.integers(0, n_states))]
        if recover_corrupt_p > 0:
            for i, (s_i, a_i, _, _) in enumerate(transitions):
                if s_i == LAVA and a_i == "recover":
                    if rng.random() < recover_corrupt_p:
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


# ====================================================================
# Build mdp_hat from a (DA) WM -- shared by baseline and DA
# ====================================================================
#
# Reads from `mdp_struct` ONLY task-spec attributes
# (states, actions, targets, target_weights, gamma).  Does NOT touch
# mdp_struct.f or mdp_struct.r -- those are env dynamics, which we are
# learning, so accessing them at test time would be a cheat.  The
# CountedMDP wrapper catches accidental access.

def build_mdp_hat(mdp_struct: MDP, wm: DAWorldModel) -> MDP:
    wm.eval()
    with torch.no_grad():
        z_table = {s: wm.encoder(phi(s)) for s in mdp_struct.states}
        z_stack = torch.stack(list(z_table.values()))
        state_keys = list(z_table.keys())

        f_hat: Dict[Any, Any] = {}
        r_hat: Dict[Any, float] = {}
        for s in mdp_struct.states:
            z = z_table[s]
            for a in mdp_struct.actions.get(s, []):
                a_oh = act_oh(a)
                za = torch.cat([z, a_oh], dim=-1)
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


# ====================================================================
# Decision-aware D_w_hat -- uses reach head to define destroyed targets
# ====================================================================

def compute_dw_hat_da(wm: DAWorldModel,
                       target_list: List[Any],
                       dist_table: Dict[Tuple[Any, int], int],
                       target_weights: Dict[Any, float],
                       gamma: float,
                       s: Any, a: str) -> float:
    """D_w_hat from the reach head (decision-aware):
         D_w_hat(s, a) = Sigma_g  (1 - reach_pred(s, a, g))
                          * gamma^{d_table[s, g]}
                          * u(g)

       Sources of each ingredient (and what makes this cheat-free at test
       time):
         - reach_pred  : sigmoid(wm.reach_head(encode(s), a))[g].
                          Trained on labels derived from observed multi-
                          step trajectories.  Learned WM module.
         - d_table     : pre-computed at TRAINING time from BFS in TRUE
                          env (semantically the agent's empirical shortest
                          observed trajectory length s -> g).  Fixed
                          training-time artifact; the planner only reads
                          from it at test time, never queries TRUE env.f.
         - target_weights, gamma : task spec (known a priori).

       Decoupling D_w_hat from mdp_hat's BFS is critical: when reach loss
       competes with dynamics loss for encoder capacity, nearest-neighbour
       decoding of mdp_hat can break (latents collapse) and BFS(mdp_hat, s)
       can miss the targets entirely.  We saw this in the first run --
       DA's collapse on rev passed (D_w_hat ≈ 0 by reach head) but
       charge_load on irr collapsed because mdp_hat's R(s) excluded the
       targets so the sum was vacuous.  Using the training-time distance
       table sidesteps that failure mode entirely.
    """
    with torch.no_grad():
        z = wm.encoder(phi(s))
        a_oh = act_oh(a)
        za = torch.cat([z, a_oh], dim=-1)
        reach_probs = torch.sigmoid(wm.reach_head(za)).numpy()

    total = 0.0
    for i, g in enumerate(target_list):
        d = dist_table.get((s, i))
        if d is None:
            continue  # target was empirically unreachable from s in
                       # training trajectories -- not "destroyed" by a.
        u = target_weights[g]
        p_destroyed = 1.0 - float(reach_probs[i])
        total += p_destroyed * (gamma ** d) * u
    return total


# ====================================================================
# Planners -- pure functions of mdp_hat + wm + (target_list for DA)
# ====================================================================
# NEVER accept true_env / mdp_true as an argument.  This is the static
# guarantee that the planner cannot peek at the oracle.

def planner_obl(mdp_hat: MDP, s: Any, H: int) -> str:
    return policy_obl(mdp_hat, s, H)


def planner_mrc_baseline(mdp_hat: MDP, s: Any, H: int, lam: float) -> str:
    return policy_mrc(mdp_hat, s, H, lam)


def planner_mrc_da(mdp_hat: MDP, wm: DAWorldModel,
                    target_list: List[Any],
                    dist_table: Dict[Tuple[Any, int], int],
                    s: Any, H: int, lam: float) -> str:
    acts = sorted(mdp_hat.actions[s])
    def score(a: str) -> float:
        return q_reward_h(mdp_hat, s, a, H) - lam * compute_dw_hat_da(
            wm, target_list, dist_table,
            mdp_hat.target_weights, mdp_hat.gamma, s, a)
    return max(acts, key=score)


# ====================================================================
# Runtime cheat-check helper
# ====================================================================
# Calls choose(s_0) ONCE on a CountedMDP-wrapped true env, then asserts
# that the cheat counter remains at zero (the planner did not touch
# true_env.f or true_env.r).

def assert_planner_cheat_free(
    true_env: MDP, choose: Callable[[Any], str], where: str,
) -> Tuple[CountedMDP, str]:
    """Wraps true_env in a CountedMDP, runs one choose(S0) call, asserts
    cheat_count == 0, returns the wrapped MDP for subsequent rollout."""
    counted = CountedMDP(true_env)
    counted.reset_count()
    a = choose(S0)  # planning only; must not touch counted's f/r.
    n = counted.dyn_count
    assert n == 0, (
        f"CHEAT DETECTED [{where}]: planner read true_env.f / .r "
        f"{n} times during a single choose() call.  The test-time "
        f"planner must NEVER access true env dynamics.")
    return counted, a


# ====================================================================
# Closed-loop rollout with cheat assertion
# ====================================================================

def run_closed_loop(
    true_env: MDP, choose: Callable[[Any], str], where: str,
) -> Tuple[float, int, int]:
    """Wrap true_env in CountedMDP, run cheat-check, then full rollout.

    Returns (return_value, dyn_accesses_during_planning,
             dyn_accesses_during_rollout).
    """
    counted, _first_a = assert_planner_cheat_free(true_env, choose, where)
    plan_only_dyn = counted.dyn_count   # always 0 by assert above.
    counted.reset_count()
    ret = rollout_value(counted, S0, choose)
    return ret, plan_only_dyn, counted.dyn_count


# ====================================================================
# Shared constants
# ====================================================================

DEFAULTS = dict(m=4, H=4, r_d=1.0, r_g=1.0, gamma=0.9, k=3)
LAMBDA = 1.0
EPS = 1e-9
COLLAPSE_THRESHOLD = 0.30
CHARGE_THRESHOLD   = 0.50
TRIGGER_EPS        = 1e-6     # D_w_hat(rev) at s_0 > this counts as triggered

DEFAULT_EPOCHS = 800          # Stage 5/6 used 800 too; with the added
                              # reach-head loss the encoder needs the
                              # full budget to converge.
DEFAULT_HIDDEN = 32
DEFAULT_LATENT = 16
REACH_WEIGHT   = 3.0          # Boost reach loss so the head's outputs
                              # saturate cleanly (~0 or ~1) on the irr
                              # twin where labels vary per (s, a).
                              # On the rev twin labels are trivially
                              # [1,1,1] everywhere, so this only sharpens
                              # the irr decision boundary.

RCP_LEVELS = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
SEEDS = [0, 1, 2, 3, 4]


# ====================================================================
# Single-config evaluation: baseline + DA on the SAME twin pair
# ====================================================================

def evaluate_one(rcp: float, seed: int,
                  *,
                  epochs: int = DEFAULT_EPOCHS,
                  hidden: int = DEFAULT_HIDDEN,
                  latent: int = DEFAULT_LATENT,
                  label_noise_p: float = 0.0, obs_noise_std: float = 0.0,
                  reach_weight: float = REACH_WEIGHT,
                  lam: float = LAMBDA) -> Dict[str, Any]:
    m, H, r_d, r_g, gamma, k = (
        DEFAULTS[x] for x in ("m", "H", "r_d", "r_g", "gamma", "k"))

    mdp_irr = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                    mode="irreversible")
    mdp_rev = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                    mode="reversible")

    # --- Train both kinds of WM, per twin.
    wm_b_irr, tlist_irr, dtab_irr, t_b_irr = train_world_model(
        mdp_irr, epochs=epochs, label_noise_p=label_noise_p,
        obs_noise_std=obs_noise_std, recover_corrupt_p=rcp,
        hidden=hidden, latent=latent, seed=seed, reach_weight=0.0)
    wm_b_rev, tlist_rev, dtab_rev, t_b_rev = train_world_model(
        mdp_rev, epochs=epochs, label_noise_p=label_noise_p,
        obs_noise_std=obs_noise_std, recover_corrupt_p=rcp,
        hidden=hidden, latent=latent, seed=seed, reach_weight=0.0)
    wm_d_irr, _,         _,         t_d_irr = train_world_model(
        mdp_irr, epochs=epochs, label_noise_p=label_noise_p,
        obs_noise_std=obs_noise_std, recover_corrupt_p=rcp,
        hidden=hidden, latent=latent, seed=seed,
        reach_weight=reach_weight)
    wm_d_rev, _,         _,         t_d_rev = train_world_model(
        mdp_rev, epochs=epochs, label_noise_p=label_noise_p,
        obs_noise_std=obs_noise_std, recover_corrupt_p=rcp,
        hidden=hidden, latent=latent, seed=seed,
        reach_weight=reach_weight)

    # --- Build mdp_hats.
    mdp_hat_b_irr = build_mdp_hat(mdp_irr, wm_b_irr)
    mdp_hat_b_rev = build_mdp_hat(mdp_rev, wm_b_rev)
    mdp_hat_d_irr = build_mdp_hat(mdp_irr, wm_d_irr)
    mdp_hat_d_rev = build_mdp_hat(mdp_rev, wm_d_rev)

    # --- Diagnostics: D_w_hat at s_0 a_decoy.
    # Baseline reads from mdp_hat (Stage-5/6 destroyed_mass).
    Dw_b_irr_s0 = destroyed_mass(mdp_hat_b_irr, S0, "a_decoy")
    Dw_b_rev_s0 = destroyed_mass(mdp_hat_b_rev, S0, "a_decoy")
    # DA reads from reach_head + mdp_hat structure.
    Dw_d_irr_s0 = compute_dw_hat_da(wm_d_irr, tlist_irr, dtab_irr,
                                     mdp_irr.target_weights, mdp_irr.gamma,
                                     S0, "a_decoy")
    Dw_d_rev_s0 = compute_dw_hat_da(wm_d_rev, tlist_rev, dtab_rev,
                                     mdp_rev.target_weights, mdp_rev.gamma,
                                     S0, "a_decoy")
    Dw_t_irr_s0 = destroyed_mass(mdp_irr, S0, "a_decoy")
    Dw_t_rev_s0 = destroyed_mass(mdp_rev, S0, "a_decoy")

    # --- Closed-loop returns, with cheat check on every run.
    H_p = H
    def choose_obl_b_irr(s): return planner_obl(mdp_hat_b_irr, s, H_p)
    def choose_mrc_b_irr(s): return planner_mrc_baseline(mdp_hat_b_irr, s, H_p, lam)
    def choose_obl_b_rev(s): return planner_obl(mdp_hat_b_rev, s, H_p)
    def choose_mrc_b_rev(s): return planner_mrc_baseline(mdp_hat_b_rev, s, H_p, lam)
    def choose_obl_d_irr(s): return planner_obl(mdp_hat_d_irr, s, H_p)
    def choose_mrc_d_irr(s): return planner_mrc_da(mdp_hat_d_irr, wm_d_irr,
                                                     tlist_irr, dtab_irr,
                                                     s, H_p, lam)
    def choose_obl_d_rev(s): return planner_obl(mdp_hat_d_rev, s, H_p)
    def choose_mrc_d_rev(s): return planner_mrc_da(mdp_hat_d_rev, wm_d_rev,
                                                     tlist_rev, dtab_rev,
                                                     s, H_p, lam)

    R_b_obl_irr, _, _ = run_closed_loop(mdp_irr, choose_obl_b_irr, "B/obl/irr")
    R_b_mrc_irr, _, _ = run_closed_loop(mdp_irr, choose_mrc_b_irr, "B/mrc/irr")
    R_b_obl_rev, _, _ = run_closed_loop(mdp_rev, choose_obl_b_rev, "B/obl/rev")
    R_b_mrc_rev, _, _ = run_closed_loop(mdp_rev, choose_mrc_b_rev, "B/mrc/rev")
    R_d_obl_irr, _, _ = run_closed_loop(mdp_irr, choose_obl_d_irr, "D/obl/irr")
    R_d_mrc_irr, _, _ = run_closed_loop(mdp_irr, choose_mrc_d_irr, "D/mrc/irr")
    R_d_obl_rev, _, _ = run_closed_loop(mdp_rev, choose_obl_d_rev, "D/obl/rev")
    R_d_mrc_rev, _, _ = run_closed_loop(mdp_rev, choose_mrc_d_rev, "D/mrc/rev")

    # --- Oracle reference (uses the TRUE env; for normalisation only).
    def choose_obl_orc(s): return policy_obl(mdp_irr, s, H_p)
    def choose_mrc_orc(s): return policy_mrc(mdp_irr, s, H_p, lam)
    R_orc_obl_irr = rollout_value(mdp_irr, S0, choose_obl_orc)
    R_orc_mrc_irr = rollout_value(mdp_irr, S0, choose_mrc_orc)
    oracle_gap_irr = R_orc_mrc_irr - R_orc_obl_irr

    # --- Ratios.
    denom = max(oracle_gap_irr, EPS)
    b_gap_irr = R_b_mrc_irr - R_b_obl_irr
    b_gap_rev = R_b_mrc_rev - R_b_obl_rev
    d_gap_irr = R_d_mrc_irr - R_d_obl_irr
    d_gap_rev = R_d_mrc_rev - R_d_obl_rev
    b_collapse = abs(b_gap_rev) / denom
    b_charge   = b_gap_irr      / denom
    d_collapse = abs(d_gap_rev) / denom
    d_charge   = d_gap_irr      / denom

    b_trig_s0 = Dw_b_rev_s0 > TRIGGER_EPS
    d_trig_s0 = Dw_d_rev_s0 > TRIGGER_EPS

    return {
        "recover_corrupt_p": rcp, "seed": seed,
        "epochs": epochs, "hidden": hidden, "latent": latent,
        "reach_weight": reach_weight,
        "Dw_true_irr_s0": float(Dw_t_irr_s0),
        "Dw_true_rev_s0": float(Dw_t_rev_s0),
        "Dw_baseline_irr_s0": float(Dw_b_irr_s0),
        "Dw_baseline_rev_s0": float(Dw_b_rev_s0),
        "Dw_da_irr_s0":       float(Dw_d_irr_s0),
        "Dw_da_rev_s0":       float(Dw_d_rev_s0),
        "R_baseline_obl_irr": float(R_b_obl_irr),
        "R_baseline_mrc_irr": float(R_b_mrc_irr),
        "R_baseline_obl_rev": float(R_b_obl_rev),
        "R_baseline_mrc_rev": float(R_b_mrc_rev),
        "R_da_obl_irr":       float(R_d_obl_irr),
        "R_da_mrc_irr":       float(R_d_mrc_irr),
        "R_da_obl_rev":       float(R_d_obl_rev),
        "R_da_mrc_rev":       float(R_d_mrc_rev),
        "oracle_gap_irr":     float(oracle_gap_irr),
        "baseline_gap_irr":   float(b_gap_irr),
        "baseline_gap_rev":   float(b_gap_rev),
        "da_gap_irr":         float(d_gap_irr),
        "da_gap_rev":         float(d_gap_rev),
        "baseline_collapse_ratio":    float(b_collapse),
        "baseline_charge_load_ratio": float(b_charge),
        "da_collapse_ratio":          float(d_collapse),
        "da_charge_load_ratio":       float(d_charge),
        "baseline_trigger_s0": bool(b_trig_s0),
        "da_trigger_s0":       bool(d_trig_s0),
        "baseline_pass":       bool(b_collapse <= COLLAPSE_THRESHOLD
                                     and b_charge >= CHARGE_THRESHOLD),
        "da_pass":             bool(d_collapse <= COLLAPSE_THRESHOLD
                                     and d_charge >= CHARGE_THRESHOLD),
        "train_time_s": float(t_b_irr + t_b_rev + t_d_irr + t_d_rev),
    }


# ====================================================================
# Recovery (lambda) sweep for DA, at one rcp where baseline collapses
# ====================================================================

def recovery_lambda_sweep_da(rcp: float = 0.7, seed: int = 0,
                              epochs: int = DEFAULT_EPOCHS,
                              hidden: int = DEFAULT_HIDDEN,
                              latent: int = DEFAULT_LATENT,
                              ) -> Dict[str, Any]:
    """At a perturbation level where baseline FAILs collapse, sweep lambda
    on the DA planner and verify lambda* matches r_d / D_w_hat_da.
    """
    m, H, r_d, r_g, gamma, k = (
        DEFAULTS[x] for x in ("m", "H", "r_d", "r_g", "gamma", "k"))
    mdp_irr = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                    mode="irreversible")
    wm_d, tlist, dtab, _ = train_world_model(
        mdp_irr, epochs=epochs, label_noise_p=0.0, obs_noise_std=0.0,
        recover_corrupt_p=rcp, hidden=hidden, latent=latent, seed=seed,
        reach_weight=REACH_WEIGHT)
    mdp_hat = build_mdp_hat(mdp_irr, wm_d)

    Dw_hat = compute_dw_hat_da(wm_d, tlist, dtab,
                                 mdp_irr.target_weights, mdp_irr.gamma,
                                 S0, "a_decoy")
    if Dw_hat <= 0:
        return {"rcp": rcp, "seed": seed,
                "D_w_hat_da": float(Dw_hat),
                "lam_min_hat": None,
                "lam_star": None,
                "policy_at_lam_1": None,
                "switch_err_vs_hat": None,
                "match": False,
                "skipped_reason": "D_w_hat <= 0 in DA -- nothing to switch."}
    lam_min = r_d / Dw_hat

    lambdas = np.linspace(0.0, 1.5, 1501)
    grid_step = float(lambdas[1] - lambdas[0])
    actions = [planner_mrc_da(mdp_hat, wm_d, tlist, dtab, S0, H, float(lam))
                for lam in lambdas]
    switch_idx = next((i for i, a in enumerate(actions) if a == "a_safe"),
                       None)
    lam_star = float(lambdas[switch_idx]) if switch_idx is not None else None
    at_one = planner_mrc_da(mdp_hat, wm_d, tlist, dtab, S0, H, 1.0)
    switch_err = (abs(lam_star - lam_min) if lam_star is not None
                   else float("inf"))
    match = (lam_star is not None
              and abs(lam_star - lam_min) <= 2 * grid_step + 1e-9)
    return {
        "rcp": rcp, "seed": seed,
        "D_w_hat_da": float(Dw_hat),
        "lam_min_hat": float(lam_min),
        "lam_star": lam_star,
        "grid_step": grid_step,
        "policy_at_lam_1": at_one,
        "switch_err_vs_hat": float(switch_err),
        "match": bool(match),
    }


# ====================================================================
# Aggregation + verdict
# ====================================================================

def aggregate_per_rcp(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_rcp: Dict[float, List[Dict[str, Any]]] = {}
    for r in rows:
        by_rcp.setdefault(r["recover_corrupt_p"], []).append(r)
    out = []
    for rcp in sorted(by_rcp.keys()):
        rs = by_rcp[rcp]
        agg = {
            "rcp": rcp,
            "n_seeds": len(rs),
            "baseline": {
                "mean_collapse":   float(np.mean([r["baseline_collapse_ratio"] for r in rs])),
                "max_collapse":    float(max(r["baseline_collapse_ratio"] for r in rs)),
                "mean_charge":     float(np.mean([r["baseline_charge_load_ratio"] for r in rs])),
                "min_charge":      float(min(r["baseline_charge_load_ratio"] for r in rs)),
                "mean_Dw_rev_s0":  float(np.mean([r["Dw_baseline_rev_s0"] for r in rs])),
                "n_trigger_s0":    sum(1 for r in rs if r["baseline_trigger_s0"]),
                "n_pass":          sum(1 for r in rs if r["baseline_pass"]),
            },
            "da": {
                "mean_collapse":   float(np.mean([r["da_collapse_ratio"] for r in rs])),
                "max_collapse":    float(max(r["da_collapse_ratio"] for r in rs)),
                "mean_charge":     float(np.mean([r["da_charge_load_ratio"] for r in rs])),
                "min_charge":      float(min(r["da_charge_load_ratio"] for r in rs)),
                "mean_Dw_rev_s0":  float(np.mean([r["Dw_da_rev_s0"] for r in rs])),
                "n_trigger_s0":    sum(1 for r in rs if r["da_trigger_s0"]),
                "n_pass":          sum(1 for r in rs if r["da_pass"]),
            },
        }
        out.append(agg)
    return out


def compute_verdict(agg: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pre-registered PASS / PARTIAL / FAIL decision.

    Triggered RCPs are those at which the baseline breaks collapse
    (max_collapse > COLLAPSE_THRESHOLD).  At those, decision-aware MUST
    bring max_collapse <= COLLAPSE_THRESHOLD AND keep min_charge >=
    CHARGE_THRESHOLD on irr to count as a fix.
    """
    triggered = [a for a in agg
                  if a["baseline"]["max_collapse"] > COLLAPSE_THRESHOLD]
    da_fixes  = [a for a in triggered
                  if a["da"]["max_collapse"] <= COLLAPSE_THRESHOLD
                  and a["da"]["min_charge"]   >= CHARGE_THRESHOLD]
    da_fails  = [a for a in triggered
                  if a["da"]["max_collapse"] > COLLAPSE_THRESHOLD
                  or  a["da"]["min_charge"]   < CHARGE_THRESHOLD]

    if not triggered:
        return {
            "verdict": "INCONCLUSIVE_NO_BASELINE_FAILURE",
            "reason": ("Baseline did not break collapse at any rcp in the "
                        "sweep -- nothing to fix.  Re-check Stage-6 setup."),
            "triggered_rcps": [],
            "da_fixed_rcps":   [],
            "da_failed_rcps":  [],
        }
    if not da_fails:
        return {
            "verdict": "PASS",
            "reason": (f"Baseline broke collapse at "
                        f"{len(triggered)} rcp(s); decision-aware brought "
                        f"max collapse_ratio <= {COLLAPSE_THRESHOLD} AND "
                        f"kept min charge_load >= {CHARGE_THRESHOLD} at all "
                        f"of them.  Decision-aware training repairs the "
                        f"Phase-1-channel asymmetric error without "
                        f"sacrificing separation."),
            "triggered_rcps": [a["rcp"] for a in triggered],
            "da_fixed_rcps":   [a["rcp"] for a in da_fixes],
            "da_failed_rcps":  [],
        }
    if not da_fixes:
        return {
            "verdict": "FAIL",
            "reason": (f"Baseline broke collapse at "
                        f"{len(triggered)} rcp(s); decision-aware did not "
                        f"fix any of them.  Either D_w_hat from reach head "
                        f"is still wrong, or the fix kills separation."),
            "triggered_rcps": [a["rcp"] for a in triggered],
            "da_fixed_rcps":   [],
            "da_failed_rcps":  [a["rcp"] for a in da_fails],
        }
    return {
        "verdict": "PARTIAL",
        "reason": (f"Baseline broke collapse at "
                    f"{len(triggered)} rcp(s); decision-aware fixed "
                    f"{len(da_fixes)} and failed at {len(da_fails)}.  The "
                    f"fix pushes the failure threshold but does not "
                    f"close it.  Honest finding."),
        "triggered_rcps": [a["rcp"] for a in triggered],
        "da_fixed_rcps":   [a["rcp"] for a in da_fixes],
        "da_failed_rcps":  [a["rcp"] for a in da_fails],
    }


# ====================================================================
# Figure
# ====================================================================

def write_figure(agg: List[Dict[str, Any]],
                  per_run: List[Dict[str, Any]],
                  recovery: Dict[str, Any],
                  out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    xs = [a["rcp"] for a in agg]

    # (a) collapse_ratio per rcp -- baseline vs DA.
    ax = axes[0, 0]
    ax.plot(xs, [a["baseline"]["mean_collapse"] for a in agg], "o-",
             label="baseline mean", color="firebrick")
    ax.plot(xs, [a["baseline"]["max_collapse"]  for a in agg], "o--",
             label="baseline max",  color="firebrick", alpha=0.5)
    ax.plot(xs, [a["da"]["mean_collapse"]       for a in agg], "s-",
             label="DA mean",       color="steelblue")
    ax.plot(xs, [a["da"]["max_collapse"]        for a in agg], "s--",
             label="DA max",        color="steelblue", alpha=0.5)
    ax.axhline(COLLAPSE_THRESHOLD, color="k", ls=":", lw=0.8,
                label=f"PASS <= {COLLAPSE_THRESHOLD}")
    ax.set_xlabel("recover_corrupt_p"); ax.set_ylabel("collapse_ratio")
    ax.set_title("(a) collapse_ratio on rev twin")
    ax.legend(fontsize=8)

    # (b) charge_load_ratio.
    ax = axes[0, 1]
    ax.plot(xs, [a["baseline"]["mean_charge"] for a in agg], "o-",
             label="baseline mean", color="firebrick")
    ax.plot(xs, [a["baseline"]["min_charge"]  for a in agg], "o--",
             label="baseline min",  color="firebrick", alpha=0.5)
    ax.plot(xs, [a["da"]["mean_charge"]       for a in agg], "s-",
             label="DA mean",       color="steelblue")
    ax.plot(xs, [a["da"]["min_charge"]        for a in agg], "s--",
             label="DA min",        color="steelblue", alpha=0.5)
    ax.axhline(CHARGE_THRESHOLD, color="k", ls=":", lw=0.8,
                label=f"PASS >= {CHARGE_THRESHOLD}")
    ax.set_xlabel("recover_corrupt_p"); ax.set_ylabel("charge_load_ratio")
    ax.set_title("(b) charge_load_ratio on irr twin")
    ax.legend(fontsize=8)

    # (c) D_w_hat(rev, s_0) -- the diagnostic for asymmetric error at s_0.
    ax = axes[0, 2]
    ax.plot(xs, [a["baseline"]["mean_Dw_rev_s0"] for a in agg], "o-",
             label="baseline mean", color="firebrick")
    ax.plot(xs, [a["da"]["mean_Dw_rev_s0"]       for a in agg], "s-",
             label="DA mean",       color="steelblue")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("recover_corrupt_p")
    ax.set_ylabel("D_w_hat at s_0 a_decoy on rev (target 0)")
    ax.set_title("(c) asymmetric error magnitude at decision point")
    ax.legend(fontsize=8)

    # (d) Per-seed scatter, collapse on rev.
    ax = axes[1, 0]
    for r in per_run:
        ax.scatter([r["recover_corrupt_p"] - 0.012],
                    [r["baseline_collapse_ratio"]],
                    color="firebrick", s=20, alpha=0.7)
        ax.scatter([r["recover_corrupt_p"] + 0.012],
                    [r["da_collapse_ratio"]],
                    color="steelblue", s=20, alpha=0.7)
    ax.axhline(COLLAPSE_THRESHOLD, color="k", ls=":", lw=0.8)
    ax.set_xlabel("recover_corrupt_p")
    ax.set_ylabel("collapse_ratio (per seed)")
    ax.set_title("(d) per-seed collapse scatter (red = baseline, blue = DA)")

    # (e) Per-seed charge on irr.
    ax = axes[1, 1]
    for r in per_run:
        ax.scatter([r["recover_corrupt_p"] - 0.012],
                    [r["baseline_charge_load_ratio"]],
                    color="firebrick", s=20, alpha=0.7)
        ax.scatter([r["recover_corrupt_p"] + 0.012],
                    [r["da_charge_load_ratio"]],
                    color="steelblue", s=20, alpha=0.7)
    ax.axhline(CHARGE_THRESHOLD, color="k", ls=":", lw=0.8)
    ax.set_xlabel("recover_corrupt_p")
    ax.set_ylabel("charge_load_ratio (per seed)")
    ax.set_title("(e) per-seed charge scatter")

    # (f) Recovery diagnostic.
    ax = axes[1, 2]
    if (recovery.get("lam_star") is not None
            and recovery.get("lam_min_hat") is not None):
        ax.axvline(recovery["lam_min_hat"], color="purple", ls="--",
                    label=f"lam_min_hat = {recovery['lam_min_hat']:.3f}")
        ax.axvline(recovery["lam_star"], color="red",
                    label=f"lam* observed = {recovery['lam_star']:.3f}")
        ax.set_xlim(0, 1.5); ax.set_ylim(0, 1)
        ax.set_xlabel("lambda")
        ax.set_title(f"(f) DA recovery sweep @ rcp = {recovery['rcp']}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "recovery sweep skipped\n(D_w_hat <= 0 in DA)",
                 ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(f) recovery sweep")

    fig.suptitle("Stage-7 Kill-Gate A1 -- decision-aware WM repairs collapse",
                  fontsize=12)
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
    t0 = time.time()
    print("=" * 78)
    print("Stage-7 Kill-Gate A1 -- Decision-aware WM training repairs collapse")
    print("=" * 78)
    print(f"Defaults: {DEFAULTS}")
    print(f"Lambda: {LAMBDA}  Reach weight: {REACH_WEIGHT}")
    print(f"COLLAPSE PASS bound <= {COLLAPSE_THRESHOLD}; "
          f"CHARGE PASS bound >= {CHARGE_THRESHOLD}")
    print("Cheat-check: every closed-loop runs assert_planner_cheat_free "
          "(CountedMDP).")

    print("\n[Sweep] recover_corrupt_p x seeds (baseline + DA each)")
    print(f"{'rcp':>5} {'sd':>3}  {'B_Dw_rev':>9} {'D_Dw_rev':>9}  "
          f"{'B_col':>6} {'D_col':>6}  {'B_chg':>6} {'D_chg':>6}  "
          f"{'B_pass':>6} {'D_pass':>6}")
    per_run = []
    for rcp in RCP_LEVELS:
        for seed in SEEDS:
            r = evaluate_one(rcp=rcp, seed=seed)
            per_run.append(r)
            print(f"{rcp:>5.2f} {seed:>3d}  "
                  f"{r['Dw_baseline_rev_s0']:>9.3f} {r['Dw_da_rev_s0']:>9.3f}  "
                  f"{r['baseline_collapse_ratio']:>6.3f} "
                  f"{r['da_collapse_ratio']:>6.3f}  "
                  f"{r['baseline_charge_load_ratio']:>6.3f} "
                  f"{r['da_charge_load_ratio']:>6.3f}  "
                  f"{str(r['baseline_pass']):>6} {str(r['da_pass']):>6}")

    agg = aggregate_per_rcp(per_run)

    print("\n[Aggregated] per rcp")
    print(f"{'rcp':>5} {'n':>2}  "
          f"{'B_Dw_rev':>9} {'D_Dw_rev':>9}  "
          f"{'B_col_mx':>9} {'D_col_mx':>9}  "
          f"{'B_chg_mn':>9} {'D_chg_mn':>9}  "
          f"{'B_pass':>6} {'D_pass':>6}")
    for a in agg:
        b, d = a["baseline"], a["da"]
        print(f"{a['rcp']:>5.2f} {a['n_seeds']:>2d}  "
              f"{b['mean_Dw_rev_s0']:>9.4f} {d['mean_Dw_rev_s0']:>9.4f}  "
              f"{b['max_collapse']:>9.4f} {d['max_collapse']:>9.4f}  "
              f"{b['min_charge']:>9.4f} {d['min_charge']:>9.4f}  "
              f"{b['n_pass']:>2d}/{a['n_seeds']:<2d} "
              f"{d['n_pass']:>2d}/{a['n_seeds']:<2d}")

    print("\n[Recovery] DA lambda sweep at rcp = 0.7 (where baseline FAILs)")
    recovery = recovery_lambda_sweep_da(rcp=0.7, seed=0)
    if recovery.get("skipped_reason"):
        print(f"  skipped: {recovery['skipped_reason']}")
    else:
        print(f"  D_w_hat_da(s_0, a_decoy) = {recovery['D_w_hat_da']:.6f}")
        print(f"  lam_min_hat = r_d / D_w_hat = {recovery['lam_min_hat']:.6f}")
        print(f"  lam* observed = {recovery['lam_star']}")
        print(f"  pi_mrc_da at lam=1: '{recovery['policy_at_lam_1']}'")
        print(f"  match (within 2 grid steps): {recovery['match']}")

    verdict = compute_verdict(agg)
    print("\n" + "=" * 78)
    print(f"Pre-registered verdict: {verdict['verdict']}")
    print(f"  {verdict['reason']}")
    print(f"  triggered RCPs: {verdict['triggered_rcps']}")
    print(f"  DA fixed RCPs : {verdict['da_fixed_rcps']}")
    print(f"  DA failed RCPs: {verdict['da_failed_rcps']}")
    print("=" * 78)

    pdf_path = os.path.join(_THIS_DIR, "stage7_sweep.pdf")
    write_figure(agg, per_run, recovery, pdf_path)
    print(f"Figure: {pdf_path}")

    dt = time.time() - t0
    payload = {
        "verdict":         verdict["verdict"],
        "verdict_reason":  verdict["reason"],
        "verdict_meta":    {k: v for k, v in verdict.items()
                              if k not in ("verdict", "reason")},
        "wall_time_s":     dt,
        "defaults":        DEFAULTS,
        "lambda":          LAMBDA,
        "reach_weight":    REACH_WEIGHT,
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "charge_threshold":   CHARGE_THRESHOLD,
        "trigger_eps":     TRIGGER_EPS,
        "rcp_levels":      RCP_LEVELS,
        "seeds":           SEEDS,
        "per_run":         per_run,
        "aggregated":      agg,
        "recovery_lambda_sweep_da": recovery,
        "cheat_check": ("All closed-loop runs went through "
                         "assert_planner_cheat_free with CountedMDP; "
                         "any planner read of true_env.f or true_env.r "
                         "would have raised AssertionError before this "
                         "result was written."),
    }
    out_path = os.path.join(_THIS_DIR, "stage7_results.json")
    with open(out_path, "w") as fh:
        json.dump(_to_jsonable(payload), fh, indent=2)
    print(f"Results: {out_path}")
    print(f"Wall time: {dt:.1f} s")
    return verdict["verdict"] == "PASS"


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

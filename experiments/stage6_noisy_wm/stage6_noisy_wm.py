"""
experiments/stage6_noisy_wm/stage6_noisy_wm.py
================================================

Stage-6 Kill-Gate 3 -- collapse robustness under noisy / imperfect WM.

WHY this stage exists (READ FIRST)
----------------------------------
  Stage-5 Kill-Gate 2 PASSed, but with a critical caveat: on this tiny
  10-state twin the learned WM achieved 100% transition accuracy and
  D_w_hat error was exactly zero across all sweeps.  So Stage-5 verified
  "learned-but-accurate D_w_hat preserves collapse" -- it did NOT touch
  Phase-1's real failure channel.

  Phase-1's real failure mode is an ASYMMETRIC error:
    on the REVERSIBLE twin, the WM mistakenly predicts that the recover
    transition does NOT preserve goal reachability -> D_w_hat(rev) > 0
    where D_w_true(rev) = 0 -> mrc charges incorrectly on rev -> mrc
    picks a different action than reward_only on rev -> reversible-twin
    collapse breaks.

  Stage-5 never observed this because its training was clean and its
  WM trivially learned everything.  Stage-6 ACTIVELY forces the WM to
  make this asymmetric error and asks the only question that matters:
  does the per-step MPPI-style D_w_hat-charge design (the structural
  anti-Phase-1 difference (2) inherited from Stage-5) absorb that
  error gracefully, or does it amplify it into a Phase-1 redux?

Knobs we use to force the WM to err (axes of the severity sweep)
----------------------------------------------------------------
  (1) Label noise         : per-epoch, replace the next-state target
                            with a uniform-random state with probability
                            label_noise_p.  Simulates noisy / stochastic
                            transition data, which is what TD-MPC2 sees
                            in practice.
  (2) Observation noise   : per-epoch, add iso-Gaussian noise of std
                            obs_noise_std to phi(s) at the encoder input.
                            Simulates a sensor that doesn't cleanly
                            separate cell types.
  (3) Training budget     : `epochs`.  Fewer epochs = more underfit WM.
  (4) WM capacity         : hidden width + latent dim.  Smaller = less
                            ability to separate (lava, absorb, corridor).

  We run a SEVERITY sweep that combines all four knobs monotonically
  (L0_clean ... L6_devastate), and then a SINGLE-KNOB ablation per knob
  to identify which one matters most.  Multi-seed (5 seeds) per cell.

Validity check (pre-registered; THIS STAGE IS INVALID WITHOUT IT)
------------------------------------------------------------------
  The whole point of Stage-6 is to probe collapse under asymmetric error.
  If the sweep never produces D_w_hat(rev) != 0, we have NOT probed the
  failure channel and cannot conclude anything.  So we explicitly verify:

      "Triggered" iff D_w_hat(s_0, a_decoy) on the reversible twin
      is > 1e-9 in at least some configurations.

  If no config triggers, the run is INCONCLUSIVE and we escalate
  (or report the limitation).  This is NOT a pass; it is a "not tested".

Pre-registered PASS / FAIL of Kill-Gate 3
------------------------------------------
  Among configurations where the asymmetric error IS triggered:
    PASS iff collapse_ratio remains <= COLLAPSE_THRESHOLD (= 0.30) on
         the reversible twin.  Equivalently: the MPPI per-step argmax
         absorbs the D_w_hat(rev) > 0 mistake without flipping its
         action choice on s_0 (so trajectory and return stay identical
         to reward_only on rev twin).
    FAIL iff at any triggered config, mean collapse_ratio over seeds
         exceeds COLLAPSE_THRESHOLD.  This is the Phase-1 mode reborn
         in the learned-WM + MPPI setting.

  FAIL is a fully acceptable, honest outcome.  The instructions
  explicitly forbid rescuing FAIL by adding training, by relaxing the
  threshold, or by truncating the sweep to drop triggered-and-broken
  configs.

Reuse
-----
  Imports from Stage-4 (build_lava_gridworld, S0) and Stage-5
  (WorldModel, phi, act_oh, build_mdp_hat, run_planner, defaults).
  Stage-1's destroyed_mass is invoked verbatim via Stage-5's import
  chain.  Asserted on load.

Runtime / cost
--------------
  CPU only; PyTorch tiny model on a ~10-state graph.
  Severity sweep:    7 levels x 5 seeds x 2 twins = 70 trainings.
  Per-knob ablation: 3 knobs x ~6 levels x 5 seeds x 2 twins ~ 180.
  ~250 trainings, each <= 1s for the small WMs we use.
  Expected wall time:  3-6 minutes on CPU.  No GPU.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------
# Imports from Stage-4 / Stage-5 (Stage-1 reached transitively).
# --------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE1_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage1_unified"))
_STAGE4_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage4_modelbased"))
_STAGE5_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "stage5_learned_wm"))
for _p in (_STAGE1_DIR, _STAGE4_DIR, _STAGE5_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1_unified_validation import (  # noqa: E402
    MDP, destroyed_mass, policy_obl, policy_mrc, rollout_value,
)
from stage4_modelbased_planning import build_lava_gridworld, S0, LAVA  # noqa: E402
from stage5_learned_wm import (  # noqa: E402
    WorldModel, phi, act_oh, collect_transitions,
    build_mdp_hat, run_planner, dw_hat_vs_exact, LATENT_DIM,
)

assert destroyed_mass.__module__ == "stage1_unified_validation"
assert policy_mrc.__module__ == "stage1_unified_validation"


# ====================================================================
# Perturbed training -- the central new function of this stage
# ====================================================================

def train_world_model_perturbed(
    mdp: MDP, epochs: int, label_noise_p: float, obs_noise_std: float,
    hidden: int, latent: int, seed: int, lr: float = 1e-3,
    recover_corrupt_p: float = 0.0,
) -> Tuple[WorldModel, float]:
    """Train the Stage-5 WorldModel with per-epoch perturbations.

    label_noise_p     : probability that a transition's target next-state
                        label is replaced (per epoch) by a uniform-random
                        state.  Models noisy / stochastic transition data.
    obs_noise_std     : iso-Gaussian noise std added (per epoch) to phi(s)
                        at the encoder input.  Models a sensor that cannot
                        cleanly separate cell types.
    hidden, latent    : WM capacity knobs.  Smaller = less capacity to
                        separate (lava, absorb, corridor) in latent space.
    recover_corrupt_p : SURGICAL knob.  Per-epoch, with this probability,
                        the (lava, recover) transition's target is
                        replaced specifically by "absorb" (no-op on
                        irreversible twin which has no such transition).
                        Simulates the natural Phase-1 failure trigger:
                        in a stochastic env, recovery actions occasionally
                        fail and land the agent in an absorbing state ->
                        the WM trained on such data may predict that
                        lava->recover destroys reachability, planting an
                        asymmetric error directly at the s_0 decision
                        point on the reversible twin.  Uniform label noise
                        rarely produces this specific failure because the
                        random replacement averages out; this targeted
                        knob is how we make the kill-gate non-trivial.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    wm = WorldModel(latent=latent, hidden=hidden)
    opt = torch.optim.Adam(wm.parameters(), lr=lr)

    transitions = collect_transitions(mdp)
    n = len(transitions)
    obs_s_clean = torch.stack([phi(t[0]) for t in transitions])
    act_a = torch.stack([act_oh(t[1]) for t in transitions])
    rewards = torch.tensor([t[3] for t in transitions], dtype=torch.float32)
    state_list = list(mdp.states)
    n_states = len(state_list)
    true_s_next = [t[2] for t in transitions]

    t0 = time.time()
    for epoch in range(epochs):
        # Per-epoch noisy next-state targets.  Start from the true list,
        # then apply (a) uniform label noise and (b) recover-specific
        # corruption.  Both knobs can be active simultaneously.
        s_next_list = list(true_s_next)
        if label_noise_p > 0:
            for i in range(n):
                if rng.random() < label_noise_p:
                    s_next_list[i] = state_list[int(rng.integers(0, n_states))]
        if recover_corrupt_p > 0:
            for i, (s_i, a_i, _, _) in enumerate(transitions):
                if s_i == LAVA and a_i == "recover":
                    if rng.random() < recover_corrupt_p:
                        s_next_list[i] = "absorb"
        obs_s_next = torch.stack([phi(s) for s in s_next_list])
        # Per-epoch obs noise on the encoder input.
        if obs_noise_std > 0:
            obs_s = obs_s_clean + torch.randn_like(obs_s_clean) * obs_noise_std
        else:
            obs_s = obs_s_clean

        z = wm.encode(obs_s)
        z_next_pred, r_pred = wm.predict(z, act_a)
        z_next_target = wm.encode(obs_s_next).detach()
        loss_dyn = F.mse_loss(z_next_pred, z_next_target)
        loss_rew = F.mse_loss(r_pred, rewards)
        loss = loss_dyn + loss_rew
        opt.zero_grad()
        loss.backward()
        opt.step()
    return wm, time.time() - t0


# ====================================================================
# Shared constants
# ====================================================================

DEFAULTS = dict(m=4, H=4, r_d=1.0, r_g=1.0, gamma=0.9, k=3)
LAMBDA = 1.0
EPS = 1e-9
COLLAPSE_THRESHOLD = 0.30
CHARGE_THRESHOLD   = 0.50
TRIGGER_EPS        = 1e-9      # D_w_hat(rev) > this counts as triggered.


# ====================================================================
# Single-config evaluation (returns one row of the sweep)
# ====================================================================

def evaluate_config(
    name: str, label_noise_p: float, obs_noise_std: float,
    epochs: int, hidden: int, latent: int, seed: int,
    recover_corrupt_p: float = 0.0,
) -> Dict[str, Any]:
    """Train WMs on both twins under the given perturbation, evaluate
    all four planners (learned reward_only / learned mrc / oracle
    reward_only / oracle mrc) on both twins, and return one sweep row.
    """
    m, H, r_d, r_g, gamma, k = (
        DEFAULTS[x] for x in ("m", "H", "r_d", "r_g", "gamma", "k"))
    lam = LAMBDA

    mdp_irr = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                    mode="irreversible")
    mdp_rev = build_lava_gridworld(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma,
                                    mode="reversible")

    wm_irr, t_irr = train_world_model_perturbed(
        mdp_irr, epochs=epochs, label_noise_p=label_noise_p,
        obs_noise_std=obs_noise_std, hidden=hidden, latent=latent, seed=seed,
        recover_corrupt_p=recover_corrupt_p)
    wm_rev, t_rev = train_world_model_perturbed(
        mdp_rev, epochs=epochs, label_noise_p=label_noise_p,
        obs_noise_std=obs_noise_std, hidden=hidden, latent=latent, seed=seed,
        recover_corrupt_p=recover_corrupt_p)

    mdp_irr_hat, diag_irr = build_mdp_hat(mdp_irr, wm_irr)
    mdp_rev_hat, diag_rev = build_mdp_hat(mdp_rev, wm_rev)

    Dw_hat_irr_s0  = destroyed_mass(mdp_irr_hat, S0, "a_decoy")
    Dw_hat_rev_s0  = destroyed_mass(mdp_rev_hat, S0, "a_decoy")
    Dw_true_irr_s0 = destroyed_mass(mdp_irr,     S0, "a_decoy")
    Dw_true_rev_s0 = destroyed_mass(mdp_rev,     S0, "a_decoy")

    # Full D_w tables -- needed for the widened trigger criterion
    # ("any asymmetric error anywhere on the reversible twin").
    dw_table_irr = dw_hat_vs_exact(mdp_irr, mdp_irr_hat)
    dw_table_rev = dw_hat_vs_exact(mdp_rev, mdp_rev_hat)
    max_Dw_hat_rev_anywhere = max(row["D_w_hat"] for row in dw_table_rev)
    max_Dw_err_rev_anywhere = max(row["abs_err"] for row in dw_table_rev)
    max_Dw_err_irr_anywhere = max(row["abs_err"] for row in dw_table_irr)

    # Closed-loop returns on TRUE env, planning over learned or oracle MDP.
    R_obl_irr_lrn = run_planner(mdp_irr, mdp_irr_hat, H, "reward_only", lam)
    R_mrc_irr_lrn = run_planner(mdp_irr, mdp_irr_hat, H, "mrc",         lam)
    R_obl_rev_lrn = run_planner(mdp_rev, mdp_rev_hat, H, "reward_only", lam)
    R_mrc_rev_lrn = run_planner(mdp_rev, mdp_rev_hat, H, "mrc",         lam)
    R_obl_irr_orc = run_planner(mdp_irr, mdp_irr,     H, "reward_only", lam)
    R_mrc_irr_orc = run_planner(mdp_irr, mdp_irr,     H, "mrc",         lam)
    R_obl_rev_orc = run_planner(mdp_rev, mdp_rev,     H, "reward_only", lam)
    R_mrc_rev_orc = run_planner(mdp_rev, mdp_rev,     H, "mrc",         lam)

    learned_gap_irr = R_mrc_irr_lrn - R_obl_irr_lrn
    learned_gap_rev = R_mrc_rev_lrn - R_obl_rev_lrn
    oracle_gap_irr  = R_mrc_irr_orc - R_obl_irr_orc
    oracle_gap_rev  = R_mrc_rev_orc - R_obl_rev_orc

    denom = max(oracle_gap_irr, EPS)
    collapse_ratio    = abs(learned_gap_rev) / denom
    charge_load_ratio = learned_gap_irr      / denom

    collapse_ok = (collapse_ratio    <= COLLAPSE_THRESHOLD)
    charge_ok   = (charge_load_ratio >= CHARGE_THRESHOLD)
    passed = bool(collapse_ok and charge_ok)

    # Validity / trigger criterion (pre-registered).
    #
    # The user spec says the kill-gate is invalid unless we produce
    # "rev twin D_w_hat != 0" somewhere in the sweep.  In our gridworld
    # the only state where the planner makes a non-trivial decision is
    # s_0; D_w_hat at intermediate states never changes the trajectory
    # (every intermediate state has a single available action).  So we
    # report BOTH:
    #   - asymmetric_triggered_global  : max rev D_w_hat over ALL (s, a)
    #                                     > eps.  Matches the user's
    #                                     phrasing exactly.
    #   - asymmetric_triggered_at_s0   : the SUBSET that can change the
    #                                     decision (and therefore the
    #                                     trajectory and collapse_ratio).
    asymmetric_triggered_global = (max_Dw_hat_rev_anywhere > TRIGGER_EPS)
    asymmetric_triggered_at_s0  = (Dw_hat_rev_s0          > TRIGGER_EPS)
    # Action chosen by mrc at s_0 on the reversible twin (diagnostic).
    a_mrc_rev = policy_mrc(mdp_rev_hat, S0, H, lam)
    a_obl_rev = policy_obl(mdp_rev_hat, S0, H)

    return {
        "name": name,
        "label_noise_p": label_noise_p, "obs_noise_std": obs_noise_std,
        "recover_corrupt_p": recover_corrupt_p,
        "epochs": epochs, "hidden": hidden, "latent": latent, "seed": seed,
        "wm_tr_acc_irr": diag_irr["transition_accuracy"],
        "wm_tr_acc_rev": diag_rev["transition_accuracy"],
        "Dw_true_irr_s0": float(Dw_true_irr_s0),
        "Dw_hat_irr_s0":  float(Dw_hat_irr_s0),
        "Dw_err_irr_s0":  float(abs(Dw_true_irr_s0 - Dw_hat_irr_s0)),
        "Dw_true_rev_s0": float(Dw_true_rev_s0),
        "Dw_hat_rev_s0":  float(Dw_hat_rev_s0),
        "Dw_err_rev_s0":  float(abs(Dw_true_rev_s0 - Dw_hat_rev_s0)),
        "max_Dw_hat_rev_anywhere": float(max_Dw_hat_rev_anywhere),
        "max_Dw_err_rev_anywhere": float(max_Dw_err_rev_anywhere),
        "max_Dw_err_irr_anywhere": float(max_Dw_err_irr_anywhere),
        "R_obl_irr_learned": float(R_obl_irr_lrn),
        "R_mrc_irr_learned": float(R_mrc_irr_lrn),
        "R_obl_rev_learned": float(R_obl_rev_lrn),
        "R_mrc_rev_learned": float(R_mrc_rev_lrn),
        "R_obl_irr_oracle":  float(R_obl_irr_orc),
        "R_mrc_irr_oracle":  float(R_mrc_irr_orc),
        "R_obl_rev_oracle":  float(R_obl_rev_orc),
        "R_mrc_rev_oracle":  float(R_mrc_rev_orc),
        "learned_gap_irr":   float(learned_gap_irr),
        "learned_gap_rev":   float(learned_gap_rev),
        "oracle_gap_irr":    float(oracle_gap_irr),
        "oracle_gap_rev":    float(oracle_gap_rev),
        "collapse_ratio":    float(collapse_ratio),
        "charge_load_ratio": float(charge_load_ratio),
        "collapse_ok": collapse_ok, "charge_ok": charge_ok,
        "passed":              passed,
        # Backwards-compat field: now means "global" trigger (any rev (s,a)).
        "asymmetric_triggered":        bool(asymmetric_triggered_global),
        "asymmetric_triggered_global": bool(asymmetric_triggered_global),
        "asymmetric_triggered_at_s0":  bool(asymmetric_triggered_at_s0),
        "a_mrc_rev_s0":  a_mrc_rev,
        "a_obl_rev_s0":  a_obl_rev,
        "train_time_irr_s": float(t_irr),
        "train_time_rev_s": float(t_rev),
    }


# ====================================================================
# Severity sweep -- escalating combined perturbation
# ====================================================================

# Format: (name, label_noise_p, obs_noise_std, epochs, hidden, latent,
#          recover_corrupt_p)
# IMPORTANT design lessons (from two earlier attempts):
#  (i)  With tiny WM (hidden <= 8, latent <= 4) under heavy uniform
#       perturbation, all learned latents collapse to ~one point in
#       latent space, so nearest-neighbour decoding routes every (s, a)
#       to one "default" state and D_w_hat goes uniformly to 0 --
#       including on the reversible twin.  Such configs fail to probe
#       the asymmetric-error channel: the WM is dead, not asymmetrically
#       wrong.  We keep one such row in the grid (L_dead) as a control,
#       so the reader sees what total-failure looks like.
# (ii)  Uniform label noise + partial training yields D_w_hat error
#       SOMEWHERE on rev (max-over-(s,a) > 0) but NOT at s_0 specifically,
#       because the random replacement averages out instead of biasing
#       toward absorb.  In this gridworld the only state with a multi-
#       action choice is s_0; intermediate D_w_hat errors do NOT affect
#       trajectory or collapse_ratio.  So uniform label noise probes
#       the broad asymmetric-error definition but never the decision-
#       affecting subset.  This is true and we report both: the global
#       and the s_0-specific trigger.
# (iii) To probe the decision-affecting subset (rev D_w_hat > 0 AT s_0)
#       we add a SURGICAL knob recover_corrupt_p that corrupts the
#       (lava, recover) target to "absorb" with that probability per
#       epoch.  This is a natural and disclosed bias -- it simulates an
#       env where the recovery action stochastically lands in an
#       absorbing state -- and is documented explicitly in
#       train_world_model_perturbed's docstring.  The severity grid
#       below sweeps it.
SEVERITY_LEVELS: List[Tuple[str, float, float, int, int, int, float]] = [
    # Baseline (zero perturbation, full training, full capacity).
    ("L0_clean",                    0.00, 0.00, 800, 32, 16, 0.00),
    # Uniform-noise rows: probe the GLOBAL (any-(s,a)) trigger.
    ("L1_undertrain",               0.00, 0.00, 200, 32, 16, 0.00),
    ("L2_label_noise",              0.20, 0.00, 200, 32, 16, 0.00),
    ("L3_more_noise",               0.40, 0.00, 200, 32, 16, 0.00),
    ("L4_high_noise",               0.70, 0.50, 150, 32, 16, 0.00),
    # Surgical recover_corrupt rows: probe the s_0-specific trigger.
    ("L5_recover_lite",             0.00, 0.00, 200, 32, 16, 0.10),
    ("L6_recover_mid",              0.00, 0.00, 200, 32, 16, 0.30),
    ("L7_recover_strong",           0.00, 0.00, 200, 32, 16, 0.60),
    ("L8_recover_total",            0.00, 0.00, 200, 32, 16, 1.00),
    # Surgical + undertraining (compound).
    ("L9_recover_x_undertr",        0.00, 0.00, 100, 32, 16, 0.60),
    # Control: WM too small to learn anything (asymmetric error neither
    # global nor at s_0 -- documented dead config).
    ("L_dead_tiny_wm",              0.30, 0.50, 100,  8,  4, 0.00),
]

# Single-knob ablations -- base at default arch (clean) so single-knob
# moves stay in the partial-learning regime.
ABL_BASE = dict(label_noise_p=0.0, obs_noise_std=0.0, epochs=200,
                hidden=32, latent=16, recover_corrupt_p=0.0)
ABL_LABEL_NOISE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
ABL_OBS_NOISE   = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
ABL_EPOCHS      = [50, 100, 150, 200, 300, 400, 600, 800]
ABL_CAPACITY    = [(4, 4), (8, 4), (8, 8), (16, 8), (32, 16)]   # (hidden, latent)
# Surgical recovery-corruption sweep at default arch (the main
# decision-affecting probe).
ABL_RECOVER     = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

SEEDS = [0, 1, 2, 3, 4]


def _eval_cfg(name: str, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return evaluate_config(
        name=name,
        label_noise_p=cfg["label_noise_p"],
        obs_noise_std=cfg["obs_noise_std"],
        epochs=cfg["epochs"],
        hidden=cfg["hidden"],
        latent=cfg["latent"],
        seed=seed,
        recover_corrupt_p=cfg["recover_corrupt_p"],
    )


def run_severity_sweep() -> List[Dict[str, Any]]:
    print(f"\n[Severity] {len(SEVERITY_LEVELS)} levels x {len(SEEDS)} seeds "
          "x 2 twins")
    print(f"{'name':>22} {'lnp':>4} {'ons':>4} {'rcp':>4} {'ep':>4} "
          f"{'h':>3} {'l':>3} {'sd':>2}  "
          f"{'trA_i':>5} {'trA_r':>5}  {'Dwh_i':>6} {'Dwh_r':>6} "
          f"{'rev*':>6}  {'col':>6} {'chg':>6}  {'asG':>3} {'asS':>3} "
          f"{'pass':>4}")
    rows = []
    for tup in SEVERITY_LEVELS:
        name, lnp, ons, ep, hid, lat, rcp = tup
        cfg = dict(label_noise_p=lnp, obs_noise_std=ons, epochs=ep,
                   hidden=hid, latent=lat, recover_corrupt_p=rcp)
        for seed in SEEDS:
            r = _eval_cfg(name, cfg, seed)
            rows.append(r)
            print(f"{name:>22} {lnp:>4.2f} {ons:>4.2f} {rcp:>4.2f} "
                  f"{ep:>4d} {hid:>3d} {lat:>3d} {seed:>2d}  "
                  f"{r['wm_tr_acc_irr']*100:>4.0f}% {r['wm_tr_acc_rev']*100:>4.0f}%  "
                  f"{r['Dw_hat_irr_s0']:>6.3f} {r['Dw_hat_rev_s0']:>6.3f} "
                  f"{r['max_Dw_hat_rev_anywhere']:>6.3f}  "
                  f"{r['collapse_ratio']:>6.3f} {r['charge_load_ratio']:>6.3f}  "
                  f"{str(r['asymmetric_triggered_global'])[0]:>3} "
                  f"{str(r['asymmetric_triggered_at_s0'])[0]:>3} "
                  f"{'PASS' if r['passed'] else 'FAIL':>4}")
    return rows


def run_ablation_sweep() -> Dict[str, List[Dict[str, Any]]]:
    print("\n[Ablation] single-knob sweeps at base "
          f"{ABL_BASE}")
    abl = {"label_noise": [], "obs_noise": [], "epochs": [], "capacity": [],
           "recover_corrupt": []}

    print("  -- label_noise_p sweep --")
    for v in ABL_LABEL_NOISE:
        cfg = {**ABL_BASE, "label_noise_p": v}
        for seed in SEEDS:
            abl["label_noise"].append(_eval_cfg(f"abl_lnp_{v}", cfg, seed))

    print("  -- obs_noise_std sweep --")
    for v in ABL_OBS_NOISE:
        cfg = {**ABL_BASE, "obs_noise_std": v}
        for seed in SEEDS:
            abl["obs_noise"].append(_eval_cfg(f"abl_ons_{v}", cfg, seed))

    print("  -- epochs sweep --")
    for v in ABL_EPOCHS:
        cfg = {**ABL_BASE, "epochs": v}
        for seed in SEEDS:
            abl["epochs"].append(_eval_cfg(f"abl_ep_{v}", cfg, seed))

    print("  -- capacity sweep --")
    for (hid, lat) in ABL_CAPACITY:
        cfg = {**ABL_BASE, "hidden": hid, "latent": lat}
        for seed in SEEDS:
            abl["capacity"].append(
                _eval_cfg(f"abl_cap_h{hid}_l{lat}", cfg, seed))

    print("  -- recover_corrupt_p sweep --")
    for v in ABL_RECOVER:
        cfg = {**ABL_BASE, "recover_corrupt_p": v}
        for seed in SEEDS:
            abl["recover_corrupt"].append(
                _eval_cfg(f"abl_rcp_{v}", cfg, seed))

    return abl


# ====================================================================
# Aggregation
# ====================================================================

def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group rows by 'name' and aggregate over seeds.

    The aggregation keeps the ORDER of first appearance of each name.
    Triggered subsets: GLOBAL (any-(s,a) on rev) and S0 (decision point).
    """
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for r in rows:
        if r["name"] not in by_name:
            by_name[r["name"]] = []
            order.append(r["name"])
        by_name[r["name"]].append(r)
    out = []
    for name in order:
        rs = by_name[name]
        triggered_global = [r for r in rs if r["asymmetric_triggered_global"]]
        triggered_s0     = [r for r in rs if r["asymmetric_triggered_at_s0"]]
        agg = {
            "name": name,
            "n_seeds": len(rs),
            "label_noise_p": rs[0]["label_noise_p"],
            "obs_noise_std": rs[0]["obs_noise_std"],
            "recover_corrupt_p": rs[0].get("recover_corrupt_p", 0.0),
            "epochs": rs[0]["epochs"],
            "hidden": rs[0]["hidden"],
            "latent": rs[0]["latent"],
            "mean_tr_acc_irr": float(np.mean([r["wm_tr_acc_irr"] for r in rs])),
            "mean_tr_acc_rev": float(np.mean([r["wm_tr_acc_rev"] for r in rs])),
            "mean_Dw_hat_irr": float(np.mean([r["Dw_hat_irr_s0"] for r in rs])),
            "mean_Dw_hat_rev": float(np.mean([r["Dw_hat_rev_s0"] for r in rs])),
            "max_Dw_hat_rev":  float(max(r["Dw_hat_rev_s0"] for r in rs)),
            "mean_max_Dw_hat_rev_anywhere": float(np.mean(
                [r["max_Dw_hat_rev_anywhere"] for r in rs])),
            "max_max_Dw_hat_rev_anywhere":  float(max(
                r["max_Dw_hat_rev_anywhere"] for r in rs)),
            "mean_collapse_ratio": float(np.mean([r["collapse_ratio"] for r in rs])),
            "max_collapse_ratio":  float(max(r["collapse_ratio"] for r in rs)),
            "mean_charge_ratio":   float(np.mean([r["charge_load_ratio"] for r in rs])),
            "min_charge_ratio":    float(min(r["charge_load_ratio"] for r in rs)),
            "n_pass":              sum(1 for r in rs if r["passed"]),
            "n_asymmetric_triggered_global": len(triggered_global),
            "n_asymmetric_triggered_at_s0":  len(triggered_s0),
            "any_triggered_global": len(triggered_global) > 0,
            "any_triggered_at_s0":  len(triggered_s0)     > 0,
        }
        if triggered_global:
            agg["mean_collapse_ratio_triggered_global"] = float(
                np.mean([r["collapse_ratio"] for r in triggered_global]))
            agg["max_collapse_ratio_triggered_global"] = float(
                max(r["collapse_ratio"] for r in triggered_global))
        else:
            agg["mean_collapse_ratio_triggered_global"] = None
            agg["max_collapse_ratio_triggered_global"] = None
        if triggered_s0:
            agg["mean_collapse_ratio_triggered_s0"] = float(
                np.mean([r["collapse_ratio"] for r in triggered_s0]))
            agg["max_collapse_ratio_triggered_s0"] = float(
                max(r["collapse_ratio"] for r in triggered_s0))
        else:
            agg["mean_collapse_ratio_triggered_s0"] = None
            agg["max_collapse_ratio_triggered_s0"] = None
        out.append(agg)
    return out


# ====================================================================
# Verdict
# ====================================================================

def compute_verdict(severity_agg: List[Dict[str, Any]],
                     ablation_aggs: Dict[str, List[Dict[str, Any]]]
                     ) -> Dict[str, Any]:
    """Decide PASS / FAIL / INCONCLUSIVE on the pre-registered protocol.

    Validity has two layers:
      GLOBAL trigger:  rev twin D_w_hat > 0 SOMEWHERE.  Matches the user
                       spec phrasing exactly; lower bar.
      S0 trigger:      rev twin D_w_hat > 0 AT s_0.  Higher bar -- only
                       a s_0 asymmetric error can change the action at
                       the decision point and break collapse in this
                       gridworld (intermediate states have a single
                       available action, so their D_w_hat doesn't move
                       the trajectory).

    The pre-registered PASS / FAIL test is run on configurations where
    the S0 trigger fires, because that is the only subset where the
    failure channel can actually express itself in this twin geometry.
    The GLOBAL trigger is reported separately so the reader can see we
    DID probe the looser definition of asymmetric error too.

    PASS iff: at every config where the S0 trigger fires, max-over-seeds
              collapse_ratio remains <= COLLAPSE_THRESHOLD.
    FAIL iff: at least one S0-trigger config has max-over-seeds
              collapse_ratio > COLLAPSE_THRESHOLD.
    INCONCLUSIVE iff: no config fired the S0 trigger.  This means the
                      sweep failed to probe the decision-affecting
                      failure channel and we cannot conclude.
    """
    all_agg = list(severity_agg)
    for kind_rows in ablation_aggs.values():
        all_agg.extend(kind_rows)

    triggered_global = [a for a in all_agg if a["any_triggered_global"]]
    triggered_s0     = [a for a in all_agg if a["any_triggered_at_s0"]]
    n_runs_global = sum(a["n_asymmetric_triggered_global"] for a in all_agg)
    n_runs_s0     = sum(a["n_asymmetric_triggered_at_s0"]  for a in all_agg)

    if not triggered_s0:
        if triggered_global:
            return {
                "verdict": "INCONCLUSIVE_DECISION_CHANNEL",
                "reason": (f"GLOBAL trigger fired in {len(triggered_global)} "
                            f"config(s) ({n_runs_global} per-seed runs): the "
                            f"WM did produce asymmetric error on rev twin at "
                            f"some (s, a).  But the S0-specific trigger never "
                            f"fired -- no config produced D_w_hat(rev, s_0, "
                            f"a_decoy) > 0.  In this gridworld only the s_0 "
                            f"decision matters for the trajectory, so we have "
                            f"NOT probed the decision-affecting failure "
                            f"channel.  Escalate: add a perturbation that "
                            f"specifically corrupts the (lava, recover) "
                            f"transition (recover_corrupt_p)."),
                "triggered_runs_global": n_runs_global,
                "triggered_runs_at_s0":  0,
            }
        return {
            "verdict": "INCONCLUSIVE",
            "reason": ("No configuration triggered the asymmetric error "
                        "(D_w_hat(rev) > 0) at any (s, a) on the reversible "
                        "twin.  The failure channel was not probed at all; "
                        "this stage is not a valid PASS.  Escalate with "
                        "more aggressive perturbations."),
            "triggered_runs_global": 0,
            "triggered_runs_at_s0":  0,
        }

    failed_aggs = [a for a in triggered_s0
                    if a["max_collapse_ratio_triggered_s0"] is not None
                    and a["max_collapse_ratio_triggered_s0"] > COLLAPSE_THRESHOLD]
    failed_mean_aggs = [a for a in triggered_s0
                        if a["mean_collapse_ratio_triggered_s0"] is not None
                        and a["mean_collapse_ratio_triggered_s0"] > COLLAPSE_THRESHOLD]

    if failed_aggs:
        return {
            "verdict": "FAIL",
            "reason": (f"Among {len(triggered_s0)} configurations where the "
                        f"S0-specific asymmetric error fired, {len(failed_aggs)}"
                        f" have at least one seed with collapse_ratio > "
                        f"{COLLAPSE_THRESHOLD} on the reversible twin -- the "
                        f"Phase-1 failure channel breaks collapse in the "
                        f"learned-WM + MPPI setting at those error levels.  "
                        f"Honest negative result; reported as-is."),
            "triggered_runs_global": n_runs_global,
            "triggered_runs_at_s0":  n_runs_s0,
            "failed_levels":      [a["name"] for a in failed_aggs],
            "failed_mean_levels": [a["name"] for a in failed_mean_aggs],
        }

    return {
        "verdict": "PASS",
        "reason": (f"S0-specific asymmetric error fired in "
                    f"{len(triggered_s0)} configurations ({n_runs_s0} per-seed "
                    f"runs) -- the decision-affecting channel WAS probed.  In "
                    f"each, max-over-seeds collapse_ratio on rev stayed <= "
                    f"{COLLAPSE_THRESHOLD}.  The per-step MPPI decision-time "
                    f"D_w_hat charge absorbed the asymmetric error gracefully "
                    f"-- no Phase-1 redux."),
        "triggered_runs_global": n_runs_global,
        "triggered_runs_at_s0":  n_runs_s0,
        "failed_mean_levels": [a["name"] for a in failed_mean_aggs],
    }


# ====================================================================
# Figure
# ====================================================================

def write_figure(severity_agg: List[Dict[str, Any]],
                  severity_rows: List[Dict[str, Any]],
                  ablation_aggs: Dict[str, List[Dict[str, Any]]],
                  out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    # (a) Severity sweep: WM transition accuracy
    ax = axes[0, 0]
    names = [a["name"] for a in severity_agg]
    accs_i = [a["mean_tr_acc_irr"] for a in severity_agg]
    accs_r = [a["mean_tr_acc_rev"] for a in severity_agg]
    x = np.arange(len(names))
    ax.plot(x, accs_i, "o-", label="WM tr.acc irr")
    ax.plot(x, accs_r, "s-", label="WM tr.acc rev")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("WM transition accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("(a) Severity -- WM accuracy")
    ax.legend(fontsize=8)

    # (b) Severity sweep: D_w_hat(s_0, a_decoy)
    ax = axes[0, 1]
    dwh_i = [a["mean_Dw_hat_irr"] for a in severity_agg]
    dwh_r = [a["mean_Dw_hat_rev"] for a in severity_agg]
    ax.plot(x, dwh_i, "o-", label="D_w_hat irr s_0 (target 1.778)")
    ax.plot(x, dwh_r, "s-", label="D_w_hat rev s_0 (target 0)")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("D_w_hat(s_0, a_decoy)")
    ax.set_title("(b) Severity -- D_w_hat at s_0")
    ax.legend(fontsize=8)

    # (c) Severity sweep: max-anywhere D_w_hat on rev
    ax = axes[0, 2]
    rev_global = [a["mean_max_Dw_hat_rev_anywhere"] for a in severity_agg]
    ax.plot(x, dwh_r,     "s-", label="rev s_0 (decision-affecting)")
    ax.plot(x, rev_global, "^-", label="rev max over all (s,a)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("D_w_hat (rev twin)")
    ax.set_title("(c) Severity -- asymmetric error: s_0 vs anywhere")
    ax.legend(fontsize=8)

    # (d) Severity sweep: collapse / charge per seed
    ax = axes[1, 0]
    for i, name in enumerate(names):
        rs = [r for r in severity_rows if r["name"] == name]
        for r in rs:
            colour = "red" if r["asymmetric_triggered_at_s0"] else (
                "orange" if r["asymmetric_triggered_global"] else "lightgrey")
            ax.scatter([i - 0.15], [r["collapse_ratio"]],
                        color=colour, marker="o", s=24)
            ax.scatter([i + 0.15], [r["charge_load_ratio"]],
                        color="green", marker="s", s=22, alpha=0.7)
    ax.axhline(COLLAPSE_THRESHOLD, color="r", ls="--", lw=1,
                label=f"collapse PASS <= {COLLAPSE_THRESHOLD}")
    ax.axhline(CHARGE_THRESHOLD, color="g", ls="--", lw=1,
                label=f"charge   PASS >= {CHARGE_THRESHOLD}")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("ratio (per seed)")
    ax.set_title("(d) Severity -- collapse (red=s_0 trig, orange=global)")
    ax.legend(fontsize=8)

    def _add_abl(ax, kind, x_field, x_label, title):
        ag = aggregate(ablation_aggs[kind])
        xs = [a[x_field] for a in ag]
        ax.plot(xs, [a["mean_Dw_hat_rev"] for a in ag], "s-",
                 label="D_w_hat rev s_0")
        ax.plot(xs, [a["mean_max_Dw_hat_rev_anywhere"] for a in ag], "^-",
                 label="D_w_hat rev max-anywhere")
        ax.plot(xs, [a["mean_collapse_ratio"] for a in ag], "o-",
                 label="mean collapse_ratio")
        ax.plot(xs, [a["mean_charge_ratio"] for a in ag], "x--",
                 label="mean charge_ratio")
        ax.axhline(COLLAPSE_THRESHOLD, color="r", ls=":", lw=0.7)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.legend(fontsize=7)

    # (e) Ablation: label noise.
    _add_abl(axes[1, 1], "label_noise", "label_noise_p",
              "label_noise_p (others fixed)",
              "(e) Ablation -- label noise")

    # (f) Ablation: obs noise.
    _add_abl(axes[1, 2], "obs_noise", "obs_noise_std",
              "obs_noise_std (others fixed)",
              "(f) Ablation -- obs noise")

    # (g) Ablation: epochs.
    _add_abl(axes[2, 0], "epochs", "epochs",
              "epochs (log axis)",
              "(g) Ablation -- training budget")
    axes[2, 0].set_xscale("log")

    # (h) Ablation: capacity.
    ag = aggregate(ablation_aggs["capacity"])
    xs = [f"h{a['hidden']}/l{a['latent']}" for a in ag]
    xi = np.arange(len(xs))
    ax = axes[2, 1]
    ax.plot(xi, [a["mean_Dw_hat_rev"] for a in ag], "s-",
             label="D_w_hat rev s_0")
    ax.plot(xi, [a["mean_max_Dw_hat_rev_anywhere"] for a in ag], "^-",
             label="D_w_hat rev max-anywhere")
    ax.plot(xi, [a["mean_collapse_ratio"] for a in ag], "o-",
             label="mean collapse_ratio")
    ax.plot(xi, [a["mean_charge_ratio"] for a in ag], "x--",
             label="mean charge_ratio")
    ax.axhline(COLLAPSE_THRESHOLD, color="r", ls=":", lw=0.7)
    ax.set_xticks(xi)
    ax.set_xticklabels(xs, fontsize=8)
    ax.set_xlabel("WM capacity (hidden/latent)")
    ax.set_title("(h) Ablation -- capacity")
    ax.legend(fontsize=7)

    # (i) Ablation: surgical recover_corrupt_p (the decision-channel probe).
    _add_abl(axes[2, 2], "recover_corrupt", "recover_corrupt_p",
              "recover_corrupt_p (surgical: lava->recover target = absorb)",
              "(i) Ablation -- surgical recover corruption")

    fig.suptitle("Stage-6 Kill-Gate 3 -- collapse robustness under noisy /"
                 " imperfect WM (perturbation sweep)", fontsize=12)
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
    print("Stage-6 Kill-Gate 3 -- collapse robustness under noisy / imperfect WM")
    print("=" * 78)
    print(f"Defaults: {DEFAULTS}")
    print(f"Lambda: {LAMBDA}  collapse_threshold: {COLLAPSE_THRESHOLD}  "
          f"charge_threshold: {CHARGE_THRESHOLD}")
    print(f"Trigger eps for asymmetric error: D_w_hat(rev) > {TRIGGER_EPS}")
    print("Reuse: Stage-4 build_lava_gridworld, Stage-5 WorldModel / "
          "build_mdp_hat / run_planner, Stage-1 destroyed_mass / policy_*")

    severity_rows = run_severity_sweep()
    severity_agg = aggregate(severity_rows)

    ablation_rows = run_ablation_sweep()
    ablation_aggs = {k: aggregate(v) for k, v in ablation_rows.items()}

    verdict = compute_verdict(severity_agg, ablation_aggs)

    # Print aggregated table
    print("\n" + "=" * 78)
    print("Aggregated severity table")
    print("=" * 78)
    print(f"{'level':>22} {'lnp':>4} {'ons':>4} {'rcp':>4} {'ep':>4} "
          f"{'h':>3} {'l':>3}  "
          f"{'trA_i':>5} {'trA_r':>5}  {'Dwh_i':>6} {'Dwh_r':>6} "
          f"{'rev*':>6}  {'col_mx':>6} {'col_mn':>6} {'chr_mn':>6}  "
          f"{'asG':>4} {'asS':>4} {'pass':>4}")
    for a in severity_agg:
        print(f"{a['name']:>22} "
              f"{a['label_noise_p']:>4.2f} {a['obs_noise_std']:>4.2f} "
              f"{a['recover_corrupt_p']:>4.2f} {a['epochs']:>4d} "
              f"{a['hidden']:>3d} {a['latent']:>3d}  "
              f"{a['mean_tr_acc_irr']*100:>4.0f}% {a['mean_tr_acc_rev']*100:>4.0f}%  "
              f"{a['mean_Dw_hat_irr']:>6.3f} {a['mean_Dw_hat_rev']:>6.3f} "
              f"{a['mean_max_Dw_hat_rev_anywhere']:>6.3f}  "
              f"{a['max_collapse_ratio']:>6.3f} {a['mean_collapse_ratio']:>6.3f} "
              f"{a['min_charge_ratio']:>6.3f}  "
              f"{a['n_asymmetric_triggered_global']:>1d}/{a['n_seeds']:<2d} "
              f"{a['n_asymmetric_triggered_at_s0']:>1d}/{a['n_seeds']:<2d} "
              f"{a['n_pass']:>1d}/{a['n_seeds']:<2d}")

    print("\nAblation summaries:")
    for kind, ag in {k: aggregate(v) for k, v in ablation_rows.items()}.items():
        print(f"\n  -- {kind} --")
        print(f"  {'config':>24} {'trA_i':>5} {'trA_r':>5} "
              f"{'Dwh_r':>6} {'rev*':>6} {'col_mn':>7} {'col_mx':>7} "
              f"{'chr_mn':>7}  {'asG':>4} {'asS':>4}")
        for a in ag:
            if kind == "label_noise":
                cfg = f"lnp={a['label_noise_p']}"
            elif kind == "obs_noise":
                cfg = f"ons={a['obs_noise_std']}"
            elif kind == "epochs":
                cfg = f"ep={a['epochs']}"
            elif kind == "capacity":
                cfg = f"h{a['hidden']}/l{a['latent']}"
            else:
                cfg = f"rcp={a['recover_corrupt_p']}"
            print(f"  {cfg:>24} "
                  f"{a['mean_tr_acc_irr']*100:>4.0f}% "
                  f"{a['mean_tr_acc_rev']*100:>4.0f}%  "
                  f"{a['mean_Dw_hat_rev']:>6.3f} "
                  f"{a['mean_max_Dw_hat_rev_anywhere']:>6.3f} "
                  f"{a['mean_collapse_ratio']:>7.4f} "
                  f"{a['max_collapse_ratio']:>7.4f} "
                  f"{a['mean_charge_ratio']:>7.4f}  "
                  f"{a['n_asymmetric_triggered_global']}/{a['n_seeds']} "
                  f"{a['n_asymmetric_triggered_at_s0']}/{a['n_seeds']}")

    print("\n" + "=" * 78)
    print(f"Pre-registered verdict: {verdict['verdict']}")
    print(f"  {verdict['reason']}")
    print(f"  Triggered runs (global, any-(s,a) on rev): "
          f"{verdict.get('triggered_runs_global', 0)}")
    print(f"  Triggered runs (at s_0, decision-affecting): "
          f"{verdict.get('triggered_runs_at_s0', 0)}")
    print("=" * 78)

    pdf_path = os.path.join(_THIS_DIR, "stage6_sweep.pdf")
    write_figure(severity_agg, severity_rows, ablation_rows, pdf_path)
    print(f"Figure: {pdf_path}")

    dt = time.time() - t_start
    payload = {
        "verdict":         verdict["verdict"],
        "verdict_reason":  verdict["reason"],
        "verdict_meta":    {k: v for k, v in verdict.items()
                              if k not in ("verdict", "reason")},
        "wall_time_s":     dt,
        "defaults":        DEFAULTS,
        "lambda":          LAMBDA,
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "charge_threshold":   CHARGE_THRESHOLD,
        "trigger_eps":        TRIGGER_EPS,
        "seeds":              SEEDS,
        "severity_levels":    [
            {"name": n, "label_noise_p": p, "obs_noise_std": o,
             "epochs": e, "hidden": h, "latent": l,
             "recover_corrupt_p": rcp}
            for n, p, o, e, h, l, rcp in SEVERITY_LEVELS
        ],
        "severity_aggregated": severity_agg,
        "severity_per_run":    severity_rows,
        "ablation_base":       ABL_BASE,
        "ablation_aggregated": {k: aggregate(v) for k, v in ablation_rows.items()},
        "ablation_per_run":    ablation_rows,
    }
    out_path = os.path.join(_THIS_DIR, "stage6_results.json")
    with open(out_path, "w") as fh:
        json.dump(_to_jsonable(payload), fh, indent=2)
    print(f"Results: {out_path}")
    print(f"Wall time: {dt:.1f} s")

    return verdict["verdict"] == "PASS"


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

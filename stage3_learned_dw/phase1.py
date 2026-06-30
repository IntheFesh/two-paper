"""
phase1.py
=========

Stage-3 Phase 1 driver: MiniGrid deep RL with learned D_w shaping.

Estimated runtime
-----------------
A single PPO training run on the small SmashGateEnv (5x5 inner grid,
N_ACTIONS = 3, flat observation) at PHASE1_TOTAL_STEPS = 200_000 takes
roughly:

    CPU (8-core, no GPU)  : ~30-50 minutes
    GPU (consumer-grade)  : ~5-10 minutes

The full matrix is 3 agents x 2 twins = 6 trainings, plus the
D_w-hat pretraining (~15 s) and final evaluation (~30 s). Total at the
default budget:

    CPU : ~4-5 hours        (single-seed; per-spec smoke configuration)
    GPU : ~30-60 minutes    (single-seed; comfortably under 20 GPU-h cap)

Set PHASE1_TOTAL_STEPS via the env var STAGE3_PHASE1_STEPS to override
(e.g., 20000 for a 30-second-per-run CPU smoke test, used by the smoke
mode below). Any single training is well under 2h on either backend at
default settings, so we do not need user re-confirmation per spec.

What it tests
-------------
Identical to Phase 0's question, but the planner is replaced by a deep
RL agent (PPO) that LEARNS its own value function from rollouts. We
shape the reward by -lambda * D_w(s, a) where D_w comes from
{reward_only -> 0, learned_mrc -> MLP, oracle_mrc -> env's exact table}.

Pre-registered PASS/FAIL (LOCKED before any run)
------------------------------------------------
Same shape as Phase 0; numerical thresholds are slightly relaxed
because deep RL is inherently noisier than tabular planning.

(a) Estimator accuracy on held-out grid layouts:
        median rel err on positives < 0.20
        rank AUC (positive vs zero) >= 0.95
(b) Charge load-bearing on irreversible twin:
        oracle gap > 0  (= mean_return(oracle_mrc, irr) - mean_return(reward_only, irr))
        learned gap > 0
        learned gap >= 0.50 * oracle gap
(c) Collapse on reversible twin:
        |residual| <= 0.30 * irr_gain          (deep-RL noise => 0.30 vs 0.20)
        where residual = mean_return(learned_mrc, rev) - mean_return(reward_only, rev)
              irr_gain = mean_return(learned_mrc, irr) - mean_return(reward_only, irr)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from phase1_env import SmashGateEnv  # noqa: E402
from phase1_regressor import (  # noqa: E402
    DEFAULT_GAMMA,
    DEFAULT_R_D,
    DEFAULT_R_G,
    LearnedDwLookup,
    collect_samples,
    samples_to_tensors,
)
from phase1_shaped import (  # noqa: E402
    ShapedRewardWrapper,
    oracle_dw_source,
    zero_dw_source,
)
from regressor import regression_metrics, train_dw_regressor  # noqa: E402

# ---------------- Pre-registered constants ----------------
LAMBDA = 1.0
GAMMA = DEFAULT_GAMMA
R_D = DEFAULT_R_D
R_G = DEFAULT_R_G
GRID_SIZE = 7
MAX_STEPS = 64
N_TRAIN_SEEDS_DW = 200          # each seed contributes only a handful of
                                 # lava-facing (s,a) positives; 200 seeds
                                 # gives ~600 positives, enough with the
                                 # weighted MSE below.
N_VAL_SEEDS_DW = 30
N_EVAL_SEEDS_DW = 60
DW_POS_WEIGHT = 20.0            # weight on positive (D_w > 0) MSE terms;
                                 # roughly 1/p where p ~= 0.5% is the
                                 # positive fraction. Without this the
                                 # regressor collapses to predicting 0
                                 # everywhere and the shaping signal goes
                                 # to noise.

PHASE1_TOTAL_STEPS = int(os.environ.get("STAGE3_PHASE1_STEPS", 200_000))
PHASE1_EVAL_EPISODES = int(os.environ.get("STAGE3_PHASE1_EVAL_EPISODES", 40))
PHASE1_SEED = int(os.environ.get("STAGE3_PHASE1_SEED", 0))

PASS_THRESHOLDS = dict(
    estimator_max_median_rel_err=0.20,
    estimator_min_rank_auc=0.95,
    learned_gap_closure_fraction=0.50,
    collapse_residual_fraction=0.30,
)


# --------------------------------------------------------------------- #
# Env factories                                                          #
# --------------------------------------------------------------------- #


def base_env(mode: str) -> SmashGateEnv:
    return SmashGateEnv(
        size=GRID_SIZE, r_d=R_D, r_g=R_G, gamma=GAMMA,
        mode=mode, max_steps=MAX_STEPS,
    )


def make_shaped_env_fn(mode: str, dw_source: Callable):
    def _f():
        env = base_env(mode)
        return ShapedRewardWrapper(env, dw_source=dw_source, lam=LAMBDA)
    return _f


# --------------------------------------------------------------------- #
# Training (one PPO run per agent x twin)                                #
# --------------------------------------------------------------------- #


def train_one_agent(
    name: str,
    mode: str,
    dw_source: Callable,
    total_steps: int,
    seed: int,
    log_dir: str,
) -> PPO:
    """Train one PPO agent on (mode, dw_source) for `total_steps` steps."""
    env_fn = make_shaped_env_fn(mode, dw_source)
    vec = DummyVecEnv([env_fn])
    model = PPO(
        "MlpPolicy", vec,
        seed=seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=4,
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=0,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    t0 = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=False)
    train_time = time.time() - t0
    save_path = os.path.join(log_dir, f"ppo_{name}_{mode}.zip")
    model.save(save_path)
    return model, train_time


# --------------------------------------------------------------------- #
# Evaluation                                                             #
# --------------------------------------------------------------------- #


def evaluate_agent(model: PPO, mode: str, n_episodes: int,
                   eval_seed_base: int = 0) -> Dict[str, float]:
    """Roll the trained policy on FRESH layouts (held-out seeds) and
    report mean / std of the RAW environment reward sum. We deliberately
    report the RAW reward, not the shaped reward, so the three agents
    are scored on the same physical task (r_g - r_d trade-offs etc.)
    rather than on their own internal training signal.
    """
    env = base_env(mode)
    returns: List[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=eval_seed_base + ep)
        ep_return = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_return += float(reward)
        returns.append(ep_return)
    arr = np.array(returns)
    return dict(mean=float(arr.mean()), std=float(arr.std()),
                n=int(arr.size), returns=arr.tolist())


# --------------------------------------------------------------------- #
# Main                                                                   #
# --------------------------------------------------------------------- #


def main() -> bool:
    t_total = time.time()
    print("=" * 72)
    print("Stage-3 Phase 1 -- MiniGrid deep RL with learned D_w shaping")
    print("=" * 72)
    print(f"  config: total_steps={PHASE1_TOTAL_STEPS}, "
          f"eval_episodes={PHASE1_EVAL_EPISODES}, seed={PHASE1_SEED}")
    print(f"  override via env vars: STAGE3_PHASE1_STEPS, "
          f"STAGE3_PHASE1_EVAL_EPISODES, STAGE3_PHASE1_SEED")

    results_dir = os.path.join(_HERE, "phase1_results")
    os.makedirs(results_dir, exist_ok=True)

    # ---------- 1. Pretrain learned D_w on procedurally generated grids ----------
    print("\n[1/4] Pretraining D_w-hat on procedurally generated grids ...")
    t0 = time.time()
    train_seeds = list(range(N_TRAIN_SEEDS_DW))
    val_seeds = list(range(N_TRAIN_SEEDS_DW, N_TRAIN_SEEDS_DW + N_VAL_SEEDS_DW))
    eval_seeds_dw = list(range(N_TRAIN_SEEDS_DW + N_VAL_SEEDS_DW,
                                N_TRAIN_SEEDS_DW + N_VAL_SEEDS_DW + N_EVAL_SEEDS_DW))
    train_samples = collect_samples(train_seeds, size=GRID_SIZE,
                                    r_d=R_D, r_g=R_G, gamma=GAMMA)
    val_samples = collect_samples(val_seeds, size=GRID_SIZE,
                                  r_d=R_D, r_g=R_G, gamma=GAMMA)
    eval_dw_samples = collect_samples(eval_seeds_dw, size=GRID_SIZE,
                                      r_d=R_D, r_g=R_G, gamma=GAMMA)
    Xt, yt = samples_to_tensors(train_samples)
    Xv, yv = samples_to_tensors(val_samples)
    Xe, ye = samples_to_tensors(eval_dw_samples)
    print(f"  D_w train={Xt.shape[0]} val={Xv.shape[0]} held-out-test={Xe.shape[0]}, "
          f"in_dim={Xt.shape[1]}, collection={time.time()-t0:.1f}s")

    t0 = time.time()
    model_dw, val_metrics = train_dw_regressor(
        Xt, yt, Xv, yv, hidden=64, epochs=80, lr=3e-3,
        seed=PHASE1_SEED, verbose=True, pos_weight=DW_POS_WEIGHT,
    )
    print(f"  D_w-hat training time = {time.time()-t0:.1f}s")
    with torch.no_grad():
        eval_pred = model_dw(Xe).numpy()
    eval_metrics = regression_metrics(eval_pred, ye.numpy())
    print(f"  D_w-hat held-out-test metrics = {eval_metrics}")

    # Pickle the model so the user can reload without retraining.
    torch.save(model_dw.state_dict(),
               os.path.join(results_dir, "dw_hat.pt"))

    # ---------- 2. Train 3 agents x 2 twins ----------
    print(f"\n[2/4] Training 3 agents x 2 twins (each {PHASE1_TOTAL_STEPS} env steps) ...")
    learned_source = LearnedDwLookup(model_dw)
    dw_sources = {
        "reward_only": zero_dw_source,
        "oracle_mrc": oracle_dw_source,
        "learned_mrc": learned_source,
    }

    trained_models: Dict[Tuple[str, str], PPO] = {}
    train_times: Dict[str, float] = {}
    for mode in ("irreversible", "reversible"):
        for agent_name, src in dw_sources.items():
            tag = f"{agent_name}__{mode}"
            print(f"  - {tag}")
            t0 = time.time()
            model, dt = train_one_agent(
                agent_name, mode, src,
                total_steps=PHASE1_TOTAL_STEPS,
                seed=PHASE1_SEED,
                log_dir=results_dir,
            )
            trained_models[(agent_name, mode)] = model
            train_times[tag] = dt
            print(f"    train_time = {dt:.1f}s")

    # ---------- 3. Evaluate all six on raw env reward ----------
    print(f"\n[3/4] Evaluating each agent over {PHASE1_EVAL_EPISODES} held-out episodes ...")
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode in ("irreversible", "reversible"):
        summary[mode] = {}
        for agent_name in dw_sources:
            res = evaluate_agent(
                trained_models[(agent_name, mode)],
                mode,
                n_episodes=PHASE1_EVAL_EPISODES,
                eval_seed_base=10_000,
            )
            summary[mode][agent_name] = res
            print(f"  {agent_name:>12s} @ {mode:13s}  "
                  f"mean = {res['mean']:.4f}  (std {res['std']:.4f}, n={res['n']})")

    # ---------- 4. Pre-registered PASS conditions ----------
    irr = summary["irreversible"]
    rev = summary["reversible"]
    oracle_gap_irr = irr["oracle_mrc"]["mean"] - irr["reward_only"]["mean"]
    learned_gap_irr = irr["learned_mrc"]["mean"] - irr["reward_only"]["mean"]
    oracle_gap_rev = rev["oracle_mrc"]["mean"] - rev["reward_only"]["mean"]
    learned_gap_rev = rev["learned_mrc"]["mean"] - rev["reward_only"]["mean"]

    cond_a_med = eval_metrics["median_rel_err_on_positives"] \
        < PASS_THRESHOLDS["estimator_max_median_rel_err"]
    cond_a_auc = eval_metrics["rank_auc_positive_vs_zero"] \
        >= PASS_THRESHOLDS["estimator_min_rank_auc"]
    cond_a = bool(cond_a_med and cond_a_auc)

    if oracle_gap_irr > 1e-6:
        gap_closure = learned_gap_irr / oracle_gap_irr
    else:
        gap_closure = 0.0
    cond_b = bool(
        oracle_gap_irr > 1e-6
        and learned_gap_irr > 1e-6
        and gap_closure >= PASS_THRESHOLDS["learned_gap_closure_fraction"]
    )

    if learned_gap_irr > 1e-6:
        collapse_ratio = abs(learned_gap_rev) / learned_gap_irr
    else:
        collapse_ratio = float("inf")
    cond_c = bool(collapse_ratio <= PASS_THRESHOLDS["collapse_residual_fraction"])

    overall = cond_a and cond_b and cond_c

    print("\n[4/4] verdict on pre-registered PASS conditions")
    print("-" * 72)
    print("(a) Estimator quality on held-out grid layouts:")
    print(f"    median rel err on positives = {eval_metrics['median_rel_err_on_positives']:.4f}"
          f"  (thr < {PASS_THRESHOLDS['estimator_max_median_rel_err']})  "
          f"=> {'PASS' if cond_a_med else 'FAIL'}")
    print(f"    rank AUC (positive vs zero) = {eval_metrics['rank_auc_positive_vs_zero']:.4f}"
          f"  (thr >= {PASS_THRESHOLDS['estimator_min_rank_auc']})  "
          f"=> {'PASS' if cond_a_auc else 'FAIL'}")
    print(f"    (a) overall: {'PASS' if cond_a else 'FAIL'}")

    print(f"\n(b) Charge load-bearing on irreversible twin "
          f"(mean reward over {PHASE1_EVAL_EPISODES} held-out episodes):")
    print(f"    reward_only  = {irr['reward_only']['mean']:+.4f}  (std {irr['reward_only']['std']:.4f})")
    print(f"    oracle_mrc   = {irr['oracle_mrc']['mean']:+.4f}  (std {irr['oracle_mrc']['std']:.4f})")
    print(f"    learned_mrc  = {irr['learned_mrc']['mean']:+.4f}  (std {irr['learned_mrc']['std']:.4f})")
    print(f"    oracle gap   = {oracle_gap_irr:+.4f}")
    print(f"    learned gap  = {learned_gap_irr:+.4f}  "
          f"({100*gap_closure:.1f}% of oracle gap; thr >= "
          f"{100*PASS_THRESHOLDS['learned_gap_closure_fraction']:.0f}%)")
    print(f"    (b) overall: {'PASS' if cond_b else 'FAIL'}")

    print("\n(c) Reversible-twin collapse (causal identification):")
    print(f"    reward_only  = {rev['reward_only']['mean']:+.4f}  (std {rev['reward_only']['std']:.4f})")
    print(f"    oracle_mrc   = {rev['oracle_mrc']['mean']:+.4f}  (std {rev['oracle_mrc']['std']:.4f})")
    print(f"    learned_mrc  = {rev['learned_mrc']['mean']:+.4f}  (std {rev['learned_mrc']['std']:.4f})")
    print(f"    oracle  residual = {oracle_gap_rev:+.4f}  (expect ~0)")
    print(f"    learned residual = {learned_gap_rev:+.4f}  "
          f"(collapse ratio |residual|/irr_gain = {collapse_ratio:.4f}; thr <= "
          f"{PASS_THRESHOLDS['collapse_residual_fraction']})")
    print(f"    (c) overall: {'PASS' if cond_c else 'FAIL'}")

    print("\n" + "=" * 72)
    print(f"Phase 1 verdict: {'PASS' if overall else 'FAIL'}")
    print("=" * 72)
    print(f"\nTotal Phase 1 runtime: {time.time()-t_total:.1f}s")

    payload = dict(
        passed=bool(overall),
        conditions=dict(
            estimator_quality=bool(cond_a),
            charge_load_bearing=bool(cond_b),
            reversible_collapse=bool(cond_c),
        ),
        thresholds=PASS_THRESHOLDS,
        config=dict(
            total_steps=PHASE1_TOTAL_STEPS,
            eval_episodes=PHASE1_EVAL_EPISODES,
            seed=PHASE1_SEED,
            grid_size=GRID_SIZE,
            r_d=R_D, r_g=R_G, gamma=GAMMA, lam=LAMBDA,
            max_steps=MAX_STEPS,
        ),
        dw_hat_val_metrics=val_metrics,
        dw_hat_eval_metrics=eval_metrics,
        agent_returns=summary,
        gaps=dict(
            oracle_gap_irr=oracle_gap_irr,
            learned_gap_irr=learned_gap_irr,
            oracle_gap_rev=oracle_gap_rev,
            learned_gap_rev=learned_gap_rev,
            gap_closure_ratio=gap_closure,
            collapse_ratio=collapse_ratio,
        ),
        train_times=train_times,
    )
    with open(os.path.join(results_dir, "phase1_results.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\nResults written to {results_dir}/phase1_results.json")

    return overall


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

"""
phase1_regressor.py
===================

Train the learned D_w estimator for Phase 1's MiniGrid SmashGateEnv from
ORACLE labels. The estimator consumes the env's observation (object-id
grid + agent x, y, dir) and an action one-hot, and predicts D_w(s, a).

The same `SmashGateEnv.oracle_dw` we use for shaping the oracle-MRC
agent provides the labels here, so the only thing the regressor is doing
is approximating that function from observation features. At deploy time
the regressor is wrapped in `LearnedDwLookup` and queried per (env, s,
action) by the shaped-reward wrapper; the env's actual reachability
graph is never consulted.

Training distribution
---------------------
N_TRAIN_SEEDS random SmashGateEnv layouts (irreversible AND reversible),
each fully enumerated over (x, y, dir, action). Held-out validation uses
disjoint seeds. The label distribution is sparse (most (s, a) have
D_w == 0; only a few "facing lava + forward" pairs are positive), so we
report median relative error on positives and rank-AUC explicitly --
overall MAE is dominated by zeros.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from phase1_env import (  # noqa: E402
    ACT_FORWARD,
    ACT_LEFT,
    ACT_RIGHT,
    N_ACTIONS,
    SmashGateEnv,
)
from regressor import DwMLP, regression_metrics, train_dw_regressor  # noqa: E402

MODES = ("irreversible", "reversible")
DEFAULT_GAMMA = 0.97
DEFAULT_R_D = 0.85
DEFAULT_R_G = 1.0


MODE_INDEX = {m: i for i, m in enumerate(MODES)}


def featurise(obs: np.ndarray, action: int, mode: str) -> np.ndarray:
    """Concatenate the env observation with an action one-hot AND a
    mode one-hot. The mode bit is essential: the env observation
    encodes the lava cell identically whether the lava is lethal or
    recoverable, but D_w differs (positive in irreversible, exactly 0
    in reversible). Without the mode signal the regressor sees the
    same input with two different labels and converges to predicting
    the average -- which is a Stage-1 violation. The mode is observable
    to an embodied agent through the env's transition behaviour (the
    recovery action present in reversible mode), so giving it as an
    input matches the embodied story.
    """
    action_oh = np.zeros(N_ACTIONS, dtype=np.float32)
    action_oh[int(action)] = 1.0
    mode_oh = np.zeros(len(MODES), dtype=np.float32)
    mode_oh[MODE_INDEX[mode]] = 1.0
    return np.concatenate([obs.astype(np.float32), action_oh, mode_oh])


def feature_dim(env: SmashGateEnv) -> int:
    return env.observation_space.shape[0] + N_ACTIONS + len(MODES)


def collect_samples(
    seeds: List[int],
    size: int = 7,
    r_d: float = DEFAULT_R_D,
    r_g: float = DEFAULT_R_G,
    gamma: float = DEFAULT_GAMMA,
) -> List[Dict]:
    """For each seed and each mode, enumerate all (x, y, dir, action)
    pairs in the grid and emit (obs, action, D_w) samples.

    We construct each sample by forcibly setting the env's agent state
    and reading the obs + oracle_dw -- avoiding any reliance on the
    actual rollout. This is fast (a few seconds for hundreds of seeds).
    """
    out: List[Dict] = []
    for seed in seeds:
        for mode in MODES:
            env = SmashGateEnv(mode=mode, size=size, r_d=r_d, r_g=r_g,
                                gamma=gamma)
            env.reset(seed=seed)
            for x in range(1, env.grid.width - 1):
                for y in range(1, env.grid.height - 1):
                    cell = env.grid.get(x, y)
                    if cell is not None and type(cell).__name__ in ("Wall", "Lava", "Goal"):
                        # Lava / goal cells are absorbing or terminal-on-entry;
                        # we never query D_w from there.
                        continue
                    for d in range(4):
                        env.agent_pos = (x, y)
                        env.agent_dir = d
                        obs = env._make_obs()
                        for a in range(N_ACTIONS):
                            dw = env.oracle_dw(a)
                            out.append({
                                "obs": obs.copy(), "action": a,
                                "mode": mode, "Dw": dw,
                            })
    return out


def samples_to_tensors(samples: List[Dict]):
    n = len(samples)
    in_dim = samples[0]["obs"].shape[0] + N_ACTIONS + len(MODES)
    X = np.zeros((n, in_dim), dtype=np.float32)
    y = np.zeros((n,), dtype=np.float32)
    for i, s in enumerate(samples):
        X[i] = featurise(s["obs"], s["action"], s["mode"])
        y[i] = s["Dw"]
    return torch.from_numpy(X), torch.from_numpy(y)


# --------------------------------------------------------------------- #
# Estimator wrapper used by the shaped-reward env                       #
# --------------------------------------------------------------------- #


class LearnedDwLookup:
    """Drop-in lookup for the shaped-reward env: given (env, action),
    queries the model on featurise(env._make_obs(), action, env._mode).
    The 'env' arg mirrors the oracle path so the shaped wrapper has a
    uniform API. The mode is read off `env._mode` at query time so the
    same trained model serves both the irreversible and reversible twin
    -- it must, for the collapse test to be meaningful.
    """

    def __init__(self, model: DwMLP):
        self.model = model
        self.model.eval()
        self._torch = torch

    def __call__(self, env: SmashGateEnv, action: int) -> float:
        obs = env._make_obs()
        x = featurise(obs, action, env._mode)
        with self._torch.no_grad():
            v = float(self.model(self._torch.from_numpy(x).unsqueeze(0)).item())
        return max(0.0, v)

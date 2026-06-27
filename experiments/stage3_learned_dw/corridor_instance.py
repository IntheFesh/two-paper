"""
corridor_instance.py
====================

Random Stage-1 corridor instances for Phase 0's FA kill-gate.

Why corridors instead of free 2D grids in Phase 0?
--------------------------------------------------
With a model-based finite-horizon planner (which is what we use in
Phase 0 to isolate the *D_w-approximation* question from a learner's
value-function-approximation question), Stage-1's exact formulation
requires the rollout after the s_0 decision to be forced. A genuine 2D
grid with full action freedom would couple two unknowns -- "does the
approximate D_w fire correctly?" with "does the agent navigate after
refusing the decoy?" -- and confound the test. Stage-1 already proved
this on the corridor; Phase 0 keeps the same MDP topology and ONLY
swaps the exact destroyed_mass for a learned regressor so the only
moving variable is the approximation. Phase 1 then lifts this into
full 2D pixel-based deep RL (MiniGrid), where the agent learns its own
value function and the navigation question gets answered jointly.

Per-instance varying parameters
-------------------------------
    k     : number of target cells in the safe corridor (1..K_MAX)
    m     : pre-target stem length (M_MIN..M_MAX); H is set to m
    r_d   : decoy reward at s0 -> trap edge
    r_g   : per-target outgoing-edge reward

We enforce V_safe(k, m, r_g) > r_d on every generated instance so the
mechanism is well-posed (the oracle MRC strictly improves over the
oblivious planner). Generation rejects instances violating this.

Feature encoding for the learned regressor
------------------------------------------
We pass the regressor:

    k_oh   (K_MAX,)   one-hot of k
    m_oh   (M_MAX,)   one-hot of m
    r_d, r_g, gamma   raw scalars
    state_oh          one-hot among {s0, trap, absorb, c1..c_{M_MAX+K_MAX}}
    action_oh         one-hot among ACTION_LIST
    mode_oh   (2,)    one-hot of {irreversible, reversible}

This is structurally sufficient for the regressor to learn the closed-
form

    D_w(s0, a_decoy)[irreversible] = sum_{j=1}^{k} gamma^{m+j-1} * r_g
    D_w(...)         [reversible]   = 0

but the network sees none of those structural primitives directly -- it
gets the raw parameters and must compose the geometric series and the
distance term itself. The regressor's quality is reported against the
exact Stage-1 destroyed_mass labels.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE1 = os.path.normpath(os.path.join(_HERE, "..", "stage1_unified"))
if _STAGE1 not in sys.path:
    sys.path.insert(0, _STAGE1)

from stage1_unified_validation import (  # noqa: E402
    MDP,
    bfs_distances,
    build_mrc_corridor,
    destroyed_mass,
    reachable_set,
)

K_MAX = 5
M_MIN, M_MAX = 2, 5

# Action labels used by Stage-1's corridor and their canonical order.
ACTION_LIST = ["a_decoy", "a_safe", "fwd", "recover"]
ACTION_INDEX = {a: i for i, a in enumerate(ACTION_LIST)}
N_ACTIONS = len(ACTION_LIST)

# State token labels.  All corridor states are labelled with one of the
# following + the cell index for c_i.  We map them into one-hot positions
# in a fixed-length vector below.
N_STATE_SLOTS = 3 + (M_MAX + K_MAX)  # s0, trap, absorb, c1..c_{M_MAX+K_MAX}


def state_index(state: Any) -> int:
    if state == "s0":
        return 0
    if state == "trap":
        return 1
    if state == "absorb":
        return 2
    if isinstance(state, str) and state.startswith("c"):
        i = int(state[1:])  # 1-indexed corridor cell
        return 2 + i  # so c1 -> 3, c2 -> 4, ...
    raise ValueError(f"unknown state token {state!r}")


MODES = ("irreversible", "reversible")
MODE_INDEX = {m: i for i, m in enumerate(MODES)}


@dataclass
class InstanceParams:
    k: int
    m: int
    r_d: float
    r_g: float
    gamma: float

    def safe_value(self) -> float:
        """V_safe = sum_{i=m}^{m+k-1} gamma^i * r_g."""
        return float(sum((self.gamma ** i) * self.r_g
                         for i in range(self.m, self.m + self.k)))


def make_random_instance(seed: int, gamma: float = 0.9,
                         max_tries: int = 200) -> InstanceParams:
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        k = int(rng.integers(1, K_MAX + 1))           # 1..K_MAX
        m = int(rng.integers(M_MIN, M_MAX + 1))       # M_MIN..M_MAX
        r_d = float(rng.uniform(0.5, 2.0))
        r_g = float(rng.uniform(1.5, 4.0))
        params = InstanceParams(k=k, m=m, r_d=r_d, r_g=r_g, gamma=gamma)
        # Well-posedness: oracle MRC must strictly beat oblivious.
        # V_safe > r_d means the lambda=1 charge tips a_safe over a_decoy.
        if params.safe_value() > params.r_d + 1e-6:
            return params
    raise RuntimeError(f"failed to generate well-posed instance for seed {seed}")


def build_mdp_from_params(params: InstanceParams, mode: str) -> MDP:
    return build_mrc_corridor(
        k=params.k, H=params.m, m=params.m,
        r_d=params.r_d, r_g=params.r_g, gamma=params.gamma, mode=mode,
    )


def horizon_for_params(params: InstanceParams) -> int:
    """Matches Stage-1's choice: H = m so the oblivious planner sees the
    decoy reward (depth 1 through a_decoy) but cannot peek through the
    safe corridor to any target (first target is at depth m, but its
    reward sits on the outgoing edge at depth m+1).
    """
    return params.m


# --------------------------------------------------------------------- #
# Feature extractor                                                      #
# --------------------------------------------------------------------- #


def featurise(params: InstanceParams, state: Any, action: str, mode: str) -> np.ndarray:
    k_oh = np.zeros(K_MAX, dtype=np.float32)
    k_oh[params.k - 1] = 1.0  # k in 1..K_MAX, slot index 0..K_MAX-1
    m_oh = np.zeros(M_MAX, dtype=np.float32)
    m_oh[params.m - 1] = 1.0  # m in 1..M_MAX (we only use M_MIN..M_MAX)

    scalars = np.array([params.r_d, params.r_g, params.gamma], dtype=np.float32)

    state_oh = np.zeros(N_STATE_SLOTS, dtype=np.float32)
    try:
        state_oh[state_index(state)] = 1.0
    except ValueError:
        # Terminal sub-tokens (none in this MDP) -- leave zeros.
        pass

    action_oh = np.zeros(N_ACTIONS, dtype=np.float32)
    if action in ACTION_INDEX:
        action_oh[ACTION_INDEX[action]] = 1.0

    mode_oh = np.zeros(len(MODES), dtype=np.float32)
    mode_oh[MODE_INDEX[mode]] = 1.0

    return np.concatenate([k_oh, m_oh, scalars, state_oh, action_oh, mode_oh])


def feature_dim() -> int:
    return K_MAX + M_MAX + 3 + N_STATE_SLOTS + N_ACTIONS + len(MODES)


# --------------------------------------------------------------------- #
# Training-data generation                                               #
# --------------------------------------------------------------------- #


@dataclass
class Sample:
    params: InstanceParams
    state: Any
    action: str
    mode: str
    Dw: float


def collect_samples(seeds: List[int], gamma: float = 0.9) -> List[Sample]:
    out: List[Sample] = []
    for seed in seeds:
        params = make_random_instance(seed=seed, gamma=gamma)
        for mode in MODES:
            mdp = build_mdp_from_params(params, mode=mode)
            for s in mdp.states:
                for a in mdp.actions.get(s, []):
                    Dw = float(destroyed_mass(mdp, s, a))
                    out.append(Sample(
                        params=params, state=s, action=a, mode=mode, Dw=Dw,
                    ))
    return out


def samples_to_tensors(samples: List[Sample]):
    import torch  # local import keeps numpy-only paths importable
    X = np.zeros((len(samples), feature_dim()), dtype=np.float32)
    y = np.zeros((len(samples),), dtype=np.float32)
    for i, s in enumerate(samples):
        X[i] = featurise(s.params, s.state, s.action, s.mode)
        y[i] = s.Dw
    return torch.from_numpy(X), torch.from_numpy(y)


__all__ = [
    "InstanceParams", "Sample",
    "K_MAX", "M_MIN", "M_MAX",
    "ACTION_LIST", "ACTION_INDEX", "N_ACTIONS", "N_STATE_SLOTS", "MODES",
    "make_random_instance", "build_mdp_from_params", "horizon_for_params",
    "featurise", "feature_dim", "collect_samples", "samples_to_tensors",
    "state_index",
    "MDP", "destroyed_mass", "bfs_distances", "reachable_set",
]

"""
phase1_shaped.py
================

Gymnasium wrapper that subtracts lambda * D_w(s, a) from the env reward,
where D_w(s, a) is supplied by ANY callable matching the signature

    dw_source(env: SmashGateEnv, action: int) -> float

For the three agents we plug in three different sources:

    oracle_mrc  -> env.oracle_dw(action)      (Stage-1 exact)
    learned_mrc -> LearnedDwLookup(model)     (Phase 1 MLP)
    reward_only -> a constant 0 source         (no shaping; behaves like
                                                the bare env)

This is the ONLY place the three agents differ -- the env, the PPO
hyperparameters, the network architecture, the eval protocol are all
shared. That keeps the test honest about isolating "the D_w source" as
the moving variable.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym

from phase1_env import SmashGateEnv


class ShapedRewardWrapper(gym.Wrapper):
    """Reward shaping by -lambda * dw_source(env, action), per step."""

    def __init__(
        self,
        env: SmashGateEnv,
        dw_source: Callable[[SmashGateEnv, int], float],
        lam: float = 1.0,
    ):
        super().__init__(env)
        self.dw_source = dw_source
        self.lam = float(lam)

    def step(self, action):
        # Query the D_w source BEFORE stepping (state is s, action is a).
        dw = float(self.dw_source(self.env.unwrapped, int(action)))
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = float(reward) - self.lam * dw
        info = dict(info)
        info["raw_reward"] = float(reward)
        info["dw_charge"] = dw
        info["shaped_reward"] = shaped
        return obs, shaped, terminated, truncated, info


def zero_dw_source(env, action: int) -> float:
    """No-op D_w source for reward_only. Returns 0 for any (state, action)."""
    return 0.0


def oracle_dw_source(env: SmashGateEnv, action: int) -> float:
    """Stage-1 exact destroyed_mass via env's precomputed table."""
    return env.oracle_dw(action)

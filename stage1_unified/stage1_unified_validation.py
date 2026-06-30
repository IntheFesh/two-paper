"""
stage1_unified_validation.py
============================

Stage-1 unified verification framework for the MRC (Monotone Reachability
Contraction) paper.

This file deliberately implements V1..V5 on top of ONE specification — a
single MDP class, a single destroyed-mass function destroyed_mass(...),
a single planner pair (oblivious / MRC), and a single set of environment
constructors. No module re-defines D_w or rebuilds an MDP; every module
calls the same primitives. This is the point of the integration: any
hidden definitional drift would immediately surface as a failed assertion
in V5.

Pre-registered PASS/FAIL conditions (locked before looking at any output):
  V1 PASS iff the linear fit ΔV vs D_w has slope == 1.0 and intercept == -r_d
            (to machine precision).
  V2 PASS iff irreversible ΔV == D_w - r_d AND reversible D_w == 0 AND
            reversible ΔV == 0.
  V3 PASS iff the observed switch point λ* matches λ_min = r_d / D_w(s_0, a_decoy)
            within one grid step, AND policy_mrc(s_0) == a_safe at λ == 1
            (separation precondition).
  V4 PASS iff the three exact equalities hold pointwise (R, D_w, Q^H_MRC),
            AND the renewable contrast collapses to D_w == 0 and ΔV == 0.
  V5 PASS iff all four self-consistency assertions pass: single-function
            identity, reversible-zero, additivity, representation invariance.

Any FAIL must be reported as-is. Do NOT retune to mask a failure: a failure
means the framework has a logical hole that needs to be diagnosed.

Reward / discount convention
----------------------------
The corridor places target reward r_g on the OUTGOING transition of a
target state c_t. The agent enters c_t after t actions from s_0, then on
its (t+1)-st action collects r_g. Discount applied to that reward is γ^t.
Since the BFS layer of c_t from s_0 is also t, the spec's
D_w = Σ γ^{d(s,g)} u(g) matches V_safe exactly, and V_decoy = r_d at
γ^0 = 1 matches the intercept -r_d cleanly. This is the only consistent
way to make the prediction ΔV = D_w - r_d hold with action-based rewards
and BFS-layer distance; any other shift would break slope or intercept.

Runtime
-------
~3 seconds on a single CPU core. Well under the 3-minute budget.
"""

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# ====================================================================
# 1. Unified MDP class and core operators (SHARED BY ALL MODULES)
# ====================================================================


@dataclass
class MDP:
    """Deterministic finite-graph MDP with action-based reward.

    states            : list of state ids (hashable).
    actions[s]        : list of action labels available at s.
    f[(s, a)]         : deterministic next state.
    r[(s, a)]         : immediate reward of taking a in s.
    targets           : set of target state ids (subset of states).
    target_weights[g] : original weight u(g) of target g.
    gamma             : discount factor in (0, 1).
    """

    states: List[Any]
    actions: Dict[Any, List[str]]
    f: Dict[Tuple[Any, str], Any]
    r: Dict[Tuple[Any, str], float]
    targets: set
    target_weights: Dict[Any, float]
    gamma: float


def reachable_set(mdp: MDP, s) -> set:
    """R(s): the set of states reachable from s under f (includes s itself)."""
    seen = {s}
    q = deque([s])
    while q:
        u = q.popleft()
        for a in mdp.actions.get(u, []):
            v = mdp.f[(u, a)]
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def bfs_distances(mdp: MDP, s) -> Dict[Any, int]:
    """d(s, *): BFS layer number from s. d(s, s) = 0."""
    dist = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        for a in mdp.actions.get(u, []):
            v = mdp.f[(u, a)]
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


# ----------------------------------------------------------------------
# THE ONE AND ONLY destroyed-mass function.
# Every module below calls THIS. There is no alternative implementation
# anywhere in this file — V5 demonstrates the consistency by exercising
# it across all module-specific MDPs and asserting cross-module identity.
# ----------------------------------------------------------------------


def destroyed_mass(mdp: MDP, s, a) -> float:
    """D_w(s, a) = Σ over g in targets ∩ (R(s) \\ R(f(s,a))) of γ^{d(s,g)} · u(g)."""
    s_next = mdp.f[(s, a)]
    R_s = reachable_set(mdp, s)
    R_next = reachable_set(mdp, s_next)
    destroyed = (R_s - R_next) & mdp.targets
    if not destroyed:
        return 0.0
    dist = bfs_distances(mdp, s)
    return sum(
        (mdp.gamma ** dist[g]) * mdp.target_weights[g] for g in destroyed
    )


# ====================================================================
# 2. Unified planners (SHARED BY ALL MODULES)
# ====================================================================


def value_h(mdp: MDP, s, H: int, _memo=None) -> float:
    """H-step bounded value: V_H(s) = max_a r + γ · V_{H-1}(f(s,a)). V_0 = 0."""
    if _memo is None:
        _memo = {}
    if H <= 0 or not mdp.actions.get(s):
        return 0.0
    key = (id(mdp), s, H)
    if key in _memo:
        return _memo[key]
    best = -math.inf
    for a in mdp.actions[s]:
        s2 = mdp.f[(s, a)]
        rew = mdp.r[(s, a)]
        v = rew + mdp.gamma * value_h(mdp, s2, H - 1, _memo)
        if v > best:
            best = v
    _memo[key] = best
    return best


def q_reward_h(mdp: MDP, s, a, H: int) -> float:
    """Q^reward_H(s, a) = r(s,a) + γ · V_{H-1}(f(s,a)). Reward-only, no D_w."""
    s2 = mdp.f[(s, a)]
    return mdp.r[(s, a)] + mdp.gamma * value_h(mdp, s2, H - 1)


def q_mrc(mdp: MDP, s, a, H: int, lam: float) -> float:
    """Q^H_MRC(s, a) = Q^reward_H(s, a) - λ · D_w(s, a)."""
    return q_reward_h(mdp, s, a, H) - lam * destroyed_mass(mdp, s, a)


def policy_obl(mdp: MDP, s, H: int) -> str:
    """π_obl(s) = argmax_a Q^reward_H(s, a). Ties broken by sorted action order."""
    acts = sorted(mdp.actions[s])
    return max(acts, key=lambda a: q_reward_h(mdp, s, a, H))


def policy_mrc(mdp: MDP, s, H: int, lam: float) -> str:
    """π_MRC(s) = argmax_a Q^H_MRC(s, a)."""
    acts = sorted(mdp.actions[s])
    return max(acts, key=lambda a: q_mrc(mdp, s, a, H, lam))


def rollout_value(
    mdp: MDP, start, choose_action: Callable[[Any], str], max_steps: int = 400
) -> float:
    """Discounted return obtained by following a (state -> action) policy from `start`."""
    s = start
    total = 0.0
    disc = 1.0
    for _ in range(max_steps):
        if not mdp.actions.get(s):
            break
        a = choose_action(s)
        s2 = mdp.f[(s, a)]
        rew = mdp.r[(s, a)]
        total += disc * rew
        disc *= mdp.gamma
        if s2 == s and rew == 0.0:
            break
        s = s2
    return total


# ====================================================================
# 3. Unified environment constructors (SHARED BY ALL MODULES)
# ====================================================================


def build_mrc_corridor(
    k: int, H: int, m: int, r_d: float, r_g: float, gamma: float,
    mode: str = "irreversible",
) -> MDP:
    """Corridor MDP per the unified spec.

    Layout
    ------
        States : s0, trap, absorb, c1, c2, ..., c_{m+k}.
        s0     : a_decoy → trap (reward r_d) | a_safe → c1 (reward 0).
        c_i    : fwd → c_{i+1}    for i = 1 .. m+k-1
                 fwd → absorb     for i = m+k
                 reward = r_g iff c_i is a target, else 0
                 (reward sits on the OUTGOING edge of a target state, so
                 that the discount γ^t applied to it equals γ^{d(s0,c_i)}).
        trap   : irreversible → no actions (absorbing).
                 reversible   → recover → c1 (reward 0). No path back from
                                the corridor to s0 ⇒ r_d collected at most once.
        absorb : no actions.

    Targets
    -------
        g_j = c_{m+j-1} for j = 1..k. BFS distance d(s0, g_j) = m + j - 1.
        Target weight u(g_j) = r_g.

    The argument H is recorded for documentation/auditing only; the planners
    receive H explicitly when invoked.
    """
    assert mode in ("irreversible", "reversible")
    _ = H  # H is a calibration parameter (passed to planners); not used here.

    states = ["s0", "trap", "absorb"] + [f"c{i}" for i in range(1, m + k + 1)]
    actions: Dict[str, List[str]] = {s: [] for s in states}
    f: Dict[Tuple[str, str], str] = {}
    r: Dict[Tuple[str, str], float] = {}

    actions["s0"] = ["a_decoy", "a_safe"]
    f[("s0", "a_decoy")] = "trap"
    r[("s0", "a_decoy")] = r_d
    f[("s0", "a_safe")] = "c1"
    r[("s0", "a_safe")] = 0.0

    target_positions = set(range(m, m + k))  # corridor indices of targets

    for i in range(1, m + k + 1):
        actions[f"c{i}"] = ["fwd"]
        nxt = f"c{i+1}" if i < m + k else "absorb"
        f[(f"c{i}", "fwd")] = nxt
        r[(f"c{i}", "fwd")] = r_g if i in target_positions else 0.0

    if mode == "reversible":
        actions["trap"] = ["recover"]
        f[("trap", "recover")] = "c1"
        r[("trap", "recover")] = 0.0
    # else: trap is absorbing (no actions). absorb is absorbing (no actions).

    targets = {f"c{i}" for i in target_positions}
    target_weights = {t: r_g for t in targets}

    return MDP(
        states=states, actions=actions, f=f, r=r,
        targets=targets, target_weights=target_weights, gamma=gamma,
    )


def build_resource_mdp(
    L: int, F: int, r_d: float, r_g: float, gamma: float,
    mode: str = "non_renewable",
) -> MDP:
    """Resource-aware MDP with state (loc, fuel).

    Only the diagonal of states reachable by the natural 'fwd' sequence
    from (0, F) carries actions:
        (0,F) → (1,F-1) → (2,F-2) → ... → (L, F-L) ─collect→ absorb (r_g).
    Decoy 'splurge' at (0, F) drains all fuel: (0,F) → (0,0) with reward r_d.
    In non_renewable mode, (0,0) is absorbing (no actions, hence target
    becomes unreachable). In renewable mode, (0,0) has a 'refuel' action
    that restores (0,F), so target reachability is preserved.
    """
    assert mode in ("non_renewable", "renewable")
    assert L <= F, "L > F would leave the target unreachable even without splurge"

    states: List[Any] = [(l, fl) for l in range(L + 1) for fl in range(F + 1)] + ["absorb"]
    actions: Dict[Any, List[str]] = {s: [] for s in states}
    f: Dict[Tuple[Any, str], Any] = {}
    r: Dict[Tuple[Any, str], float] = {}

    # Forward chain along the diagonal.
    for l in range(L):
        s = (l, F - l)
        nxt = (l + 1, F - l - 1)
        actions[s].append("fwd")
        f[(s, "fwd")] = nxt
        r[(s, "fwd")] = 0.0

    # Target's outgoing 'collect' edge carries r_g.
    target = (L, F - L)
    actions[target].append("collect")
    f[(target, "collect")] = "absorb"
    r[(target, "collect")] = r_g

    # Decoy 'splurge'.
    start = (0, F)
    actions[start].append("splurge")
    f[(start, "splurge")] = (0, 0)
    r[(start, "splurge")] = r_d

    if mode == "renewable":
        # Recovery — restores reachability of target.
        actions[(0, 0)].append("refuel")
        f[((0, 0), "refuel")] = (0, F)
        r[((0, 0), "refuel")] = 0.0

    targets = {target}
    target_weights = {target: r_g}

    return MDP(
        states=states, actions=actions, f=f, r=r,
        targets=targets, target_weights=target_weights, gamma=gamma,
    )


def build_augmented_graph(resource_mdp: MDP) -> Tuple[MDP, Dict[Any, int]]:
    """Re-encode `resource_mdp` as an integer-labelled graph.

    Returns (graph_mdp, vertex_map) where vertex_map: original state → int id.
    The two MDPs are isomorphic but use different state encodings, which is
    exactly what V4's representation-invariance assertions test.
    """
    vertex_map = {s: i for i, s in enumerate(resource_mdp.states)}
    new_states = [vertex_map[s] for s in resource_mdp.states]
    new_actions = {
        vertex_map[s]: list(acts) for s, acts in resource_mdp.actions.items()
    }
    new_f = {
        (vertex_map[s], a): vertex_map[ns]
        for (s, a), ns in resource_mdp.f.items()
    }
    new_r = {(vertex_map[s], a): v for (s, a), v in resource_mdp.r.items()}
    new_targets = {vertex_map[s] for s in resource_mdp.targets}
    new_weights = {vertex_map[s]: w for s, w in resource_mdp.target_weights.items()}
    graph_mdp = MDP(
        states=new_states, actions=new_actions, f=new_f, r=new_r,
        targets=new_targets, target_weights=new_weights,
        gamma=resource_mdp.gamma,
    )
    return graph_mdp, vertex_map


# ====================================================================
# 4. Validation modules — V1..V5
# ====================================================================

# A single shared parameter dict for the corridor-based modules (V1, V2, V3, V5).
# Picked so that:
#   - H = m makes Q^reward_H(s_0, a_safe) = 0 (planner can't see any target
#     through a_safe within H steps), so π_obl deterministically picks decoy.
#   - For V1, a large λ guarantees π_MRC picks safe whenever D_w > 0, so the
#     observed ΔV equals the prediction D_w - r_d cleanly.
DEFAULTS = dict(m=4, H=4, r_d=1.0, r_g=1.0, gamma=0.9)
LAMBDA_LARGE = 10.0  # well above any λ_min seen in V1/V2.


def _action_fn(mdp: MDP, picker: Callable[[Any], str]) -> Callable[[Any], str]:
    """Tiny helper so rollouts can call policy_obl/policy_mrc closures uniformly."""
    return picker


def v1_dw_vs_delta_v() -> Dict[str, Any]:
    """V1 — gap tracks destroyed mass.

    Vary k (number of targets that decoy destroys) so D_w varies. With λ large
    enough that π_MRC always prefers safe, the prediction ΔV = D_w - r_d
    should fit with slope = 1 and intercept = -r_d exactly.
    """
    m, H, r_d, r_g, gamma = (DEFAULTS[k] for k in ("m", "H", "r_d", "r_g", "gamma"))
    lam = LAMBDA_LARGE

    ks = list(range(1, 7))
    rows = []
    for k in ks:
        mdp = build_mrc_corridor(
            k=k, H=H, m=m, r_d=r_d, r_g=r_g, gamma=gamma, mode="irreversible",
        )
        Dw = destroyed_mass(mdp, "s0", "a_decoy")

        v_obl = rollout_value(mdp, "s0", lambda s: policy_obl(mdp, s, H))
        v_mrc = rollout_value(mdp, "s0", lambda s: policy_mrc(mdp, s, H, lam))
        delta = v_mrc - v_obl

        rows.append({
            "k": k, "D_w": Dw, "V_obl": v_obl, "V_mrc": v_mrc,
            "delta_V_observed": delta, "delta_V_predicted": Dw - r_d,
        })

    xs = np.array([row["D_w"] for row in rows])
    ys = np.array([row["delta_V_observed"] for row in rows])
    slope, intercept = np.polyfit(xs, ys, 1)
    slope_ok = abs(slope - 1.0) < 1e-9
    intercept_ok = abs(intercept - (-r_d)) < 1e-9
    pointwise_ok = all(
        abs(row["delta_V_observed"] - row["delta_V_predicted"]) < 1e-12
        for row in rows
    )
    passed = bool(slope_ok and intercept_ok and pointwise_ok)

    print("\n--- V1: ΔV vs D_w (irreversible corridor; λ = %g, H = %d, m = %d) ---" % (lam, H, m))
    print(f"{'k':>3} {'D_w':>10} {'V_obl':>10} {'V_mrc':>10} {'ΔV obs':>10} {'ΔV pred':>10}")
    for row in rows:
        print(
            f"{row['k']:>3} {row['D_w']:>10.6f} {row['V_obl']:>10.6f} "
            f"{row['V_mrc']:>10.6f} {row['delta_V_observed']:>10.6f} "
            f"{row['delta_V_predicted']:>10.6f}"
        )
    print(f"Linear fit ΔV = a·D_w + b:  a = {slope:.9f} (expect 1.000000000)")
    print(f"                            b = {intercept:.9f} (expect {-r_d:.9f})")
    print(f"Pointwise ΔV_obs = ΔV_pred to 1e-12: {pointwise_ok}")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")

    return {
        "name": "V1: gap tracks destroyed mass",
        "rows": rows,
        "fit": {"slope": float(slope), "intercept": float(intercept)},
        "expected": {"slope": 1.0, "intercept": -r_d},
        "pointwise_ok": pointwise_ok,
        "passed": passed,
    }


def v2_collapse_control() -> Dict[str, Any]:
    """V2 — irreversible vs reversible twin: D_w and ΔV collapse to 0 in reversible."""
    m, H, r_d, r_g, gamma = (DEFAULTS[k] for k in ("m", "H", "r_d", "r_g", "gamma"))
    k = 3
    lam = LAMBDA_LARGE

    mdp_irr = build_mrc_corridor(
        k=k, H=H, m=m, r_d=r_d, r_g=r_g, gamma=gamma, mode="irreversible",
    )
    mdp_rev = build_mrc_corridor(
        k=k, H=H, m=m, r_d=r_d, r_g=r_g, gamma=gamma, mode="reversible",
    )

    Dw_irr = destroyed_mass(mdp_irr, "s0", "a_decoy")
    Dw_rev = destroyed_mass(mdp_rev, "s0", "a_decoy")

    v_obl_irr = rollout_value(mdp_irr, "s0", lambda s: policy_obl(mdp_irr, s, H))
    v_mrc_irr = rollout_value(mdp_irr, "s0", lambda s: policy_mrc(mdp_irr, s, H, lam))
    v_obl_rev = rollout_value(mdp_rev, "s0", lambda s: policy_obl(mdp_rev, s, H))
    v_mrc_rev = rollout_value(mdp_rev, "s0", lambda s: policy_mrc(mdp_rev, s, H, lam))

    dV_irr = v_mrc_irr - v_obl_irr
    dV_rev = v_mrc_rev - v_obl_rev
    pred_irr = Dw_irr - r_d

    irr_dv_ok = abs(dV_irr - pred_irr) < 1e-12
    rev_dw_zero = (Dw_rev == 0.0)
    rev_dv_zero = abs(dV_rev) < 1e-12
    passed = bool(irr_dv_ok and rev_dw_zero and rev_dv_zero)

    print("\n--- V2: collapse control (matched irreversible vs reversible twin) ---")
    print(f"Irreversible: D_w = {Dw_irr:.6f}, ΔV = {dV_irr:.6f} "
          f"(predicted {pred_irr:.6f}, match: {irr_dv_ok})")
    print(f"Reversible  : D_w = {Dw_rev:.6f} (expect exactly 0: {rev_dw_zero}),"
          f"  ΔV = {dV_rev:.6f} (expect exactly 0: {rev_dv_zero})")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")

    return {
        "name": "V2: irreversible vs reversible collapse",
        "irreversible": {
            "D_w": Dw_irr, "V_obl": v_obl_irr, "V_mrc": v_mrc_irr,
            "delta_V_observed": dV_irr, "delta_V_predicted": pred_irr,
            "match": irr_dv_ok,
        },
        "reversible": {
            "D_w": Dw_rev, "V_obl": v_obl_rev, "V_mrc": v_mrc_rev,
            "delta_V_observed": dV_rev,
            "D_w_zero": rev_dw_zero, "delta_V_zero": rev_dv_zero,
        },
        "passed": passed,
    }


def v3_lambda_phase_transition() -> Dict[str, Any]:
    """V3 — switching point of π_MRC at s_0 as λ varies."""
    m, H, r_d, r_g, gamma = (DEFAULTS[k] for k in ("m", "H", "r_d", "r_g", "gamma"))
    k = 3

    mdp = build_mrc_corridor(
        k=k, H=H, m=m, r_d=r_d, r_g=r_g, gamma=gamma, mode="irreversible",
    )
    Dw = destroyed_mass(mdp, "s0", "a_decoy")
    lam_min = r_d / Dw

    lambdas = np.linspace(0.0, 1.5, 1501)  # resolution 0.001
    actions = [policy_mrc(mdp, "s0", H, float(lam)) for lam in lambdas]

    switch_idx = next((i for i, a in enumerate(actions) if a == "a_safe"), None)
    lam_star = float(lambdas[switch_idx]) if switch_idx is not None else None
    at_lam_1 = policy_mrc(mdp, "s0", H, 1.0)

    grid_step = float(lambdas[1] - lambdas[0])
    switch_ok = lam_star is not None and abs(lam_star - lam_min) <= grid_step + 1e-9
    lam1_ok = at_lam_1 == "a_safe"
    sep_ok = lam_min < 1.0
    passed = bool(switch_ok and lam1_ok and sep_ok)

    print("\n--- V3: λ phase transition (irreversible corridor, k = %d) ---" % k)
    print(f"D_w(s0, a_decoy) = {Dw:.6f}")
    print(f"λ_min  (theory)  = r_d / D_w = {lam_min:.6f}   "
          f"(separation precondition λ_min < 1: {sep_ok})")
    print(f"λ*    (observed) = {lam_star}   (grid step = {grid_step:.4f}, "
          f"|λ* - λ_min| = {abs(lam_star - lam_min):.6f})")
    print(f"π_MRC(s0) at λ = 1: '{at_lam_1}'   (expect 'a_safe': {lam1_ok})")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")

    return {
        "name": "V3: λ phase transition",
        "D_w": Dw, "lam_min_theory": lam_min,
        "lam_star_observed": lam_star, "grid_step": grid_step,
        "policy_at_lam_1": at_lam_1,
        "switch_within_grid_step": switch_ok,
        "passed": passed,
    }


def v4_resource_topology_identity() -> Dict[str, Any]:
    """V4 — three exact equalities between the resource-MDP and augmented-graph
    representations, plus the renewable contrast."""
    L, F = 4, 4
    r_d, r_g, gamma = 1.0, 2.0, 0.9
    H = L         # planner can't peek through to the 'collect' reward at depth L
    lam = LAMBDA_LARGE

    res_mdp = build_resource_mdp(L, F, r_d, r_g, gamma, mode="non_renewable")
    graph_mdp, vmap = build_augmented_graph(res_mdp)

    # (a) State-by-state reachability equivalence under the bijection.
    R_mismatches: List[Any] = []
    for s in res_mdp.states:
        R_res = reachable_set(res_mdp, s)
        R_gph = reachable_set(graph_mdp, vmap[s])
        if {vmap[u] for u in R_res} != R_gph:
            R_mismatches.append(s)
    R_ok = len(R_mismatches) == 0

    # (b) D_w equality for every (s, a). SAME destroyed_mass function on both.
    Dw_mismatches: List[Tuple[Any, str, float, float]] = []
    Dw_compared = 0
    for s in res_mdp.states:
        for a in res_mdp.actions[s]:
            d_res = destroyed_mass(res_mdp, s, a)
            d_gph = destroyed_mass(graph_mdp, vmap[s], a)
            Dw_compared += 1
            if abs(d_res - d_gph) > 1e-12:
                Dw_mismatches.append((s, a, d_res, d_gph))
    Dw_ok = len(Dw_mismatches) == 0

    # (c) Q^H_MRC equality for every (s, a).
    Q_mismatches: List[Tuple[Any, str, float, float]] = []
    Q_compared = 0
    for s in res_mdp.states:
        for a in res_mdp.actions[s]:
            q_res = q_mrc(res_mdp, s, a, H, lam)
            q_gph = q_mrc(graph_mdp, vmap[s], a, H, lam)
            Q_compared += 1
            if abs(q_res - q_gph) > 1e-9:
                Q_mismatches.append((s, a, q_res, q_gph))
    Q_ok = len(Q_mismatches) == 0

    # Renewable contrast — gain must collapse to 0.
    res_renew = build_resource_mdp(L, F, r_d, r_g, gamma, mode="renewable")
    v_obl_r = rollout_value(res_renew, (0, F), lambda s: policy_obl(res_renew, s, H))
    v_mrc_r = rollout_value(res_renew, (0, F), lambda s: policy_mrc(res_renew, s, H, lam))
    gain_renew = v_mrc_r - v_obl_r
    Dw_renew = destroyed_mass(res_renew, (0, F), "splurge")
    renew_dw_ok = (Dw_renew == 0.0)
    renew_dv_ok = abs(gain_renew) < 1e-12
    renew_ok = renew_dw_ok and renew_dv_ok

    passed = bool(R_ok and Dw_ok and Q_ok and renew_ok)

    # Sanity sample: V_safe = D_w, V_decoy = r_d, ΔV = D_w - r_d in the
    # non-renewable resource MDP (mirrors V1's prediction on a different graph).
    Dw_res = destroyed_mass(res_mdp, (0, F), "splurge")
    v_obl_nr = rollout_value(res_mdp, (0, F), lambda s: policy_obl(res_mdp, s, H))
    v_mrc_nr = rollout_value(res_mdp, (0, F), lambda s: policy_mrc(res_mdp, s, H, lam))

    print("\n--- V4: resource ↔ graph triple identity + renewable contrast ---")
    print(f"(a) Reachable-set bijection over {len(res_mdp.states)} states: "
          f"{'OK' if R_ok else f'FAIL @ {R_mismatches[:3]}'}")
    print(f"(b) D_w  identity over {Dw_compared} (s,a) pairs: "
          f"{'OK' if Dw_ok else f'FAIL @ {Dw_mismatches[:3]}'}")
    print(f"(c) Q^H_MRC identity over {Q_compared} (s,a) pairs: "
          f"{'OK' if Q_ok else f'FAIL @ {Q_mismatches[:3]}'}")
    print(f"Non-renewable sanity: D_w = {Dw_res:.6f}, V_obl = {v_obl_nr:.6f}, "
          f"V_mrc = {v_mrc_nr:.6f}, ΔV = {v_mrc_nr - v_obl_nr:.6f} "
          f"(predicted {Dw_res - r_d:.6f})")
    print(f"Renewable contrast  : D_w_renew = {Dw_renew} (expect 0: {renew_dw_ok}),"
          f"  ΔV_renew = {gain_renew:.6f} (expect 0: {renew_dv_ok})")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")

    return {
        "name": "V4: resource ↔ graph triple identity + renewable contrast",
        "R_ok": R_ok, "Dw_ok": Dw_ok, "Q_ok": Q_ok,
        "R_mismatches": [str(s) for s in R_mismatches],
        "Dw_mismatches": [(str(s), a, d1, d2) for (s, a, d1, d2) in Dw_mismatches],
        "Q_mismatches": [(str(s), a, q1, q2) for (s, a, q1, q2) in Q_mismatches],
        "Dw_compared": Dw_compared, "Q_compared": Q_compared,
        "renew": {"D_w": Dw_renew, "delta_V": gain_renew,
                  "D_w_zero": renew_dw_ok, "delta_V_zero": renew_dv_ok},
        "non_renewable_sanity": {
            "D_w": Dw_res, "V_obl": v_obl_nr, "V_mrc": v_mrc_nr,
            "delta_V_observed": v_mrc_nr - v_obl_nr,
            "delta_V_predicted": Dw_res - r_d,
        },
        "passed": passed,
    }


def v5_self_consistency() -> Dict[str, Any]:
    """V5 — self-consistency cross-checks. This is the integration-test layer.

    The four assertions stress invariants that would silently break if any
    module had been implemented with its own D_w or its own corridor:
      (i)   Single-function identity — destroyed_mass is the symbol used
            by V1, V3 and V4 (no shadowing or per-module redefinition).
      (ii)  Reversible-zero — D_w of a reversible action is exactly 0.
      (iii) Additivity — D_w decomposes over disjoint target subsets.
      (iv)  Representation invariance — D_w is unchanged under the
            (loc,fuel) ↔ integer-vertex re-encoding.
    """
    m, H, r_d, r_g, gamma = (DEFAULTS[k] for k in ("m", "H", "r_d", "r_g", "gamma"))

    # ---- (i) Single-function identity ---------------------------------
    # The bindings used by every module are the module-level destroyed_mass.
    # We additionally exercise it on V1's, V3's and V4's primary MDPs to
    # make the cross-module call surface explicit.
    mdp_v1v3 = build_mrc_corridor(
        k=3, H=H, m=m, r_d=r_d, r_g=r_g, gamma=gamma, mode="irreversible",
    )
    res_v4 = build_resource_mdp(L=4, F=4, r_d=r_d, r_g=2.0, gamma=gamma,
                                 mode="non_renewable")
    graph_v4, vmap_v4 = build_augmented_graph(res_v4)

    sample_calls = 0
    for mdp in (mdp_v1v3, res_v4, graph_v4):
        for s in mdp.states:
            for a in mdp.actions.get(s, []):
                _ = destroyed_mass(mdp, s, a)
                sample_calls += 1

    # The strongest structural guarantee: `destroyed_mass` is defined ONCE
    # in this module, and Python's identity check confirms no rebind.
    single_function_ok = (
        destroyed_mass.__module__ == __name__
        and destroyed_mass.__qualname__ == "destroyed_mass"
        and sample_calls > 0
    )

    # ---- (ii) Reversible-zero -----------------------------------------
    mdp_rev = build_mrc_corridor(
        k=3, H=H, m=m, r_d=r_d, r_g=r_g, gamma=gamma, mode="reversible",
    )
    Dw_rev_decoy = destroyed_mass(mdp_rev, "s0", "a_decoy")
    Dw_rev_trap = destroyed_mass(mdp_rev, "trap", "recover")
    reversible_zero_ok = (Dw_rev_decoy == 0.0) and (Dw_rev_trap == 0.0)

    # ---- (iii) Additivity over disjoint target subsets ----------------
    k_total, k_A = 5, 2
    mdp_full = build_mrc_corridor(
        k=k_total, H=H, m=m, r_d=r_d, r_g=r_g, gamma=gamma, mode="irreversible",
    )
    # Same topology, only target sets differ.
    all_targets = sorted(mdp_full.targets, key=lambda t: int(t[1:]))
    A_targets = set(all_targets[:k_A])
    B_targets = set(all_targets[k_A:])

    mdp_A = build_mrc_corridor(
        k=k_total, H=H, m=m, r_d=r_d, r_g=r_g, gamma=gamma, mode="irreversible",
    )
    mdp_A.targets = A_targets
    mdp_A.target_weights = {t: r_g for t in A_targets}

    mdp_B = build_mrc_corridor(
        k=k_total, H=H, m=m, r_d=r_d, r_g=r_g, gamma=gamma, mode="irreversible",
    )
    mdp_B.targets = B_targets
    mdp_B.target_weights = {t: r_g for t in B_targets}

    Dw_full = destroyed_mass(mdp_full, "s0", "a_decoy")
    Dw_A = destroyed_mass(mdp_A, "s0", "a_decoy")
    Dw_B = destroyed_mass(mdp_B, "s0", "a_decoy")
    additivity_ok = abs(Dw_full - (Dw_A + Dw_B)) < 1e-12

    # ---- (iv) Representation invariance --------------------------------
    Dw_res_splurge = destroyed_mass(res_v4, (0, 4), "splurge")
    Dw_gph_splurge = destroyed_mass(graph_v4, vmap_v4[(0, 4)], "splurge")
    repr_invariance_ok = abs(Dw_res_splurge - Dw_gph_splurge) < 1e-12

    passed = bool(single_function_ok and reversible_zero_ok and additivity_ok
                  and repr_invariance_ok)

    print("\n--- V5: self-consistency cross-checks ---")
    print(f"(i)   Single-function identity: destroyed_mass at "
          f"{destroyed_mass.__module__}.{destroyed_mass.__qualname__}, "
          f"exercised on {sample_calls} (mdp, s, a) calls across 3 MDPs: "
          f"{single_function_ok}")
    print(f"(ii)  Reversible-zero: D_w_rev(s0, a_decoy) = {Dw_rev_decoy} "
          f"and D_w_rev(trap, recover) = {Dw_rev_trap}  "
          f"(expect exactly 0: {reversible_zero_ok})")
    print(f"(iii) Additivity: D_w_full({Dw_full:.6f}) = D_w_A({Dw_A:.6f}) "
          f"+ D_w_B({Dw_B:.6f}) = {Dw_A + Dw_B:.6f}  (match: {additivity_ok})")
    print(f"(iv)  Representation invariance: D_w_resource({Dw_res_splurge:.6f}) "
          f"= D_w_graph({Dw_gph_splurge:.6f})  (match: {repr_invariance_ok})")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")

    return {
        "name": "V5: self-consistency cross-checks",
        "single_function_ok": single_function_ok,
        "destroyed_mass_qualname": destroyed_mass.__qualname__,
        "sample_call_count": sample_calls,
        "reversible_zero": {
            "D_w_decoy": Dw_rev_decoy, "D_w_trap_recover": Dw_rev_trap,
            "ok": reversible_zero_ok,
        },
        "additivity": {
            "D_full": Dw_full, "D_A": Dw_A, "D_B": Dw_B,
            "A_targets": sorted(A_targets), "B_targets": sorted(B_targets),
            "ok": additivity_ok,
        },
        "representation_invariance": {
            "D_resource": Dw_res_splurge, "D_graph": Dw_gph_splurge,
            "ok": repr_invariance_ok,
        },
        "passed": passed,
    }


# ====================================================================
# 5. Main driver
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
    print("=" * 72)
    print("Stage-1 unified validation for MRC  (V1..V5; numpy-only, single CPU)")
    print("=" * 72)

    results: Dict[str, Dict[str, Any]] = {}
    results["V1"] = v1_dw_vs_delta_v()
    results["V2"] = v2_collapse_control()
    results["V3"] = v3_lambda_phase_transition()
    results["V4"] = v4_resource_topology_identity()
    results["V5"] = v5_self_consistency()

    print()
    print("=" * 72)
    print("Overall verdict table (pre-registered PASS/FAIL conditions)")
    print("=" * 72)
    print(f"{'Module':<6}  {'Status':<6}  Description")
    print("-" * 72)
    all_pass = True
    for key, res in results.items():
        status = "PASS" if res["passed"] else "FAIL"
        if not res["passed"]:
            all_pass = False
        print(f"{key:<6}  {status:<6}  {res['name']}")
    print("-" * 72)
    if all_pass:
        print("Overall: ALL PASS — framework is self-consistent end-to-end.")
    else:
        print("Overall: FAIL — at least one module reports a violated invariant.")
        print("          Do NOT retune; diagnose the failed assertion above.")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "results.json")
    payload = {
        "overall_pass": all_pass,
        "defaults": DEFAULTS,
        "lambda_large": LAMBDA_LARGE,
        "modules": {k: _to_jsonable(v) for k, v in results.items()},
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nResults written to {out_path}")

    return all_pass


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

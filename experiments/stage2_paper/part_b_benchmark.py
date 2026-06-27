"""
Stage-2 / Part B — lightweight external benchmark: MiniGrid DoorKey-style task
with matched irreversible / reversible variants.

Why "DoorKey-style" without the minigrid package
-------------------------------------------------
The spec explicitly allows (and prefers) doing the benchmark on a small grid
with BFS-based exact planning rather than deep RL ("如果能用 planning(小 grid
上 BFS-based)就纯 CPU 更好，优先 CPU planning"). The structural mechanics —
agent on a 2D grid, key pickup, locked door, goal cell, decoy action with
matched-but-reversible twin — are reproduced here as a deterministic finite
MDP so that Stage-1's destroyed_mass / planners apply verbatim. Reusing the
same MDP class also lets us check Stage-1's PASS-side invariants on a new
graph topology, not just on the synthetic corridor.

Matched-control design
----------------------
The two variants share:
    - Identical grid layout, identical reward magnitudes (r_d, r_g, r_pickup,
      r_toggle), identical state space, identical action labels.
The ONLY structural difference:
    - In `irreversible`, trap is absorbing — after smash, the goal state is
      no longer reachable, so D_w(s_0, smash) > 0.
    - In `reversible`,   trap has a `recover` action that leads to the cell
      east of the start (not back to start itself). This keeps the goal
      reachable (D_w = 0) and prevents repeated decoy-collection by giving
      no return path to the smash cell.

Pre-registered Part B PASS conditions
-------------------------------------
    Irreversible:
        - V_MRC > V_reward_only (the contraction-aware planner avoids smash).
        - V_MRC ≈ V_oracle      (the avoidance approaches the optimal return).
        - D_w(s_0, smash) > 0.
    Reversible:
        - D_w(s_0, smash) == 0  (matched control: no destruction).
        - V_MRC ≈ V_reward_only ≈ V_oracle (no MRC advantage when D_w = 0).
Any FAIL must be reported as-is. In particular, if reversible-mode V_MRC
significantly exceeds V_reward_only, that is the signature of a bug, not a
result to keep — do not retune to mask it.

GPU budget
----------
None. Pure CPU planning, exact BFS over the state graph + finite-lookahead
+ value iteration. Runtime < 2 seconds on a single core. The 20 GPU-h
benchmark budget in the spec is unused.

Outputs (in experiments/stage2_paper/benchmark/)
------------------------------------------------
    benchmark_results.csv      — 3 planners × 2 variants returns / success table.
    benchmark_returns.pdf      — paired bar chart, the headline figure.
    benchmark_h_sweep.pdf      — H-sensitivity sanity sweep (CPU-cheap).
    benchmark_summary.json     — machine-readable PASS/FAIL + numbers.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Reuse Stage-1 primitives ------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "experiments", "stage1_unified"))

from stage1_unified_validation import (  # noqa: E402
    MDP,
    destroyed_mass,
    reachable_set,
    bfs_distances,
    q_reward_h,
    q_mrc,
    policy_obl,
    policy_mrc,
    rollout_value,
)

BENCH_DIR = os.path.join(_THIS_DIR, "benchmark")
os.makedirs(BENCH_DIR, exist_ok=True)

# Plot style consistent with Part A.
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# =====================================================================
# 1. DoorKey-style grid → MDP
# =====================================================================

DEFAULT_LAYOUT: Tuple[str, ...] = (
    "........G",   # row 0  — only (0,8) = G; the rest of row 0 is unreachable
    "########.",   # row 1  — single column-8 passage upward
    "S.K.D....",   # row 2  — main corridor: S, ., K, ., D, ., ., ., .
    "#########",   # row 3  — walled off
)


def build_doorkey_mdp(
    layout: Tuple[str, ...] = DEFAULT_LAYOUT,
    r_d: float = 0.3,
    r_g: float = 2.0,
    r_pickup: float = 0.01,
    r_toggle: float = 0.01,
    gamma: float = 0.9,
    mode: str = "irreversible",
) -> Tuple[MDP, Dict[str, Any]]:
    """Build a deterministic DoorKey-style MDP using Stage-1's MDP class.

    Cells:
        '.' floor, '#' wall, 'S' start, 'K' key cell, 'D' door (impassable
        when closed), 'G' goal cell. Movements are one-way (forward_E /
        forward_N) so the planner cannot accidentally backtrack into a
        smash-capable cell.

    Actions:
        forward_E / forward_N  : enter the eastern / northern neighbour if
                                  it is in-bounds, not a wall, and not a
                                  closed door cell.
        pickup                  : at K, if not has_key.
        toggle                  : at the cell directly west of D, if has_key
                                  and not door_open.
        smash                   : at S — gives r_d, transitions to 'trap'.
        collect                 : at G — gives r_g, transitions to 'absorb'.
        recover                 : trap → (cell east-of-start, has_key=F,
                                  door_open=F) — only in `reversible` mode.

    Trap behaviour encodes the matched control:
        irreversible : no outgoing actions, so the goal becomes unreachable
                       (and D_w(smash) > 0).
        reversible   : `recover` leads to the cell *east* of the start,
                       not back to S itself. The goal stays reachable
                       (D_w = 0), and the smash decoy can be taken at most
                       once (because the one-way grid never routes back to S).
    """
    assert mode in ("irreversible", "reversible")
    rows = list(layout)
    H_grid = len(rows)
    W_grid = len(rows[0])
    for r in rows:
        assert len(r) == W_grid, "layout rows must have equal width"

    def cell(rr: int, cc: int) -> str:
        if 0 <= rr < H_grid and 0 <= cc < W_grid:
            return rows[rr][cc]
        return "#"

    def find(ch: str) -> Tuple[int, int]:
        for rr in range(H_grid):
            for cc in range(W_grid):
                if rows[rr][cc] == ch:
                    return (rr, cc)
        raise ValueError(f"cell {ch!r} not found in layout")

    start_pos = find("S")
    key_pos = find("K")
    door_pos = find("D")
    goal_pos = find("G")
    toggle_pos = (door_pos[0], door_pos[1] - 1)
    recover_pos = (start_pos[0], start_pos[1] + 1)

    def can_enter(pos: Tuple[int, int], door_open: bool) -> bool:
        rr, cc = pos
        ch = cell(rr, cc)
        if ch == "#":
            return False
        if ch == "D" and not door_open:
            return False
        return True

    initial = ("grid", start_pos, False, False)

    states_list: List[Any] = []
    actions: Dict[Any, List[str]] = {}
    f: Dict[Tuple[Any, str], Any] = {}
    r: Dict[Tuple[Any, str], float] = {}

    seen: set = set()
    queue: List[Any] = [initial]

    while queue:
        s = queue.pop()
        if s in seen:
            continue
        seen.add(s)
        states_list.append(s)

        if s == "absorb":
            actions[s] = []
            continue

        if s == "trap":
            acts: List[str] = []
            if mode == "reversible":
                acts.append("recover")
                recover_state = ("grid", recover_pos, False, False)
                f[(s, "recover")] = recover_state
                r[(s, "recover")] = 0.0
                if recover_state not in seen:
                    queue.append(recover_state)
            actions[s] = acts
            continue

        # ('grid', pos, has_key, door_open)
        _, pos, has_key, door_open = s
        rr, cc = pos
        acts = []

        # smash at start cell
        if pos == start_pos:
            acts.append("smash")
            f[(s, "smash")] = "trap"
            r[(s, "smash")] = r_d
            if "trap" not in seen:
                queue.append("trap")

        # pickup at key cell if no key yet
        if pos == key_pos and not has_key:
            acts.append("pickup")
            ns = ("grid", pos, True, door_open)
            f[(s, "pickup")] = ns
            r[(s, "pickup")] = r_pickup
            if ns not in seen:
                queue.append(ns)

        # toggle at toggle_pos if has_key and door closed
        if pos == toggle_pos and has_key and not door_open:
            acts.append("toggle")
            ns = ("grid", pos, has_key, True)
            f[(s, "toggle")] = ns
            r[(s, "toggle")] = r_toggle
            if ns not in seen:
                queue.append(ns)

        # collect at goal cell
        if pos == goal_pos:
            acts.append("collect")
            f[(s, "collect")] = "absorb"
            r[(s, "collect")] = r_g
            if "absorb" not in seen:
                queue.append("absorb")

        # forward_E
        tgt_e = (rr, cc + 1)
        if can_enter(tgt_e, door_open):
            acts.append("forward_E")
            ns = ("grid", tgt_e, has_key, door_open)
            f[(s, "forward_E")] = ns
            r[(s, "forward_E")] = 0.0
            if ns not in seen:
                queue.append(ns)

        # forward_N
        tgt_n = (rr - 1, cc)
        if can_enter(tgt_n, door_open):
            acts.append("forward_N")
            ns = ("grid", tgt_n, has_key, door_open)
            f[(s, "forward_N")] = ns
            r[(s, "forward_N")] = 0.0
            if ns not in seen:
                queue.append(ns)

        actions[s] = acts

    # Target = every reachable state whose `pos` is the goal cell.
    targets = {s for s in states_list
               if isinstance(s, tuple) and len(s) == 4 and s[1] == goal_pos}
    target_weights = {t: r_g for t in targets}

    mdp = MDP(
        states=states_list, actions=actions, f=f, r=r,
        targets=targets, target_weights=target_weights, gamma=gamma,
    )
    info = {
        "layout": list(rows),
        "start_pos": start_pos, "key_pos": key_pos, "door_pos": door_pos,
        "goal_pos": goal_pos, "toggle_pos": toggle_pos,
        "recover_pos": recover_pos, "initial_state": initial,
        "mode": mode,
    }
    return mdp, info


# =====================================================================
# 2. Oracle planner — exact value iteration
# =====================================================================

def value_iteration(mdp: MDP, tol: float = 1e-12,
                    max_iter: int = 100_000) -> Tuple[Dict[Any, float], int]:
    V = {s: 0.0 for s in mdp.states}
    for it in range(max_iter):
        delta = 0.0
        V_new: Dict[Any, float] = {}
        for s in mdp.states:
            acts = mdp.actions.get(s, [])
            if not acts:
                V_new[s] = 0.0
                continue
            best = max(mdp.r[(s, a)] + mdp.gamma * V[mdp.f[(s, a)]] for a in acts)
            V_new[s] = best
            delta = max(delta, abs(best - V[s]))
        V = V_new
        if delta < tol:
            return V, it + 1
    return V, max_iter


def make_oracle_policy(mdp: MDP, V_star: Dict[Any, float]) -> Callable[[Any], str]:
    def pol(s):
        acts = sorted(mdp.actions[s])
        return max(acts, key=lambda a: mdp.r[(s, a)] + mdp.gamma * V_star[mdp.f[(s, a)]])
    return pol


# =====================================================================
# 3. Run all three planners on both variants
# =====================================================================

def trace_policy(mdp: MDP, start: Any, choose_action: Callable[[Any], str],
                 max_steps: int = 400) -> Dict[str, Any]:
    """Discounted return, raw return, # steps, whether the rollout ends in `absorb`
    (i.e. the goal was collected)."""
    s = start
    total_disc = 0.0
    total_raw = 0.0
    disc = 1.0
    steps = 0
    success = False
    trajectory: List[Tuple[Any, str, float, Any]] = []
    for _ in range(max_steps):
        if not mdp.actions.get(s):
            if s == "absorb":
                success = True
            break
        a = choose_action(s)
        s2 = mdp.f[(s, a)]
        rew = mdp.r[(s, a)]
        total_disc += disc * rew
        total_raw += rew
        disc *= mdp.gamma
        trajectory.append((s, a, rew, s2))
        steps += 1
        if s2 == s and rew == 0.0:
            break
        s = s2
    if s == "absorb":
        success = True
    return {
        "discounted_return": total_disc,
        "raw_return": total_raw,
        "steps": steps,
        "success": success,
        "final_state": str(s),
        "trajectory_len": len(trajectory),
    }


def run_one_mode(mode: str, *, H: int, lam: float, gamma: float,
                 r_d: float, r_g: float, r_pickup: float, r_toggle: float,
                 layout: Tuple[str, ...] = DEFAULT_LAYOUT) -> Dict[str, Any]:
    mdp, info = build_doorkey_mdp(
        layout=layout, r_d=r_d, r_g=r_g, r_pickup=r_pickup,
        r_toggle=r_toggle, gamma=gamma, mode=mode,
    )
    start = info["initial_state"]

    Dw_smash = destroyed_mass(mdp, start, "smash")
    dist = bfs_distances(mdp, start)

    V_star, vi_iters = value_iteration(mdp)
    oracle_pi = make_oracle_policy(mdp, V_star)

    reward_only_pi = lambda s: policy_obl(mdp, s, H)
    mrc_pi = lambda s: policy_mrc(mdp, s, H, lam)

    res_reward = trace_policy(mdp, start, reward_only_pi)
    res_mrc = trace_policy(mdp, start, mrc_pi)
    res_oracle = trace_policy(mdp, start, oracle_pi)

    # Goal-state BFS distance for logging
    goal_states = sorted(mdp.targets, key=lambda t: dist.get(t, 10**9))
    goal_state_min = goal_states[0] if goal_states else None
    goal_dist = dist.get(goal_state_min, None) if goal_state_min is not None else None

    return {
        "mode": mode,
        "n_states": len(mdp.states),
        "n_targets": len(mdp.targets),
        "goal_state_bfs_distance": goal_dist,
        "D_w_smash": Dw_smash,
        "V_star_start": V_star[start],
        "vi_iterations": vi_iters,
        "results": {
            "reward_only": res_reward,
            "mrc": res_mrc,
            "oracle": res_oracle,
        },
        "policy_at_start": {
            "reward_only": reward_only_pi(start),
            "mrc": mrc_pi(start),
            "oracle": oracle_pi(start),
        },
    }


# =====================================================================
# 4. Plots
# =====================================================================

def plot_paired_bars(per_mode: Dict[str, Dict[str, Any]], out_path: str) -> None:
    modes = ["irreversible", "reversible"]
    planners = ["reward_only", "mrc", "oracle"]
    planner_labels = ["reward-only (Π^H_obl)", "contraction-aware (MRC, λ=1)", "oracle (VI)"]
    colours = ["#5079a8", "#d96241", "#3e8e57"]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    n_groups = len(modes)
    bar_w = 0.24
    x = np.arange(n_groups)

    for i, (planner, label, colour) in enumerate(zip(planners, planner_labels, colours)):
        vals = [per_mode[m]["results"][planner]["discounted_return"] for m in modes]
        bars = ax.bar(x + (i - 1) * bar_w, vals, width=bar_w, color=colour,
                      label=label, edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m + f"\n(D_w = {per_mode[m]['D_w_smash']:.3f})" for m in modes])
    ax.set_ylabel("discounted return  V^π(s_0)")
    ax.set_title("Part B — MiniGrid DoorKey-style benchmark\n"
                 "matched twin: irreversible (smash burns access to G) vs reversible "
                 "(recover preserves access)")
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_h_sweep(layout: Tuple[str, ...], *, r_d: float, r_g: float,
                 r_pickup: float, r_toggle: float, gamma: float,
                 lam: float, out_path: str) -> Dict[str, Any]:
    """Show V(planner) vs H for both modes. Confirms the chosen H sits in the
    regime where reward-only myopically picks smash AND MRC avoidance pays off."""
    H_values = list(range(1, 16))
    series: Dict[str, Dict[str, List[float]]] = {}
    for mode in ("irreversible", "reversible"):
        mdp, info = build_doorkey_mdp(
            layout=layout, r_d=r_d, r_g=r_g, r_pickup=r_pickup,
            r_toggle=r_toggle, gamma=gamma, mode=mode,
        )
        start = info["initial_state"]
        V_star, _ = value_iteration(mdp)
        oracle_pi = make_oracle_policy(mdp, V_star)
        v_oracle = trace_policy(mdp, start, oracle_pi)["discounted_return"]

        vs_reward: List[float] = []
        vs_mrc: List[float] = []
        for H in H_values:
            ro = trace_policy(mdp, start, lambda s, H=H: policy_obl(mdp, s, H))
            mr = trace_policy(mdp, start, lambda s, H=H: policy_mrc(mdp, s, H, lam))
            vs_reward.append(ro["discounted_return"])
            vs_mrc.append(mr["discounted_return"])
        series[mode] = {
            "H": H_values,
            "reward_only": vs_reward,
            "mrc": vs_mrc,
            "oracle": v_oracle,
        }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, mode in zip(axes, ("irreversible", "reversible")):
        s = series[mode]
        ax.plot(s["H"], s["reward_only"], "o-", color="#5079a8",
                label="reward-only (Π^H_obl)")
        ax.plot(s["H"], s["mrc"], "s-", color="#d96241",
                label="contraction-aware (MRC, λ=1)")
        ax.axhline(s["oracle"], color="#3e8e57", linestyle="--",
                   label=f"oracle (VI) = {s['oracle']:.3f}")
        ax.set_xlabel("planner horizon H")
        ax.set_title(f"{mode} variant")
        ax.legend(framealpha=0.95)
    axes[0].set_ylabel("discounted return  V^π(s_0)")
    fig.suptitle("H-sensitivity sanity sweep — selected H sits in the regime where "
                 "reward-only myopically picks smash", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return series


# =====================================================================
# 5. Main
# =====================================================================

def main() -> bool:
    t0 = time.time()
    # ---- Pre-registered parameters (locked before looking at output) -----
    GAMMA = 0.9
    R_D = 0.3
    R_G = 2.0
    R_PICKUP = 0.01
    R_TOGGLE = 0.01
    H_PLANNER = 4              # short enough that Q^reward_H(s_0, a_safe) ≈ 0
    LAMBDA = 1.0               # spec mandates λ = 1
    LAYOUT = DEFAULT_LAYOUT
    PASS_TOL = 1e-9            # exact-equality tolerance for V comparisons
    GAP_TOL = 1e-6             # minimum visible advantage of MRC over reward-only

    print("=" * 72)
    print("Stage-2 / Part B — DoorKey-style benchmark (CPU planning, matched twin)")
    print("=" * 72)

    per_mode: Dict[str, Dict[str, Any]] = {}
    for mode in ("irreversible", "reversible"):
        per_mode[mode] = run_one_mode(
            mode, H=H_PLANNER, lam=LAMBDA, gamma=GAMMA,
            r_d=R_D, r_g=R_G, r_pickup=R_PICKUP, r_toggle=R_TOGGLE,
            layout=LAYOUT,
        )

    # ---- Print headline table -------------------------------------------
    def fmt_row(planner: str) -> str:
        v_irr = per_mode["irreversible"]["results"][planner]["discounted_return"]
        v_rev = per_mode["reversible"]["results"][planner]["discounted_return"]
        s_irr = per_mode["irreversible"]["results"][planner]["success"]
        s_rev = per_mode["reversible"]["results"][planner]["success"]
        return (f"  {planner:<14}  "
                f"V_irrev = {v_irr:>7.4f}  (goal_reached={s_irr})    "
                f"V_rev = {v_rev:>7.4f}  (goal_reached={s_rev})")

    print(f"\nLayout (rows top→bottom, cols left→right):")
    for row in LAYOUT:
        print(f"    {row}")
    print()
    print(f"D_w(s_0, smash) under each variant:")
    print(f"  irreversible : {per_mode['irreversible']['D_w_smash']:.6f}")
    print(f"  reversible   : {per_mode['reversible']['D_w_smash']:.6f}  (expect exactly 0)")
    print(f"\nGoal-state BFS distances:")
    print(f"  irreversible : {per_mode['irreversible']['goal_state_bfs_distance']}")
    print(f"  reversible   : {per_mode['reversible']['goal_state_bfs_distance']}")
    print(f"\nDiscounted return V^π(s_0) per planner × variant:")
    print(fmt_row("reward_only"))
    print(fmt_row("mrc"))
    print(fmt_row("oracle"))

    print(f"\nPolicy at start state:")
    for mode in ("irreversible", "reversible"):
        pol = per_mode[mode]["policy_at_start"]
        print(f"  {mode:<13} reward_only='{pol['reward_only']}'  "
              f"mrc='{pol['mrc']}'  oracle='{pol['oracle']}'")

    # ---- PASS / FAIL evaluation -----------------------------------------
    irr = per_mode["irreversible"]
    rev = per_mode["reversible"]

    v_ro_irr = irr["results"]["reward_only"]["discounted_return"]
    v_mrc_irr = irr["results"]["mrc"]["discounted_return"]
    v_ora_irr = irr["results"]["oracle"]["discounted_return"]
    v_ro_rev = rev["results"]["reward_only"]["discounted_return"]
    v_mrc_rev = rev["results"]["mrc"]["discounted_return"]
    v_ora_rev = rev["results"]["oracle"]["discounted_return"]

    # Irreversible: MRC > reward-only, MRC ≈ oracle, D_w > 0
    irr_dw_positive = irr["D_w_smash"] > 0.0
    irr_mrc_beats_reward = v_mrc_irr > v_ro_irr + GAP_TOL
    irr_mrc_near_oracle = abs(v_mrc_irr - v_ora_irr) < PASS_TOL
    irr_ok = irr_dw_positive and irr_mrc_beats_reward and irr_mrc_near_oracle

    # Reversible: D_w == 0 AND all three converge
    rev_dw_zero = (rev["D_w_smash"] == 0.0)
    rev_mrc_vs_reward = abs(v_mrc_rev - v_ro_rev) < PASS_TOL
    rev_mrc_vs_oracle = abs(v_mrc_rev - v_ora_rev) < PASS_TOL
    rev_reward_vs_oracle = abs(v_ro_rev - v_ora_rev) < PASS_TOL
    rev_converged = rev_mrc_vs_reward and rev_mrc_vs_oracle and rev_reward_vs_oracle
    rev_ok = rev_dw_zero and rev_converged

    passed = irr_ok and rev_ok

    print(f"\nPre-registered PASS / FAIL evaluation:")
    print(f"  Irreversible:")
    print(f"    D_w(smash) > 0                : {irr_dw_positive}  "
          f"(D_w = {irr['D_w_smash']:.6f})")
    print(f"    V_MRC > V_reward_only + {GAP_TOL:g}: {irr_mrc_beats_reward}  "
          f"(gap = {v_mrc_irr - v_ro_irr:+.6f})")
    print(f"    |V_MRC − V_oracle| < {PASS_TOL:g}    : {irr_mrc_near_oracle}  "
          f"(diff = {v_mrc_irr - v_ora_irr:+.6e})")
    print(f"  Reversible:")
    print(f"    D_w(smash) == 0               : {rev_dw_zero}  "
          f"(D_w = {rev['D_w_smash']})")
    print(f"    |V_MRC − V_reward_only| < {PASS_TOL:g}: {rev_mrc_vs_reward}  "
          f"(diff = {v_mrc_rev - v_ro_rev:+.6e})")
    print(f"    |V_MRC − V_oracle|      < {PASS_TOL:g}: {rev_mrc_vs_oracle}  "
          f"(diff = {v_mrc_rev - v_ora_rev:+.6e})")
    print(f"    |V_reward_only − V_oracle| < {PASS_TOL:g}: {rev_reward_vs_oracle}  "
          f"(diff = {v_ro_rev - v_ora_rev:+.6e})")
    print(f"\nOverall Part B: {'PASS' if passed else 'FAIL'}")

    # ---- Persist --------------------------------------------------------
    out_csv = os.path.join(BENCH_DIR, "benchmark_results.csv")
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode", "planner", "discounted_return", "raw_return",
                          "steps", "success", "D_w_smash", "policy_at_start",
                          "V_star_start"])
        for mode in ("irreversible", "reversible"):
            d = per_mode[mode]
            for planner in ("reward_only", "mrc", "oracle"):
                row = d["results"][planner]
                writer.writerow([
                    mode, planner,
                    f"{row['discounted_return']:.9f}",
                    f"{row['raw_return']:.9f}",
                    row["steps"], row["success"],
                    f"{d['D_w_smash']:.9f}",
                    d["policy_at_start"][planner],
                    f"{d['V_star_start']:.9f}",
                ])
    print(f"\nCSV table written to {out_csv}")

    # Plots
    out_bars = os.path.join(BENCH_DIR, "benchmark_returns.pdf")
    plot_paired_bars(per_mode, out_bars)
    print(f"Bar chart written to {out_bars}")

    out_h = os.path.join(BENCH_DIR, "benchmark_h_sweep.pdf")
    h_sweep = plot_h_sweep(
        LAYOUT, r_d=R_D, r_g=R_G, r_pickup=R_PICKUP, r_toggle=R_TOGGLE,
        gamma=GAMMA, lam=LAMBDA, out_path=out_h,
    )
    print(f"H sweep written to {out_h}")

    runtime_s = time.time() - t0
    print(f"\nRuntime: {runtime_s:.2f} s  (CPU only, no GPU)")

    summary = {
        "overall_pass": passed,
        "runtime_seconds": runtime_s,
        "parameters": {
            "gamma": GAMMA, "r_d": R_D, "r_g": R_G,
            "r_pickup": R_PICKUP, "r_toggle": R_TOGGLE,
            "H_planner": H_PLANNER, "lambda": LAMBDA,
            "pass_tol": PASS_TOL, "gap_tol": GAP_TOL,
            "layout": list(LAYOUT),
        },
        "checks": {
            "irreversible": {
                "D_w_positive": irr_dw_positive,
                "MRC_beats_reward_only": irr_mrc_beats_reward,
                "MRC_matches_oracle": irr_mrc_near_oracle,
                "all_ok": irr_ok,
            },
            "reversible": {
                "D_w_zero": rev_dw_zero,
                "MRC_eq_reward_only": rev_mrc_vs_reward,
                "MRC_eq_oracle": rev_mrc_vs_oracle,
                "reward_only_eq_oracle": rev_reward_vs_oracle,
                "all_ok": rev_ok,
            },
        },
        "per_mode": _jsonable(per_mode),
        "h_sweep": _jsonable(h_sweep),
        "artifacts": {
            "csv": out_csv,
            "bars_pdf": out_bars,
            "h_sweep_pdf": out_h,
        },
    }
    summary_path = os.path.join(_THIS_DIR, "results_part_b.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary written to {summary_path}")
    return passed


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_jsonable(x) for x in obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    return str(obj)


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

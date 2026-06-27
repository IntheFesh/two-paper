"""
Stage-2 / Part A — paper-quality figures for the MRC paper.

This script reuses Stage-1's primitives verbatim:
    - MDP class
    - destroyed_mass(mdp, s, a)            (the canonical D_w)
    - build_mrc_corridor / build_resource_mdp / build_augmented_graph
    - value_h / q_reward_h / q_mrc / policy_obl / policy_mrc / rollout_value
None of them are redefined here; we only fan out their inputs to produce
publication-grade figures.

Produces (in experiments/stage2_paper/figures/):
    fig1_gap_vs_dw.pdf    — ΔV = D_w − r_d across k and u, with reversible twin overlay
    fig2_collapse.pdf     — irreversible vs reversible twin across many parameter combos
    fig3_lambda_phase.pdf — λ phase transition for several (r_d, D_w) configurations
    fig4_resource_graph.pdf — three-panel scatter (R, D_w, Q) on the y=x diagonal + renewable contrast
    table1_properties_scaling.csv — numerical values for reversible-zero, additivity,
                                    representation-invariance + scaling rows

Pre-registered Part A PASS conditions:
    - Fig 1 linear fit slope == 1.0, intercept == −r_d (machine precision).
    - Fig 2 reversible ΔV ≡ 0 across all sampled parameter combos.
    - Fig 3 each empirical switch λ* matches λ_min = r_d / D_w within grid step.
    - Fig 4 all R/D_w/Q points lie on y=x for the non-renewable case; renewable
            D_w cluster collapses to 0.
    - Table 1 numerical assertions all hold exactly (1e-12).
Any FAIL is printed and recorded in the CSV — do NOT mask.

Estimated runtime: ~30 seconds on a single CPU core. No GPU. numpy + matplotlib only.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt

# ---- Reuse Stage-1 primitives without modification ---------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "experiments", "stage1_unified"))

from stage1_unified_validation import (  # noqa: E402
    MDP,
    destroyed_mass,
    reachable_set,
    bfs_distances,
    value_h,
    q_reward_h,
    q_mrc,
    policy_obl,
    policy_mrc,
    rollout_value,
    build_mrc_corridor,
    build_resource_mdp,
    build_augmented_graph,
)

# ---- Plot style --------------------------------------------------------
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
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

FIG_DIR = os.path.join(_THIS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# =====================================================================
# Helpers — single source of truth for "build + measure" used everywhere
# =====================================================================

def make_corridor(k: int, m: int, r_d: float, r_g: float, gamma: float,
                  mode: str = "irreversible") -> MDP:
    """Thin wrapper around stage-1's build_mrc_corridor. H is for documentation
    only — planners receive H explicitly."""
    return build_mrc_corridor(
        k=k, H=m, m=m, r_d=r_d, r_g=r_g, gamma=gamma, mode=mode,
    )


def measure_gap(mdp: MDP, H: int, lam: float, start: Any = "s0") -> Dict[str, float]:
    """Return D_w (decoy), V_obl, V_mrc, ΔV from a corridor-shaped MDP."""
    Dw = destroyed_mass(mdp, start, "a_decoy")
    v_obl = rollout_value(mdp, start, lambda s: policy_obl(mdp, s, H))
    v_mrc = rollout_value(mdp, start, lambda s: policy_mrc(mdp, s, H, lam))
    return {"D_w": Dw, "V_obl": v_obl, "V_mrc": v_mrc, "delta_V": v_mrc - v_obl}


# =====================================================================
# Fig 1 — gap tracks destroyed mass
# =====================================================================

def fig1_gap_vs_dw() -> Dict[str, Any]:
    """ΔV vs D_w across many (k, u) settings; reversible twin overlay; theory line."""
    r_d = 1.0
    gamma = 0.9
    m = 4
    H = m
    lam = 20.0           # well above any λ_min seen here
    ks = list(range(1, 31))
    weights = [0.5, 1.0, 1.5, 2.0]

    # Sweep target weight u via r_g (build_mrc_corridor links edge reward
    # and target weight, which is required for the slope-1 prediction to hold).
    irr_pts: List[Dict[str, float]] = []
    rev_pts: List[Dict[str, float]] = []
    for u in weights:
        for k in ks:
            mdp_irr = make_corridor(k, m, r_d, r_g=u, gamma=gamma, mode="irreversible")
            mdp_rev = make_corridor(k, m, r_d, r_g=u, gamma=gamma, mode="reversible")
            irr = measure_gap(mdp_irr, H, lam)
            rev = measure_gap(mdp_rev, H, lam)
            irr_pts.append({"k": k, "u": u, **irr})
            rev_pts.append({"k": k, "u": u, **rev})

    xs_irr = np.array([p["D_w"] for p in irr_pts])
    ys_irr = np.array([p["delta_V"] for p in irr_pts])
    xs_rev = np.array([p["D_w"] for p in rev_pts])
    ys_rev = np.array([p["delta_V"] for p in rev_pts])

    slope, intercept = np.polyfit(xs_irr, ys_irr, 1)
    slope_ok = abs(slope - 1.0) < 1e-9
    intercept_ok = abs(intercept - (-r_d)) < 1e-9
    pointwise_ok = bool(np.max(np.abs(ys_irr - (xs_irr - r_d))) < 1e-12)
    rev_zero_ok = bool(np.max(np.abs(ys_rev)) < 1e-12) and bool(np.max(np.abs(xs_rev)) < 1e-12)
    passed = slope_ok and intercept_ok and pointwise_ok and rev_zero_ok

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cmap = plt.get_cmap("viridis")
    for i, u in enumerate(weights):
        mask = np.array([p["u"] == u for p in irr_pts])
        ax.scatter(xs_irr[mask], ys_irr[mask], s=22, alpha=0.85,
                   color=cmap(i / max(1, len(weights) - 1)),
                   label=f"irreversible, u = {u:g}")
    # Reversible twins, single overlay (they all sit on y = 0 with D_w = 0)
    ax.scatter(xs_rev, ys_rev, s=30, marker="x", color="firebrick",
               label="reversible twin (D_w = 0, ΔV = 0)", zorder=5)
    # Theory line
    xs_line = np.linspace(0.0, float(xs_irr.max()) * 1.05, 200)
    ax.plot(xs_line, xs_line - r_d, color="black", linestyle="--",
            label=f"theory: ΔV = D_w − r_d (r_d = {r_d:g})")
    ax.axhline(0.0, color="gray", linewidth=0.6, alpha=0.7)
    ax.set_xlabel(r"$D_w(s_0, a_{\mathrm{decoy}})$  (destroyed reachable mass)")
    ax.set_ylabel(r"$\Delta V = V^{\pi_{\mathrm{MRC}}}(s_0) - V^{\pi_{\mathrm{obl}}}(s_0)$")
    ax.set_title("Fig 1 — gap tracks destroyed mass\n"
                 f"fit: slope = {slope:.6f}, intercept = {intercept:.6f}  "
                 f"(expect 1, {-r_d:g})")
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig1_gap_vs_dw.pdf")
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "name": "Fig 1 — gap tracks destroyed mass",
        "n_irr_points": len(irr_pts),
        "n_rev_points": len(rev_pts),
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_ok": slope_ok,
        "intercept_ok": intercept_ok,
        "pointwise_ok": pointwise_ok,
        "reversible_zero_ok": rev_zero_ok,
        "passed": bool(passed),
        "out_path": out_path,
    }


# =====================================================================
# Fig 2 — collapse causal identification (matched twin sweep)
# =====================================================================

def fig2_collapse() -> Dict[str, Any]:
    """Many parameter combos. Show irreversible gap > 0 and reversible gap ≡ 0."""
    gamma = 0.9
    H_calibrate = lambda m: m  # planner can't see through a_safe within m steps
    lam = 20.0

    configs: List[Dict[str, Any]] = []
    rng_k = [1, 2, 3, 4, 5, 8, 12]
    rng_m = [3, 4, 6]
    rng_rd = [0.5, 1.0, 1.5, 2.0]
    rng_rg = [0.5, 1.0, 2.0]
    for m in rng_m:
        for k in rng_k:
            for r_d in rng_rd:
                for r_g in rng_rg:
                    H = H_calibrate(m)
                    mdp_irr = make_corridor(k, m, r_d, r_g, gamma, "irreversible")
                    mdp_rev = make_corridor(k, m, r_d, r_g, gamma, "reversible")
                    irr = measure_gap(mdp_irr, H, lam)
                    rev = measure_gap(mdp_rev, H, lam)
                    configs.append({
                        "k": k, "m": m, "r_d": r_d, "r_g": r_g,
                        "D_w_irr": irr["D_w"], "delta_V_irr": irr["delta_V"],
                        "D_w_rev": rev["D_w"], "delta_V_rev": rev["delta_V"],
                        "predicted_irr": irr["D_w"] - r_d,
                    })

    # Causal identification: reversible twin must collapse, in ALL configs.
    rev_dw_max = max(abs(c["D_w_rev"]) for c in configs)
    rev_dv_max = max(abs(c["delta_V_rev"]) for c in configs)
    irr_match_max = max(abs(c["delta_V_irr"] - c["predicted_irr"]) for c in configs)
    rev_zero_ok = rev_dw_max < 1e-12 and rev_dv_max < 1e-12
    irr_pred_ok = irr_match_max < 1e-12
    passed = rev_zero_ok and irr_pred_ok

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # Left: ΔV histogram, paired by config.
    irr_dv = np.array([c["delta_V_irr"] for c in configs])
    rev_dv = np.array([c["delta_V_rev"] for c in configs])
    bins = np.linspace(min(irr_dv.min(), rev_dv.min()) - 0.1,
                        max(irr_dv.max(), rev_dv.max()) + 0.1, 50)
    ax = axes[0]
    ax.hist(irr_dv, bins=bins, alpha=0.6, color="steelblue",
            label=f"irreversible (n = {len(configs)})")
    ax.hist(rev_dv, bins=bins, alpha=0.7, color="firebrick",
            label=f"reversible twin  (n = {len(configs)})")
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel(r"$\Delta V = V^{\pi_{\mathrm{MRC}}} - V^{\pi_{\mathrm{obl}}}$")
    ax.set_ylabel("# configs")
    ax.set_title("ΔV distribution across all parameter combos")
    ax.legend()

    # Right: ΔV vs D_w with theory line; reversible all at origin.
    ax = axes[1]
    Dw_irr = np.array([c["D_w_irr"] for c in configs])
    pred_irr = np.array([c["predicted_irr"] for c in configs])
    rd_arr = np.array([c["r_d"] for c in configs])
    sc = ax.scatter(Dw_irr, irr_dv, c=rd_arr, cmap="plasma", s=20, alpha=0.85,
                    label="irreversible (color = r_d)")
    ax.scatter(np.array([c["D_w_rev"] for c in configs]),
               rev_dv, marker="x", color="firebrick", s=30,
               label="reversible twin", zorder=5)
    # Per-r_d theory rays
    for r_d in rng_rd:
        xs = np.linspace(0, Dw_irr.max() * 1.05, 200)
        ax.plot(xs, xs - r_d, linestyle="--", linewidth=0.8, alpha=0.7,
                color="black")
    ax.axhline(0.0, color="gray", linewidth=0.6, alpha=0.7)
    ax.set_xlabel(r"$D_w(s_0, a_{\mathrm{decoy}})$")
    ax.set_ylabel(r"$\Delta V$")
    ax.set_title("ΔV vs D_w (dashed: theory ΔV = D_w − r_d, one ray per r_d)")
    fig.colorbar(sc, ax=ax, label="r_d", pad=0.02)
    ax.legend(loc="upper left", framealpha=0.95)

    fig.suptitle(f"Fig 2 — collapse causal identification across {len(configs)} matched configs",
                 y=1.02)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig2_collapse.pdf")
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "name": "Fig 2 — collapse causal identification",
        "n_configs": len(configs),
        "reversible_D_w_max_abs": rev_dw_max,
        "reversible_delta_V_max_abs": rev_dv_max,
        "irreversible_prediction_max_abs_error": irr_match_max,
        "reversible_zero_ok": bool(rev_zero_ok),
        "irreversible_prediction_ok": bool(irr_pred_ok),
        "passed": bool(passed),
        "out_path": out_path,
    }


# =====================================================================
# Fig 3 — λ phase transition (multiple (r_d, D_w))
# =====================================================================

def fig3_lambda_phase() -> Dict[str, Any]:
    """λ scan for several configs; verify switching at λ_min = r_d / D_w."""
    gamma = 0.9
    r_g = 1.0
    m = 4
    H = m
    lambdas = np.linspace(0.0, 1.5, 3001)  # resolution 0.0005
    grid_step = float(lambdas[1] - lambdas[0])

    configs: List[Dict[str, Any]] = []
    # Spread (r_d, k) so λ_min spans a range below and above small values
    for r_d in [0.5, 1.0, 1.5, 2.0]:
        for k in [2, 3, 5, 8]:
            configs.append({"r_d": r_d, "k": k})

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    cmap = plt.get_cmap("tab20")
    rows: List[Dict[str, Any]] = []
    n_pass = 0
    for i, cfg in enumerate(configs):
        mdp = make_corridor(cfg["k"], m, cfg["r_d"], r_g, gamma, "irreversible")
        Dw = destroyed_mass(mdp, "s0", "a_decoy")
        if Dw <= 0:
            continue
        lam_min = cfg["r_d"] / Dw
        # only configs with separation precondition λ_min < 1 are eligible
        if lam_min >= 1.5:
            continue
        actions = [policy_mrc(mdp, "s0", H, float(lam)) for lam in lambdas]
        is_safe = np.array([1.0 if a == "a_safe" else 0.0 for a in actions])
        switch_idx = int(np.argmax(is_safe > 0)) if is_safe.any() else -1
        lam_star = float(lambdas[switch_idx]) if switch_idx >= 0 else None
        ok = (lam_star is not None) and abs(lam_star - lam_min) <= grid_step + 1e-9
        at_lam_1 = policy_mrc(mdp, "s0", H, 1.0) if lam_min < 1.0 else None
        lam1_ok = (at_lam_1 == "a_safe") if lam_min < 1.0 else True

        color = cmap(i % 20)
        ax.step(lambdas, is_safe + 0.0, where="post", color=color, alpha=0.85,
                label=f"r_d={cfg['r_d']:g}, k={cfg['k']}, λ_min={lam_min:.3f}")
        ax.axvline(lam_min, color=color, linestyle=":", linewidth=0.8, alpha=0.7)

        rows.append({
            "r_d": cfg["r_d"], "k": cfg["k"], "D_w": Dw, "lam_min": lam_min,
            "lam_star": lam_star, "policy_at_lam_1": at_lam_1,
            "match_within_grid_step": bool(ok),
            "lam_1_picks_safe_when_required": bool(lam1_ok),
        })
        if ok and lam1_ok:
            n_pass += 1

    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--",
               label="λ = 1 (separation precondition reference)")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\mathbb{1}\{\pi_{\mathrm{MRC}}(s_0) = a_{\mathrm{safe}}\}$")
    ax.set_yticks([0.0, 1.0])
    ax.set_title("Fig 3 — λ phase transition for many (r_d, D_w) configs "
                 "(dotted: theoretical λ_min)")
    ax.legend(loc="center right", fontsize=7, framealpha=0.95, ncol=1,
              bbox_to_anchor=(1.46, 0.5))
    ax.set_xlim(0.0, 1.5)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig3_lambda_phase.pdf")
    fig.savefig(out_path)
    plt.close(fig)

    all_ok = (n_pass == len(rows)) and (len(rows) > 0)
    return {
        "name": "Fig 3 — λ phase transition",
        "n_configs_plotted": len(rows),
        "n_passing": n_pass,
        "grid_step": grid_step,
        "configs": rows,
        "passed": bool(all_ok),
        "out_path": out_path,
    }


# =====================================================================
# Fig 4 — resource–topology triple identity (3-panel y=x scatter)
# =====================================================================

def fig4_resource_graph() -> Dict[str, Any]:
    """Across varied (L, F), verify resource ↔ augmented-graph identity for R, D_w, Q^H_MRC.
    Plus renewable contrast — D_w collapses to 0 cluster."""
    gamma = 0.9
    r_d = 1.0
    r_g = 2.0
    lam = 20.0

    sizes = [(2, 3), (3, 4), (4, 4), (4, 6), (5, 6), (6, 8), (8, 10)]

    # Triples: list of (R_overlap_size_res, R_overlap_size_graph, D_res, D_graph, Q_res, Q_graph)
    R_pairs: List[Tuple[int, int]] = []
    Dw_pairs: List[Tuple[float, float]] = []
    Q_pairs: List[Tuple[float, float]] = []

    # Renewable D_w collapse: for each size, record D_w(splurge) under renewable mode.
    renew_Dw: List[float] = []
    renew_gap: List[float] = []

    for (L, F) in sizes:
        H = L
        res = build_resource_mdp(L, F, r_d, r_g, gamma, mode="non_renewable")
        gph, vmap = build_augmented_graph(res)

        # Per-state R cardinality (an injective check: size should match exactly).
        for s in res.states:
            R_res = reachable_set(res, s)
            R_gph = reachable_set(gph, vmap[s])
            R_pairs.append((len(R_res), len(R_gph)))

        # D_w + Q^H_MRC for every (s, a).
        for s in res.states:
            for a in res.actions[s]:
                Dw_pairs.append((destroyed_mass(res, s, a),
                                  destroyed_mass(gph, vmap[s], a)))
                Q_pairs.append((q_mrc(res, s, a, H, lam),
                                 q_mrc(gph, vmap[s], a, H, lam)))

        # Renewable contrast at (0, F).
        ren = build_resource_mdp(L, F, r_d, r_g, gamma, mode="renewable")
        D_ren = destroyed_mass(ren, (0, F), "splurge")
        v_obl_r = rollout_value(ren, (0, F), lambda s: policy_obl(ren, s, H))
        v_mrc_r = rollout_value(ren, (0, F), lambda s: policy_mrc(ren, s, H, lam))
        renew_Dw.append(D_ren)
        renew_gap.append(v_mrc_r - v_obl_r)

    R_pairs_arr = np.array(R_pairs)
    Dw_arr = np.array(Dw_pairs)
    Q_arr = np.array(Q_pairs)
    renew_Dw_arr = np.array(renew_Dw)
    renew_gap_arr = np.array(renew_gap)

    R_ok = bool(np.all(R_pairs_arr[:, 0] == R_pairs_arr[:, 1]))
    Dw_ok = bool(np.max(np.abs(Dw_arr[:, 0] - Dw_arr[:, 1])) < 1e-12)
    Q_ok = bool(np.max(np.abs(Q_arr[:, 0] - Q_arr[:, 1])) < 1e-9)
    renew_ok = bool(np.max(np.abs(renew_Dw_arr)) < 1e-12
                    and np.max(np.abs(renew_gap_arr)) < 1e-12)
    passed = R_ok and Dw_ok and Q_ok and renew_ok

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))

    # Panel A: |R| resource vs graph.
    ax = axes[0]
    ax.scatter(R_pairs_arr[:, 0], R_pairs_arr[:, 1], s=14, alpha=0.7,
               color="steelblue")
    lim = float(R_pairs_arr.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linestyle="--",
            label="y = x")
    ax.set_xlabel("|R(s)|   (resource representation)")
    ax.set_ylabel("|R(v_s)|  (augmented-graph repr.)")
    ax.set_title(f"(a) Reachable-set size  (n = {len(R_pairs)})")
    ax.legend(loc="upper left")

    # Panel B: D_w resource vs graph (non-renewable). Add renewable cluster at origin.
    ax = axes[1]
    ax.scatter(Dw_arr[:, 0], Dw_arr[:, 1], s=14, alpha=0.7, color="steelblue",
               label=f"non-renewable (n = {len(Dw_pairs)})")
    ax.scatter(renew_Dw_arr, renew_Dw_arr, s=60, marker="x", color="firebrick",
               label=f"renewable splurge D_w = 0 (n = {len(renew_Dw)})", zorder=5)
    lim = float(max(Dw_arr.max(), 1e-3)) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", label="y = x")
    ax.set_xlabel(r"$D_w$  (resource repr.)")
    ax.set_ylabel(r"$D_w$  (graph repr.)")
    ax.set_title("(b) Destroyed mass identity + renewable collapse")
    ax.legend(loc="upper left", fontsize=8)

    # Panel C: Q^H_MRC resource vs graph.
    ax = axes[2]
    ax.scatter(Q_arr[:, 0], Q_arr[:, 1], s=14, alpha=0.7, color="steelblue")
    lo, hi = float(Q_arr.min()), float(Q_arr.max())
    pad = (hi - lo) * 0.05 + 1e-3
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color="black", linestyle="--", label="y = x")
    ax.set_xlabel(r"$Q^H_{\mathrm{MRC}}$  (resource repr.)")
    ax.set_ylabel(r"$Q^H_{\mathrm{MRC}}$  (graph repr.)")
    ax.set_title(f"(c) $Q^H_{{\\mathrm{{MRC}}}}$ identity  (n = {len(Q_pairs)})")
    ax.legend(loc="upper left")

    fig.suptitle("Fig 4 — resource ↔ augmented-graph triple identity "
                 "(non-renewable) + renewable collapse", y=1.03)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig4_resource_graph.pdf")
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "name": "Fig 4 — resource ↔ graph triple identity",
        "n_R_pairs": len(R_pairs),
        "n_Dw_pairs": len(Dw_pairs),
        "n_Q_pairs": len(Q_pairs),
        "R_ok": R_ok, "Dw_ok": Dw_ok, "Q_ok": Q_ok,
        "renew_ok": renew_ok,
        "passed": bool(passed),
        "out_path": out_path,
    }


# =====================================================================
# Table 1 — properties + scaling
# =====================================================================

def table1_properties_scaling() -> Tuple[Dict[str, Any], str]:
    """Numerical evidence for reversible-zero / additivity / repr-invariance,
    plus scaling of D_w and gap with k and state count."""
    gamma = 0.9
    r_d = 1.0
    r_g = 1.0
    m = 4
    H = m
    lam = 20.0

    rows: List[Dict[str, Any]] = []

    # --- Property 1: reversible-zero ----------------------------------
    for k in [1, 3, 5, 10, 20]:
        mdp_rev = make_corridor(k, m, r_d, r_g, gamma, "reversible")
        Dw_rev = destroyed_mass(mdp_rev, "s0", "a_decoy")
        v_obl = rollout_value(mdp_rev, "s0", lambda s: policy_obl(mdp_rev, s, H))
        v_mrc = rollout_value(mdp_rev, "s0", lambda s: policy_mrc(mdp_rev, s, H, lam))
        rows.append({
            "section": "P1 reversible-zero",
            "config": f"k={k}, m={m}, reversible",
            "metric": "D_w(s0, a_decoy)",
            "value": Dw_rev,
            "expected": 0.0,
            "abs_error": abs(Dw_rev - 0.0),
            "ok": (Dw_rev == 0.0),
        })
        rows.append({
            "section": "P1 reversible-zero",
            "config": f"k={k}, m={m}, reversible",
            "metric": "ΔV(s0)",
            "value": v_mrc - v_obl,
            "expected": 0.0,
            "abs_error": abs(v_mrc - v_obl),
            "ok": abs(v_mrc - v_obl) < 1e-12,
        })

    # --- Property 2: additivity over disjoint target subsets ----------
    for (k_total, k_A) in [(5, 2), (8, 3), (10, 4), (12, 6)]:
        mdp_full = make_corridor(k_total, m, r_d, r_g, gamma, "irreversible")
        all_tgts = sorted(mdp_full.targets, key=lambda t: int(t[1:]))
        A_tgts = set(all_tgts[:k_A])
        B_tgts = set(all_tgts[k_A:])
        mdp_A = make_corridor(k_total, m, r_d, r_g, gamma, "irreversible")
        mdp_B = make_corridor(k_total, m, r_d, r_g, gamma, "irreversible")
        mdp_A.targets = A_tgts
        mdp_A.target_weights = {t: r_g for t in A_tgts}
        mdp_B.targets = B_tgts
        mdp_B.target_weights = {t: r_g for t in B_tgts}
        Dw_full = destroyed_mass(mdp_full, "s0", "a_decoy")
        Dw_A = destroyed_mass(mdp_A, "s0", "a_decoy")
        Dw_B = destroyed_mass(mdp_B, "s0", "a_decoy")
        rows.append({
            "section": "P2 additivity",
            "config": f"k_total={k_total}, |A|={k_A}, |B|={k_total-k_A}",
            "metric": "D_w(full) − [D_w(A) + D_w(B)]",
            "value": Dw_full - (Dw_A + Dw_B),
            "expected": 0.0,
            "abs_error": abs(Dw_full - (Dw_A + Dw_B)),
            "ok": abs(Dw_full - (Dw_A + Dw_B)) < 1e-12,
        })

    # --- Property 3: representation invariance ------------------------
    for (L, F) in [(2, 3), (3, 4), (4, 4), (5, 6), (6, 8)]:
        res = build_resource_mdp(L, F, r_d, r_g=2.0, gamma=gamma, mode="non_renewable")
        gph, vmap = build_augmented_graph(res)
        # Pick a non-trivial (s, a) pair: splurge at (0, F).
        d_res = destroyed_mass(res, (0, F), "splurge")
        d_gph = destroyed_mass(gph, vmap[(0, F)], "splurge")
        rows.append({
            "section": "P3 representation-invariance",
            "config": f"L={L}, F={F}, action=splurge",
            "metric": "D_w_resource − D_w_graph",
            "value": d_res - d_gph,
            "expected": 0.0,
            "abs_error": abs(d_res - d_gph),
            "ok": abs(d_res - d_gph) < 1e-12,
        })

    # --- Scaling: D_w(k) and ΔV(k) in corridor ------------------------
    for k in [1, 3, 5, 10, 15, 20, 25, 30]:
        mdp = make_corridor(k, m, r_d, r_g, gamma, "irreversible")
        Dw = destroyed_mass(mdp, "s0", "a_decoy")
        v_obl = rollout_value(mdp, "s0", lambda s: policy_obl(mdp, s, H))
        v_mrc = rollout_value(mdp, "s0", lambda s: policy_mrc(mdp, s, H, lam))
        n_states = len(mdp.states)
        rows.append({
            "section": "S corridor scaling (k)",
            "config": f"k={k}, m={m}",
            "metric": "n_states",
            "value": n_states, "expected": "", "abs_error": "", "ok": True,
        })
        rows.append({
            "section": "S corridor scaling (k)",
            "config": f"k={k}, m={m}",
            "metric": "D_w(s0, a_decoy)",
            "value": Dw, "expected": "", "abs_error": "", "ok": True,
        })
        rows.append({
            "section": "S corridor scaling (k)",
            "config": f"k={k}, m={m}",
            "metric": "ΔV(s0)  (V_mrc − V_obl)",
            "value": v_mrc - v_obl, "expected": "", "abs_error": "", "ok": True,
        })

    # --- Scaling: resource MDP state count ----------------------------
    for (L, F) in [(3, 4), (5, 6), (8, 10), (10, 12), (12, 15)]:
        t0 = time.time()
        res = build_resource_mdp(L, F, r_d, r_g=2.0, gamma=gamma, mode="non_renewable")
        Dw = destroyed_mass(res, (0, F), "splurge")
        runtime_ms = (time.time() - t0) * 1000.0
        rows.append({
            "section": "S resource scaling (L, F)",
            "config": f"L={L}, F={F}",
            "metric": "n_states",
            "value": len(res.states),
            "expected": "", "abs_error": "", "ok": True,
        })
        rows.append({
            "section": "S resource scaling (L, F)",
            "config": f"L={L}, F={F}",
            "metric": "D_w(splurge)",
            "value": Dw,
            "expected": "", "abs_error": "", "ok": True,
        })
        rows.append({
            "section": "S resource scaling (L, F)",
            "config": f"L={L}, F={F}",
            "metric": "build+D_w runtime (ms)",
            "value": runtime_ms,
            "expected": "", "abs_error": "", "ok": True,
        })

    # Persist.
    csv_path = os.path.join(FIG_DIR, "table1_properties_scaling.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["section", "config", "metric",
                                                 "value", "expected", "abs_error",
                                                 "ok"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    prop_rows = [r for r in rows if r["section"].startswith(("P1", "P2", "P3"))]
    passed = all(r["ok"] for r in prop_rows)
    return {
        "name": "Table 1 — properties + scaling",
        "n_rows": len(rows),
        "n_property_rows": len(prop_rows),
        "all_properties_ok": bool(passed),
        "out_path": csv_path,
        "passed": bool(passed),
    }, csv_path


# =====================================================================
# Driver
# =====================================================================

def main() -> bool:
    t0 = time.time()
    print("=" * 72)
    print("Stage-2 / Part A — paper-quality figures")
    print("=" * 72)

    res_fig1 = fig1_gap_vs_dw()
    res_fig2 = fig2_collapse()
    res_fig3 = fig3_lambda_phase()
    res_fig4 = fig4_resource_graph()
    res_table1, _ = table1_properties_scaling()

    print()
    print("=" * 72)
    print("Part A verdict table (pre-registered PASS/FAIL conditions)")
    print("=" * 72)
    print(f"{'Item':<8}  {'Status':<6}  Description")
    print("-" * 72)
    items = [("Fig 1", res_fig1), ("Fig 2", res_fig2), ("Fig 3", res_fig3),
             ("Fig 4", res_fig4), ("Table 1", res_table1)]
    all_pass = True
    for name, r in items:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_pass = False
        print(f"{name:<8}  {status:<6}  {r['name']}")
        path_key = "out_path"
        if path_key in r:
            print(f"           ↳  {r[path_key]}")
    print("-" * 72)
    runtime_s = time.time() - t0
    if all_pass:
        print(f"Overall Part A: ALL PASS  (runtime {runtime_s:.1f} s)")
    else:
        print(f"Overall Part A: FAIL — see above. (runtime {runtime_s:.1f} s)")

    # Persist top-level summary
    summary_path = os.path.join(_THIS_DIR, "results_part_a.json")
    payload = {
        "overall_pass": all_pass,
        "runtime_seconds": runtime_s,
        "items": {
            "fig1": res_fig1,
            "fig2": res_fig2,
            "fig3": res_fig3,
            "fig4": res_fig4,
            "table1": res_table1,
        },
    }
    with open(summary_path, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    print(f"\nSummary written to {summary_path}")
    return all_pass


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
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

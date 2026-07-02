"""
figures/make_figures.py
========================

Generates the 5 TMLR paper figures as vector PDFs, using ONLY data already
produced by the Stage-4..10 experiment scripts and committed under
stage*/results*.json in this repo. This script trains nothing, rolls out no
episodes, and adds no seeds -- it is pure post-hoc arithmetic and plotting.

The only "computation" beyond reading stored JSON is:
  (a) simple arithmetic on already-stored numbers -- e.g. deriving the
      Q-error / reachability-error decomposition (eps_Q, eps_D) from stored
      delta_exact / delta_learned values at two already-computed lambda grid
      points (a 2-point linear solve; both quantities are lambda-independent
      by construction, see stage9_common.margin_eval / margin_preservation_eval).
  (b) three instant, deterministic, zero-training calls to the repo's own
      EXACT-model primitives (q_reward_h, destroyed_mass from
      stage1_unified_validation) on the exact twins that stage9's own
      __main__ blocks already build every run -- needed because
      stage9_common.margin_rows only logs the LEARNED-model Q/D_w side, not
      the exact side, per environment.
No world model is retrained; no new MiniGrid/gridworld episode is executed.

Run:  python figures/make_figures.py
Output (this directory): fig_lambda_threshold.pdf, fig_margin_crossing.pdf,
    fig_error_decomposition.pdf, fig_localization.pdf, fig_repair.pdf
"""

import importlib
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in ("stage1_unified", "stage9_embodied_family", "stage4_modelbased"):
    _fp = os.path.join(ROOT, _p)
    if _fp not in sys.path:
        sys.path.insert(0, _fp)

from stage1_unified_validation import q_reward_h, destroyed_mass  # noqa: E402

# ---------------------------------------------------------------------
# Shared style: vector PDF, colorblind-safe categorical palette (dataviz
# skill reference palette), all in-figure text >= 9pt, linewidth >= 1.5.
# ---------------------------------------------------------------------
mpl.rcParams.update({
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,   # embed as real (editable) vector text, not Type3
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "svg.fonttype": "none",
})

# Categorical palette (light-mode, fixed order) -- blue/orange as the
# primary binary contrast so no figure relies on a red-green distinction.
C = dict(blue="#2a78d6", aqua="#1baf7a", yellow="#c98500", green="#008300",
          violet="#4a3aa7", red="#e34948", magenta="#e87ba4", orange="#eb6834",
          gridline="#e1e0d9", muted="#898781", secondary="#52514e")


def load(*parts):
    with open(os.path.join(ROOT, *parts)) as fh:
        return json.load(fh)


def savefig(fig, name):
    path = os.path.join(HERE, name)
    fig.canvas.draw()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("wrote", path)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=C["gridline"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# =======================================================================
# Figure 1 -- fig_lambda_threshold.pdf  (Theorem 1: lambda threshold)
# Source: stage9_embodied_family/results/stage9_results.json
#         (environments[*].margin_rows, .mechanism_per_seed, .recovery)
#         stage10_minigrid/results/stage10_results.json (same schema,
#         reused from stage9_common.evaluate_environment).
# =======================================================================

def fig1_lambda_threshold():
    d9 = load("stage9_embodied_family", "results", "stage9_results.json")
    d10 = load("stage10_minigrid", "results", "stage10_results.json")

    name_label = {
        "env1_doorkey_lava": "Env1 DoorKey-Lava",
        "env2_sokoban_barrier": "Env2 Sokoban-barrier",
        "env3_resource_depletion": "Env3 Resource-depletion",
    }
    envs = []
    for env in d9["environments"]:
        envs.append({
            "label": name_label[env["name"]],
            "a_decoy": env["config"]["a_decoy"],
            "margin_rows": env["margin_rows"],
            "mech": {r["seed"]: r for r in env["mechanism_per_seed"]},
            "lam_min": env["recovery"]["lam_min_hat"],
            "lam_star": env["recovery"]["lam_star"],
        })
    r10 = d10["result"]
    envs.append({
        "label": "Env4 MiniGrid (native)",
        "a_decoy": r10["config"]["a_decoy"],
        "margin_rows": r10["margin_rows"],
        "mech": {r["seed"]: r for r in r10["mechanism_per_seed"]},
        "lam_min": r10["recovery"]["lam_min_hat"],
        "lam_star": r10["recovery"]["lam_star"],
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15))

    # ---- left panel: representative env (Env1), raw lambda axis, return
    ax = axes[0]
    env = envs[0]
    seeds = sorted(set(env["mech"].keys())
                    & set(r["seed"] for r in env["margin_rows"]))
    lams = sorted(set(r["lam"] for r in env["margin_rows"]))
    R = []
    for sd in seeds:
        mech = env["mech"][sd]
        rows_sd = sorted([r for r in env["margin_rows"] if r["seed"] == sd],
                          key=lambda r: r["lam"])
        rets = [mech["R_reward_only_irr"] if r["action"] == env["a_decoy"]
                else mech["R_mrc_learned_irr"] for r in rows_sd]
        R.append(rets)
    R = np.array(R)
    mean_R, std_R = R.mean(axis=0), R.std(axis=0)
    lam_arr = np.array(lams)
    ax.plot(lam_arr, mean_R, color=C["blue"], marker="o", markersize=3,
             label=f"achieved return (mean, {len(seeds)} seeds)")
    ax.fill_between(lam_arr, mean_R - std_R, mean_R + std_R,
                     color=C["blue"], alpha=0.2, label="±1 std")
    ax.axvline(env["lam_min"], color="black", ls="--", lw=1.5,
                label=r"theoretical $\lambda_{\min}$" + f" = {env['lam_min']:.3f}")
    ax.axvline(env["lam_star"], color=C["orange"], ls=":", lw=1.8,
                label=r"observed flip $\lambda^\ast$" + f" = {env['lam_star']:.3f}")
    ax.set_xlabel(r"MRC weight $\lambda$")
    ax.set_ylabel("achieved return (irreversible twin)")
    ax.set_xlim(0, 1.5)
    style_axes(ax)
    ax.legend(loc="lower right", fontsize=7.6, framealpha=0.9)
    ax.text(0.40, 0.97, env["label"], transform=ax.transAxes, fontsize=8.5,
             color=C["secondary"], va="top")

    # ---- right panel: normalized lambda/lambda_min, all 4 envs
    ax = axes[1]
    colors_seq = [C["blue"], C["aqua"], C["violet"], C["orange"]]
    markers = ["o", "s", "^", "D"]
    for env, col, mk in zip(envs, colors_seq, markers):
        lams = sorted(set(r["lam"] for r in env["margin_rows"]))
        frac_safe = []
        for lam in lams:
            rows_l = [r for r in env["margin_rows"] if abs(r["lam"] - lam) < 1e-9]
            frac_safe.append(np.mean([r["action"] != env["a_decoy"] for r in rows_l]))
        lam_norm = np.array(lams) / env["lam_min"]
        ax.plot(lam_norm, frac_safe, color=col, marker=mk, markersize=3.2,
                 lw=1.6, label=env["label"])
    ax.axvline(1.0, color="black", ls="--", lw=1.5, label=r"$\lambda/\lambda_{\min}=1$")
    ax.set_xlabel(r"$\lambda / \lambda_{\min}$ (normalized)")
    ax.set_ylabel("fraction of seeds choosing safe action")
    ax.set_xlim(0, 3.5)
    ax.set_ylim(-0.03, 1.05)
    style_axes(ax)
    ax.legend(loc="lower right", fontsize=7.4, framealpha=0.9)

    fig.tight_layout()
    savefig(fig, "fig_lambda_threshold.pdf")


# =======================================================================
# Figure 2 -- fig_margin_crossing.pdf  (Theorem 2: margin-preservation)
# Source: stage10_minigrid/results/stage10_results.json
#         (margin_preservation_rows: delta_exact, delta_learned, preserved)
# =======================================================================

def fig2_margin_crossing():
    d10 = load("stage10_minigrid", "results", "stage10_results.json")
    rows = d10["margin_preservation_rows"]
    eps = np.array([r["delta_learned"] - r["delta_exact"] for r in rows])
    dM = np.array([r["delta_exact"] for r in rows])
    preserved = np.array([r["preserved"] for r in rows])
    seeds = sorted(set(r["seed"] for r in rows))
    lams = sorted(set(r["lam"] for r in rows))

    # eps_score is tiny (WM Q/reach error) while Delta_M spans a wide range
    # over the lambda sweep (0.05 at lambda=0 down to -0.75 at lambda=1.5);
    # a flip is only ever possible where |Delta_M| is small, i.e. near
    # lambda_min. The main panel zooms there (where the boundary is
    # actually legible); the inset shows the untouched full range so the
    # "zero violations, always" claim is visibly not an artifact of cropping.
    ZOOM = 0.20

    def scatter_on(axp, mask_p, mask_f):
        axp.scatter(eps[mask_p], dM[mask_p], s=14, c=C["blue"], marker="o",
                     alpha=0.7, edgecolors="none", zorder=3)
        axp.scatter(eps[mask_f], dM[mask_f], s=42, c=C["orange"], marker="X",
                     alpha=0.95, edgecolors="black", linewidths=0.5, zorder=4)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))

    in_zoom = np.abs(dM) <= ZOOM
    lo_x, hi_x = float(eps.min()) - 0.004, float(eps.max()) + 0.004
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    xx = np.linspace(lo_x, hi_x, 50)
    ax.plot(xx, -xx, color="black", ls="--", lw=1.5,
             label=r"boundary $\varepsilon_{\rm score}=-\Delta_M$", zorder=2)
    scatter_on(ax, preserved & in_zoom, (~preserved) & in_zoom)
    ax.axhline(0, color=C["gridline"], lw=0.8, zorder=1)
    ax.axvline(0, color=C["gridline"], lw=0.8, zorder=1)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(-ZOOM, ZOOM)
    ax.set_xlabel(r"combined-score error $\varepsilon_{\rm score}=\hat\Delta-\Delta$")
    ax.set_ylabel(r"exact margin $\Delta_M$ (signed)")
    style_axes(ax)
    n_off = int((~in_zoom).sum())
    ax.scatter([], [], s=14, c=C["blue"], marker="o", label=f"preserved (n={int(preserved.sum())})")
    ax.scatter([], [], s=42, c=C["orange"], marker="X", edgecolors="black",
                linewidths=0.5, label=f"flipped (n={int((~preserved).sum())})")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.text(0.98, 0.03,
             f"{len(seeds)} seeds x {len(lams)} " + r"$\lambda\in[0,1.5]$" +
             f" = {len(rows)} decisions (native MiniGrid, $s_0$)\n"
             f"main panel zoomed to $|\\Delta_M|\\leq{ZOOM:g}$; "
             f"{n_off} more preserved points off-frame\n"
             "(see inset for the untouched full range)",
             transform=ax.transAxes, fontsize=7.2, color=C["secondary"],
             ha="right", va="bottom")

    # ---- inset: full untouched range, same axes/markers, no legend ----
    axi = ax.inset_axes([0.62, 0.60, 0.36, 0.36])
    lo_full = float(min(eps.min(), dM.min())) * 1.15
    hi_full = float(max(eps.max(), dM.max())) * 1.15
    xxf = np.linspace(lo_full, hi_full, 50)
    axi.plot(xxf, -xxf, color="black", ls="--", lw=1.0, zorder=2)
    scatter_on(axi, preserved, ~preserved)
    axi.set_xlim(lo_full, hi_full)
    axi.set_ylim(lo_full, hi_full)
    axi.tick_params(labelsize=6.5)
    axi.set_title("full range", fontsize=7.2, pad=2)
    for spine in axi.spines.values():
        spine.set_linewidth(0.6)

    fig.tight_layout()
    savefig(fig, "fig_margin_crossing.pdf")


# =======================================================================
# Figure 3 -- fig_error_decomposition.pdf  (support for Theorem 2)
# Source: stage9_results.json margin_rows (learned side) + exact q_reward_h
#         / destroyed_mass on the same exact twins (exact side, uncomputed
#         in the stored JSON); stage10 margin_preservation_rows (both sides
#         already stored -- 2-point solve, no recompute needed).
# =======================================================================

def fig3_error_decomposition():
    d9 = load("stage9_embodied_family", "results", "stage9_results.json")
    d10 = load("stage10_minigrid", "results", "stage10_results.json")

    labels = {"env1_doorkey_lava": "Env1\nDoorKey-Lava",
               "env2_sokoban_barrier": "Env2\nSokoban",
               "env3_resource_depletion": "Env3\nResource"}
    groups = []
    for mod_name in ("env1_doorkey_lava", "env2_sokoban_barrier",
                       "env3_resource_depletion"):
        mod = importlib.import_module(mod_name)
        spec = mod.SPEC
        mdp = spec.build_twin("irreversible")
        Qd_e = q_reward_h(mdp, spec.S0, spec.a_decoy, spec.H)
        Qs_e = q_reward_h(mdp, spec.S0, spec.a_safe, spec.H)
        Dd_e = destroyed_mass(mdp, spec.S0, spec.a_decoy)
        Ds_e = destroyed_mass(mdp, spec.S0, spec.a_safe)
        A_e, B_e = Qd_e - Qs_e, Dd_e - Ds_e
        env = next(e for e in d9["environments"] if e["name"] == mod_name)
        seeds = sorted(set(r["seed"] for r in env["margin_rows"]))
        epsQ, epsD = [], []
        for sd in seeds:
            r0 = next(r for r in env["margin_rows"] if r["seed"] == sd)
            A_l = r0["Q_decoy"] - r0["Q_safe"]
            B_l = r0["Dw_hat_decoy"] - r0["Dw_hat_safe"]
            epsQ.append(A_l - A_e)
            epsD.append(B_e - B_l)
        groups.append((labels[mod_name], np.array(epsQ), np.array(epsD)))

    mp_rows = d10["margin_preservation_rows"]
    seeds10 = sorted(set(r["seed"] for r in mp_rows))
    epsQ10, epsD10 = [], []
    for sd in seeds10:
        rs = [r for r in mp_rows if r["seed"] == sd]
        r0 = next(r for r in rs if abs(r["lam"] - 0.0) < 1e-9)
        r1 = next(r for r in rs if abs(r["lam"] - 1.0) < 1e-9)
        A_e, A_l = r0["delta_exact"], r0["delta_learned"]
        B_e = A_e - r1["delta_exact"]
        B_l = A_l - r1["delta_learned"]
        epsQ10.append(A_l - A_e)
        epsD10.append(B_e - B_l)
    groups.append(("Env4\nMiniGrid", np.array(epsQ10), np.array(epsD10)))

    LAM_REF = 1.0
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    x = np.arange(len(groups))
    w = 0.32
    epsQ_mean = [float(np.mean(np.abs(g[1]))) for g in groups]
    epsQ_std = [float(np.std(np.abs(g[1]))) for g in groups]
    epsD_mean = [float(np.mean(np.abs(LAM_REF * g[2]))) for g in groups]
    epsD_std = [float(np.std(np.abs(LAM_REF * g[2]))) for g in groups]
    ax.bar(x - w / 2, epsQ_mean, width=w, yerr=epsQ_std, capsize=3,
            color=C["blue"], label=r"$|\varepsilon_Q|$ (reward-error term)")
    ax.bar(x + w / 2, epsD_mean, width=w, yerr=epsD_std, capsize=3,
            color=C["orange"],
            label=r"$|\lambda\cdot\varepsilon_D|$ at $\lambda{=}1$ (reachability-error term)")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(r"error magnitude at decision point $s_0$")
    n_seeds_txt = ", ".join(f"{g[0].split(chr(10))[0]}:{len(g[1])}" for g in groups)
    ax.text(0.99, 0.97, f"seeds -- {n_seeds_txt}", transform=ax.transAxes,
             ha="right", va="top", fontsize=7.4, color=C["secondary"])
    style_axes(ax)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig_error_decomposition.pdf")
    return groups


# =======================================================================
# Figure 4 -- fig_localization.pdf  (Localization Lemma)
# Source: stage8_aaai/results/block3_results.json (per_run: three
# perturbation families x strength x seed, on the reversible twin where
# the TRUE decision-point cost gap is exactly 0).
# =======================================================================

def fig4_localization():
    d8b3 = load("stage8_aaai", "results", "block3_results.json")
    per_run = d8b3["per_run"]
    family_order = [
        ("global_random", "(a) global\ndiffuse noise"),
        ("off_decision", "(b) off-decision\ndirectional error"),
        ("decision_state", "(c) decision-point\ndirectional error"),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    means, stds, labels = [], [], []
    colors_bar = [C["blue"], C["aqua"], C["orange"]]
    for fam, lbl in family_order:
        rows_fam = [r for r in per_run if r["family"] == fam]
        max_strength = max(r["strength"] for r in rows_fam)
        rows_max = [r for r in rows_fam if r["strength"] == max_strength]
        errs = [abs(r["Dw_hat_decoy_s0"] - r["Dw_hat_safe_s0"]) for r in rows_max]
        means.append(float(np.mean(errs)))
        stds.append(float(np.std(errs)))
        labels.append(lbl + f"\n(strength={max_strength:.1f})")
    x = np.arange(3)
    ax.bar(x, means, yerr=stds, capsize=4, color=colors_bar, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"decision-point cost-gap error $|\hat\Delta D_w-\Delta D_w|$")
    ax.text(0.02, 0.97,
             "n=5 seeds per bar; strongest tested level per family\n"
             "reversible twin, true value = 0",
             transform=ax.transAxes, fontsize=7.6, va="top", color=C["secondary"])
    style_axes(ax)
    fig.tight_layout()
    savefig(fig, "fig_localization.pdf")


# =======================================================================
# Figure 5 -- fig_repair.pdf  (Proposition 3 + reachability-consistency)
# Source: stage8_aaai/results/block4_results.json (per_run: baseline /
# oracle-DA(exact-auditing) / non_oracle-DA(rollout-observed) x rcp x seed,
# Dw_rev on the reversible twin, true value = 0); exact flip threshold from
# q_reward_h on the exact reversible twin (stage4_modelbased primitives).
# =======================================================================

def fig5_repair():
    d8b4 = load("stage8_aaai", "results", "block4_results.json")
    per_run = d8b4["per_run"]
    rcp_levels = sorted(set(r["rcp"] for r in per_run))

    from stage4_modelbased_planning import build_lava_gridworld, S0  # noqa: E402
    mdp_rev = build_lava_gridworld(k=3, m=4, r_d=1.0, r_g=1.0, gamma=0.9,
                                     mode="reversible")
    threshold = (q_reward_h(mdp_rev, S0, "a_decoy", 4)
                  - q_reward_h(mdp_rev, S0, "a_safe", 4))

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    methods = [
        ("baseline", "pre-repair (baseline WM)", C["orange"], "o"),
        ("oracle", "post-repair, exact-auditing labels", C["blue"], "s"),
        ("non_oracle", "post-repair, rollout-observed labels", C["violet"], "^"),
    ]
    for key, lbl, col, mk in methods:
        means, stds = [], []
        for rcp in rcp_levels:
            vals = [r[key]["Dw_rev"] for r in per_run if r["rcp"] == rcp]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        means_a, stds_a = np.array(means), np.array(stds)
        ax.plot(rcp_levels, means_a, color=col, marker=mk, markersize=5,
                 lw=1.9, label=lbl)
        ax.fill_between(rcp_levels, means_a - stds_a, means_a + stds_a,
                          color=col, alpha=0.15)
    ax.axhline(threshold, color="black", ls="--", lw=1.5,
                label=r"flip threshold $(\Delta_{\rm margin}-|\varepsilon_Q|)/\lambda \approx$"
                      f" {threshold:.2f}")
    ax.set_xlabel("recover-transition corruption probability (rcp)")
    ax.set_ylabel(r"decision-point cost-gap error $|\hat\Delta D_w-\Delta D_w|$"
                   "\n(reversible twin; true value = 0)")
    ax.text(0.02, 0.97, "n=3 seeds per point", transform=ax.transAxes,
             fontsize=7.6, va="top", color=C["secondary"])
    style_axes(ax)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig_repair.pdf")
    return threshold


if __name__ == "__main__":
    fig1_lambda_threshold()
    fig2_margin_crossing()
    fig3_error_decomposition()
    fig4_localization()
    fig5_repair()
    print("\nAll 5 figures written to", HERE)

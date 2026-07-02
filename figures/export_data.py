"""
figures/export_data.py
=======================

Exports the exact data underlying each of the 5 TMLR figures (see
make_figures.py) as tidy CSVs under figures/data/. Reuses make_figures.py's
loaders and the same exact-model primitives (q_reward_h, destroyed_mass) --
no experiment is re-run, no new seed is trained.

Also answers "how many flip samples, and can we get points closer to the
boundary" for fig_margin_crossing: delta_exact(lambda) and
delta_learned(lambda) are both AFFINE in lambda for a fixed seed (Q_decoy,
Q_safe, D_w_hat_decoy, D_w_hat_safe do not depend on lambda -- see
stage9_common.margin_preservation_eval). So the two free parameters per
seed (intercept, slope) are already fully pinned down by any two stored
lambda rows, and evaluating the SAME affine function at extra lambda values
is exact interpolation of an already-fully-determined function, not a new
simulation. fig2_margin_crossing_finegrid.csv does exactly this on a 0.001
step grid (vs. the original 0.05 step / 31 points) and is cross-checked
against the original 31 stored points at matching lambda.

Run: python figures/export_data.py
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from make_figures import (  # noqa: E402
    ROOT, load, q_reward_h, destroyed_mass,
)

DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def write_csv(name, fieldnames, rows):
    path = os.path.join(DATA_DIR, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({len(rows)} rows)")


# =======================================================================
# fig1_lambda_threshold.csv
# =======================================================================

def export_fig1():
    d9 = load("stage9_embodied_family", "results", "stage9_results.json")
    d10 = load("stage10_minigrid", "results", "stage10_results.json")

    name_label = {
        "env1_doorkey_lava": "Env1_DoorKey-Lava",
        "env2_sokoban_barrier": "Env2_Sokoban-barrier",
        "env3_resource_depletion": "Env3_Resource-depletion",
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
        "label": "Env4_MiniGrid",
        "a_decoy": r10["config"]["a_decoy"],
        "margin_rows": r10["margin_rows"],
        "mech": {r["seed"]: r for r in r10["mechanism_per_seed"]},
        "lam_min": r10["recovery"]["lam_min_hat"],
        "lam_star": r10["recovery"]["lam_star"],
    })

    rows = []
    for env in envs:
        for r in env["margin_rows"]:
            sd = r["seed"]
            is_safe = r["action"] != env["a_decoy"]
            mech = env["mech"].get(sd)
            achieved_return = None
            if mech is not None:
                achieved_return = (mech["R_mrc_learned_irr"] if is_safe
                                     else mech["R_reward_only_irr"])
            rows.append({
                "env": env["label"], "seed": sd, "lam": r["lam"],
                "lam_min_hat": env["lam_min"], "lam_star": env["lam_star"],
                "lam_over_lam_min": r["lam"] / env["lam_min"],
                "a_decoy": env["a_decoy"], "action_chosen": r["action"],
                "is_safe_action": is_safe,
                "achieved_return_irr_twin": achieved_return,
            })
    write_csv("fig1_lambda_threshold.csv",
               ["env", "seed", "lam", "lam_min_hat", "lam_star",
                "lam_over_lam_min", "a_decoy", "action_chosen",
                "is_safe_action", "achieved_return_irr_twin"], rows)


# =======================================================================
# fig2_margin_crossing.csv (raw, as stored) + finegrid (exact refinement)
# =======================================================================

def export_fig2():
    d10 = load("stage10_minigrid", "results", "stage10_results.json")
    raw = d10["margin_preservation_rows"]

    rows = []
    for r in raw:
        eps_score = r["delta_learned"] - r["delta_exact"]
        rows.append({
            "seed": r["seed"], "lam": r["lam"],
            "delta_exact": r["delta_exact"], "delta_learned": r["delta_learned"],
            "eps_score": eps_score, "delta_M": r["delta_exact"],
            "exact_decision": r["exact_decision"],
            "learned_decision": r["learned_decision"],
            "score_error": r["score_error"], "exact_margin": r["exact_margin"],
            "within_margin": r["within_margin"], "preserved": r["preserved"],
        })
    write_csv("fig2_margin_crossing.csv",
               ["seed", "lam", "delta_exact", "delta_learned", "eps_score",
                "delta_M", "exact_decision", "learned_decision",
                "score_error", "exact_margin", "within_margin", "preserved"],
               rows)
    n_flip = sum(1 for r in rows if not r["preserved"])
    print(f"  -> {n_flip} flipped / {len(rows)} total "
          f"(original 0.05-step, 31-lambda grid x 5 seeds)")

    # ---- finegrid: exact affine reconstruction, step 0.001 -----------
    seeds = sorted(set(r["seed"] for r in raw))
    fine_lams = np.round(np.arange(0.0, 1.5 + 1e-9, 0.001), 3)
    fine_rows = []
    cross_check_errs = []
    for sd in seeds:
        rs = sorted([r for r in raw if r["seed"] == sd], key=lambda r: r["lam"])
        r0 = next(r for r in rs if abs(r["lam"] - 0.0) < 1e-9)
        r1 = next(r for r in rs if abs(r["lam"] - 1.0) < 1e-9)
        A_e, A_l = r0["delta_exact"], r0["delta_learned"]
        B_e = A_e - r1["delta_exact"]
        B_l = A_l - r1["delta_learned"]
        stored_by_lam = {round(r["lam"], 3): r for r in rs}
        for lam in fine_lams:
            lam = float(lam)
            d_e = A_e - lam * B_e
            d_l = A_l - lam * B_l
            eps_score = d_l - d_e
            dec_e = "decoy" if d_e > 0 else "safe"
            dec_l = "decoy" if d_l > 0 else "safe"
            preserved = dec_e == dec_l
            margin = abs(d_e)
            fine_rows.append({
                "seed": sd, "lam": lam, "delta_exact": d_e, "delta_learned": d_l,
                "eps_score": eps_score, "delta_M": d_e,
                "exact_decision": dec_e, "learned_decision": dec_l,
                "score_error": abs(eps_score), "exact_margin": margin,
                "within_margin": abs(eps_score) < margin, "preserved": preserved,
            })
            if lam in stored_by_lam:
                cross_check_errs.append(abs(d_e - stored_by_lam[lam]["delta_exact"]))
    write_csv("fig2_margin_crossing_finegrid.csv",
               ["seed", "lam", "delta_exact", "delta_learned", "eps_score",
                "delta_M", "exact_decision", "learned_decision",
                "score_error", "exact_margin", "within_margin", "preserved"],
               fine_rows)
    n_flip_fine = sum(1 for r in fine_rows if not r["preserved"])
    max_cc_err = max(cross_check_errs) if cross_check_errs else float("nan")
    print(f"  -> finegrid (step=0.001): {n_flip_fine} flipped / {len(fine_rows)} "
          f"total; cross-check max |delta_exact diff| vs stored = {max_cc_err:.2e}")

    # per-seed flip-interval summary (closed form, exact)
    interval_rows = []
    for sd in seeds:
        rs = sorted([r for r in raw if r["seed"] == sd], key=lambda r: r["lam"])
        r0 = next(r for r in rs if abs(r["lam"] - 0.0) < 1e-9)
        r1 = next(r for r in rs if abs(r["lam"] - 1.0) < 1e-9)
        A_e, A_l = r0["delta_exact"], r0["delta_learned"]
        B_e = A_e - r1["delta_exact"]
        B_l = A_l - r1["delta_learned"]
        lam_min_exact = A_e / B_e
        lam_min_learned = A_l / B_l
        interval_rows.append({
            "seed": sd, "lam_min_exact": lam_min_exact,
            "lam_min_learned": lam_min_learned,
            "flip_interval_lo": min(lam_min_exact, lam_min_learned),
            "flip_interval_hi": max(lam_min_exact, lam_min_learned),
            "flip_interval_width": abs(lam_min_learned - lam_min_exact),
        })
    write_csv("fig2_flip_intervals_per_seed.csv",
               ["seed", "lam_min_exact", "lam_min_learned",
                "flip_interval_lo", "flip_interval_hi",
                "flip_interval_width"], interval_rows)


# =======================================================================
# fig3_error_decomposition.csv
# =======================================================================

def export_fig3():
    import importlib
    d9 = load("stage9_embodied_family", "results", "stage9_results.json")
    d10 = load("stage10_minigrid", "results", "stage10_results.json")
    labels = {"env1_doorkey_lava": "Env1_DoorKey-Lava",
               "env2_sokoban_barrier": "Env2_Sokoban",
               "env3_resource_depletion": "Env3_Resource"}
    rows = []
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
        for sd in seeds:
            r0 = next(r for r in env["margin_rows"] if r["seed"] == sd)
            A_l = r0["Q_decoy"] - r0["Q_safe"]
            B_l = r0["Dw_hat_decoy"] - r0["Dw_hat_safe"]
            rows.append({
                "env": labels[mod_name], "seed": sd,
                "A_exact_Q_gap": A_e, "B_exact_D_gap": B_e,
                "A_learned_Q_gap": A_l, "B_learned_D_gap": B_l,
                "eps_Q": A_l - A_e, "eps_D": B_e - B_l, "lambda_ref": 1.0,
                "abs_eps_Q": abs(A_l - A_e),
                "abs_lambda_eps_D": abs(1.0 * (B_e - B_l)),
            })

    mp_rows = d10["margin_preservation_rows"]
    seeds10 = sorted(set(r["seed"] for r in mp_rows))
    for sd in seeds10:
        rs = [r for r in mp_rows if r["seed"] == sd]
        r0 = next(r for r in rs if abs(r["lam"] - 0.0) < 1e-9)
        r1 = next(r for r in rs if abs(r["lam"] - 1.0) < 1e-9)
        A_e, A_l = r0["delta_exact"], r0["delta_learned"]
        B_e = A_e - r1["delta_exact"]
        B_l = A_l - r1["delta_learned"]
        rows.append({
            "env": "Env4_MiniGrid", "seed": sd,
            "A_exact_Q_gap": A_e, "B_exact_D_gap": B_e,
            "A_learned_Q_gap": A_l, "B_learned_D_gap": B_l,
            "eps_Q": A_l - A_e, "eps_D": B_e - B_l, "lambda_ref": 1.0,
            "abs_eps_Q": abs(A_l - A_e),
            "abs_lambda_eps_D": abs(1.0 * (B_e - B_l)),
        })
    write_csv("fig3_error_decomposition.csv",
               ["env", "seed", "A_exact_Q_gap", "B_exact_D_gap",
                "A_learned_Q_gap", "B_learned_D_gap", "eps_Q", "eps_D",
                "lambda_ref", "abs_eps_Q", "abs_lambda_eps_D"], rows)


# =======================================================================
# fig4_localization.csv
# =======================================================================

def export_fig4():
    d8b3 = load("stage8_aaai", "results", "block3_results.json")
    rows = []
    for r in d8b3["per_run"]:
        rows.append({
            "family": r["family"], "strength": r["strength"], "seed": r["seed"],
            "Dw_hat_decoy_s0": r["Dw_hat_decoy_s0"],
            "Dw_hat_safe_s0": r["Dw_hat_safe_s0"],
            "abs_cost_gap_error": abs(r["Dw_hat_decoy_s0"] - r["Dw_hat_safe_s0"]),
            "s0_flipped": r["s0_flipped"],
            "used_in_bar_chart": r["strength"] == max(
                x["strength"] for x in d8b3["per_run"] if x["family"] == r["family"]),
        })
    write_csv("fig4_localization.csv",
               ["family", "strength", "seed", "Dw_hat_decoy_s0",
                "Dw_hat_safe_s0", "abs_cost_gap_error", "s0_flipped",
                "used_in_bar_chart"], rows)


# =======================================================================
# fig5_repair.csv
# =======================================================================

def export_fig5():
    sys.path.insert(0, os.path.join(ROOT, "stage4_modelbased"))
    from stage4_modelbased_planning import build_lava_gridworld, S0  # noqa: E402
    mdp_rev = build_lava_gridworld(k=3, m=4, r_d=1.0, r_g=1.0, gamma=0.9,
                                     mode="reversible")
    threshold = (q_reward_h(mdp_rev, S0, "a_decoy", 4)
                  - q_reward_h(mdp_rev, S0, "a_safe", 4))

    d8b4 = load("stage8_aaai", "results", "block4_results.json")
    rows = []
    for r in d8b4["per_run"]:
        for method in ("baseline", "oracle", "non_oracle"):
            m = r[method]
            rows.append({
                "method": method, "rcp": r["rcp"], "seed": r["seed"],
                "Dw_rev_abs_error": m["Dw_rev"],
                "collapse_ratio": m["collapse_ratio"],
                "flip_threshold_exact": threshold,
                "above_threshold": m["Dw_rev"] > threshold,
            })
    write_csv("fig5_repair.csv",
               ["method", "rcp", "seed", "Dw_rev_abs_error", "collapse_ratio",
                "flip_threshold_exact", "above_threshold"], rows)


if __name__ == "__main__":
    export_fig1()
    export_fig2()
    export_fig3()
    export_fig4()
    export_fig5()
    print("\nAll CSVs written to", DATA_DIR)

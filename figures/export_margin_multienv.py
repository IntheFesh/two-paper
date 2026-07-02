"""
figures/export_margin_multienv.py
==================================

Extends the fig_margin_crossing dense-boundary evidence from the single
native-MiniGrid decision point (fig2_margin_crossing_finegrid.csv) to all
4 environments (Env1 DoorKey-Lava, Env2 Sokoban-barrier, Env3
Resource-depletion from Stage-9's embodied family, plus Env4 MiniGrid from
Stage-10). Same principle as before: for a fixed seed and decision state,

    delta_M(lam)     = A     - lam * B       (exact)
    delta_M_hat(lam) = A_hat - lam * B_hat   (learned)

are both AFFINE in lambda, because Q^rew and D_hat_w do not themselves
depend on lambda -- lambda only enters as the weight on D_w in
Q^H_MRC = Q^rew - lambda * D_w. The two free parameters per seed are
therefore already fully pinned down by the existing (already-run)
experiments; evaluating the same affine function on a finer lambda grid is
exact interpolation, NOT a new simulation, no new randomness.

Data sources (already committed, no experiment re-run):
  - Env1/Env2/Env3: stage9_embodied_family/results/stage9_results.json,
    environments[*].margin_rows (learned side: reward_margin, Dw_hat_decoy,
    Dw_hat_safe are already the exact affine coefficients A_hat, B_hat --
    stored directly per row, constant across lambda by construction).
    Exact side (A, B) is NOT stored in that JSON; it is obtained via the
    same zero-training, deterministic calls already used for
    fig_error_decomposition: q_reward_h / destroyed_mass from
    stage1_unified_validation, applied to each env module's own
    build_twin("irreversible") -- the exact twin these scripts already
    build every run.
  - Env4 MiniGrid: stage10_minigrid/results/stage10_results.json,
    margin_preservation_rows already stores BOTH delta_exact and
    delta_learned per (seed, lam) row directly.

Cross-validation (mandatory, per task spec): the fitted/derived affine
coefficients are evaluated at every lambda point that was ACTUALLY stored
in the original (coarse) experiment output, and compared row-by-row
against the stored margin values. If any environment's max deviation is
not at floating-point noise level (~1e-10 or smaller), that is reported
explicitly and NOT papered over -- it would mean the affine assumption or
the coefficient extraction is broken for that environment.

Run: python figures/export_margin_multienv.py
"""

import csv
import importlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from make_figures import ROOT, load, q_reward_h, destroyed_mass  # noqa: E402

_STAGE9_DIR = os.path.join(ROOT, "stage9_embodied_family")
if _STAGE9_DIR not in sys.path:
    sys.path.insert(0, _STAGE9_DIR)

DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FINE_LAMS = np.round(np.arange(0.0, 1.5 + 1e-9, 0.001), 3)


def write_csv(name, fieldnames, rows):
    path = os.path.join(DATA_DIR, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({len(rows)} rows)")


def decision(delta):
    return "decoy" if delta > 0 else "safe"


# =======================================================================
# Step 1: gather, per environment/seed, the affine coefficients (A, B) for
# the EXACT margin and (A_hat, B_hat) for the LEARNED margin, plus the
# original stored (lam -> delta) points used for cross-validation.
# =======================================================================

def gather_env1_2_3():
    """Env1/Env2/Env3 (Stage-9 embodied family)."""
    d9 = load("stage9_embodied_family", "results", "stage9_results.json")
    labels = {"env1_doorkey_lava": "Env1_DoorKey-Lava",
               "env2_sokoban_barrier": "Env2_Sokoban",
               "env3_resource_depletion": "Env3_Resource"}
    out = {}
    for mod_name, label in labels.items():
        mod = importlib.import_module(mod_name)
        spec = mod.SPEC
        mdp = spec.build_twin("irreversible")
        Qd_e = q_reward_h(mdp, spec.S0, spec.a_decoy, spec.H)
        Qs_e = q_reward_h(mdp, spec.S0, spec.a_safe, spec.H)
        Dd_e = destroyed_mass(mdp, spec.S0, spec.a_decoy)
        Ds_e = destroyed_mass(mdp, spec.S0, spec.a_safe)
        A_e, B_e = Qd_e - Qs_e, Dd_e - Ds_e

        env_json = next(e for e in d9["environments"] if e["name"] == mod_name)
        rows = env_json["margin_rows"]
        seeds = sorted(set(r["seed"] for r in rows))

        per_seed = {}
        for sd in seeds:
            rs_sd = sorted([r for r in rows if r["seed"] == sd],
                             key=lambda r: r["lam"])
            # Two-point fit of the LEARNED affine line using the stored
            # delta_learned = reward_margin - cost_gap at two distinct
            # stored lambdas (lam=0 and the largest stored lambda).
            r0 = rs_sd[0]
            r1 = rs_sd[-1]
            d0 = r0["reward_margin"] - r0["cost_gap"]
            d1 = r1["reward_margin"] - r1["cost_gap"]
            lam0, lam1 = r0["lam"], r1["lam"]
            B_hat = (d0 - d1) / (lam1 - lam0)
            A_hat = d0 + B_hat * lam0
            # Directly-stored coefficients, for an independent check that
            # the 2-point fit recovers exactly what margin_eval computed.
            A_hat_direct = r0["reward_margin"]
            B_hat_direct = r0["Dw_hat_decoy"] - r0["Dw_hat_safe"]
            fit_vs_direct_err = max(abs(A_hat - A_hat_direct),
                                      abs(B_hat - B_hat_direct))
            per_seed[sd] = {
                "A_hat": A_hat, "B_hat": B_hat,
                "fit_vs_direct_err": fit_vs_direct_err,
                "stored_points": [(r["lam"],
                                     r["reward_margin"] - r["cost_gap"])
                                    for r in rs_sd],
            }
        out[label] = {"A_e": A_e, "B_e": B_e, "per_seed": per_seed}
    return out


def gather_env4():
    """Env4 MiniGrid (Stage-10) -- both sides already stored per row."""
    d10 = load("stage10_minigrid", "results", "stage10_results.json")
    rows = d10["margin_preservation_rows"]
    seeds = sorted(set(r["seed"] for r in rows))
    per_seed = {}
    A_e = B_e = None
    for sd in seeds:
        rs_sd = sorted([r for r in rows if r["seed"] == sd],
                         key=lambda r: r["lam"])
        r0, r1 = rs_sd[0], rs_sd[-1]
        lam0, lam1 = r0["lam"], r1["lam"]
        # exact side
        Be = (r0["delta_exact"] - r1["delta_exact"]) / (lam1 - lam0)
        Ae = r0["delta_exact"] + Be * lam0
        if A_e is None:
            A_e, B_e = Ae, Be
        # learned side
        B_hat = (r0["delta_learned"] - r1["delta_learned"]) / (lam1 - lam0)
        A_hat = r0["delta_learned"] + B_hat * lam0
        per_seed[sd] = {
            "A_hat": A_hat, "B_hat": B_hat,
            "fit_vs_direct_err": 0.0,  # nothing "direct" to compare here
            "stored_points": [(r["lam"], r["delta_learned"]) for r in rs_sd],
            "stored_points_exact": [(r["lam"], r["delta_exact"]) for r in rs_sd],
        }
    return {"Env4_MiniGrid": {"A_e": A_e, "B_e": B_e, "per_seed": per_seed}}


# =======================================================================
# Step 2/3: dense grid + cross-validation
# =======================================================================

def process_env(label, env_data):
    A_e, B_e = env_data["A_e"], env_data["B_e"]
    fine_rows = []
    interval_rows = []
    cross_check = []
    n_flip = 0
    n_violations = 0
    violation_examples = []

    for sd, sdat in env_data["per_seed"].items():
        A_hat, B_hat = sdat["A_hat"], sdat["B_hat"]

        # ---- mandatory cross-validation against stored grid points ----
        for lam, stored_delta in sdat["stored_points"]:
            recon = A_hat - B_hat * lam
            cross_check.append(abs(recon - stored_delta))
        if "stored_points_exact" in sdat:
            for lam, stored_delta in sdat["stored_points_exact"]:
                recon_e = A_e - B_e * lam
                cross_check.append(abs(recon_e - stored_delta))
        if sdat["fit_vs_direct_err"] > 0:
            cross_check.append(sdat["fit_vs_direct_err"])

        lam_min_exact = A_e / B_e
        lam_min_learned = A_hat / B_hat
        interval_rows.append({
            "env": label, "seed": sd,
            "lam_min_exact": lam_min_exact,
            "lam_min_learned": lam_min_learned,
            "flip_interval_lo": min(lam_min_exact, lam_min_learned),
            "flip_interval_hi": max(lam_min_exact, lam_min_learned),
            "flip_interval_width": abs(lam_min_learned - lam_min_exact),
        })

        for lam in FINE_LAMS:
            lam = float(lam)
            d_e = A_e - lam * B_e
            d_l = A_hat - lam * B_hat
            eps_score = d_l - d_e
            dec_e = decision(d_e)
            dec_l = decision(d_l)
            preserved = dec_e == dec_l
            margin = abs(d_e)
            within_margin = abs(eps_score) < margin
            if not preserved:
                n_flip += 1
            # Violation of the margin-preservation theorem's sufficient
            # condition: within_margin (|eps_score| < |delta_M|) implies
            # preserved. A violation is within_margin=True but the
            # decision flipped anyway. (The exact-boundary side check is
            # definitionally identical to `preserved` itself -- dec_l is
            # computed from d_l = d_e + eps_score -- so it is not an
            # independent test and is not counted separately here.)
            if within_margin and not preserved:
                n_violations += 1
                if len(violation_examples) < 5:
                    violation_examples.append(
                        dict(env=label, seed=sd, lam=lam, delta_exact=d_e,
                              delta_learned=d_l, eps_score=eps_score,
                              exact_margin=margin))
            fine_rows.append({
                "env": label, "seed": sd, "lam": lam,
                "delta_exact": d_e, "delta_learned": d_l,
                "eps_score": eps_score, "delta_M": d_e,
                "exact_decision": dec_e, "learned_decision": dec_l,
                "preserved": preserved,
            })

    max_cc = max(cross_check) if cross_check else float("nan")
    return {
        "fine_rows": fine_rows, "interval_rows": interval_rows,
        "n_total": len(fine_rows), "n_flip": n_flip,
        "n_violations": n_violations, "violation_examples": violation_examples,
        "max_cross_check_err": max_cc, "n_seeds": len(env_data["per_seed"]),
    }


def main():
    envs = {}
    envs.update(gather_env1_2_3())
    envs.update(gather_env4())

    all_fine_rows = []
    all_interval_rows = []
    summary = []
    for label, env_data in envs.items():
        res = process_env(label, env_data)
        all_fine_rows.extend(res["fine_rows"])
        all_interval_rows.extend(res["interval_rows"])
        summary.append((label, res))
        print(f"[{label}] seeds={res['n_seeds']}  total={res['n_total']}  "
              f"flips={res['n_flip']}  violations={res['n_violations']}  "
              f"max_cross_check_err={res['max_cross_check_err']:.3e}")
        if res["violation_examples"]:
            print("  VIOLATION EXAMPLES:")
            for v in res["violation_examples"]:
                print("   ", v)

    write_csv("fig2_margin_crossing_multienv_finegrid.csv",
               ["env", "seed", "lam", "delta_exact", "delta_learned",
                "eps_score", "delta_M", "exact_decision", "learned_decision",
                "preserved"], all_fine_rows)
    write_csv("fig2_flip_intervals_multienv.csv",
               ["env", "seed", "lam_min_exact", "lam_min_learned",
                "flip_interval_lo", "flip_interval_hi",
                "flip_interval_width"], all_interval_rows)

    total_flip = sum(r["n_flip"] for _, r in summary)
    total_viol = sum(r["n_violations"] for _, r in summary)
    total_n = sum(r["n_total"] for _, r in summary)
    print(f"\nTOTAL: {total_n} decisions across {len(summary)} environments, "
          f"{total_flip} flipped, {total_viol} violations")
    return summary


if __name__ == "__main__":
    main()

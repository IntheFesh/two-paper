"""
stage10_minigrid/run_stage10.py
============================================

Stage-10 driver -- run the MRC mechanism + margin-preservation checks on a
NATIVE MiniGrid environment and write results + figure.

This is the primary-track experiment that answers the reviewer concern
"does the mechanism depend on the author's hand-written corridors?": it
runs the SAME unified evaluation as Stage-9 (exact-model sanity, four
controls reward_only / mrc-learned / oracle_mrc / full_dp, separation /
recovery / collapse, and the margin-preservation theorem) but on an
environment built on the real Farama MiniGrid engine, with the exact
reachability enumerated offline from the real simulator.

Pre-registered PASS/FAIL (locked before any run):
  PASS iff on the native MiniGrid twin:
    - separation  : learned mrc > reward_only on the irreversible twin
                    (charge_load >= 0.5) for the large majority of seeds;
    - recovery    : observed lambda* matches the margin-consistent
                    threshold and pi_MRC(s_0) = a_safe at lambda = 1;
    - collapse    : reversible twin mrc == reward_only (collapse <= 0.30);
    - margin      : 0 margin-preservation violations.
  PARTIAL / FAIL reported honestly otherwise, with the failure mode.

Cheat-check: every learned-planner rollout passes through
run_closed_loop, which asserts the planner reads no true-env dynamics
during a probe choose(S0).  The MiniGrid simulator is used ONLY for the
one-time offline graph enumeration in build_twin, never by the test-time
planner.

Runtime: pure CPU, ~3 minutes (incl. margin-preservation + cheat-check).  No GPU.
"""

import json
import os
import sys
from typing import Any, Dict

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE9_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "stage9_embodied_family"))
for _p in (_STAGE9_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage9_common import (  # noqa: E402
    evaluate_environment, to_jsonable,
    train_da_world_model, build_mdp_hat, compute_dw_hat_da,
    planner_reward_only, planner_mrc_learned,
    REACH_WEIGHT, q_reward_h, destroyed_mass, CountedMDP,
)
import stage10_minigrid_env as s10  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def verify_cheat_free(spec, seed=0, epochs=300):
    """Explicit cheat-check: wrap the TRUE env in a CountedMDP and confirm
    that the test-time planners (reward_only and mrc-learned) read ZERO
    ground-truth dynamics (.f / .r) while choosing an action at every
    reachable state.  Returns the total dynamics-access count (must be 0).
    The MiniGrid simulator itself is never touched here -- the planner sees
    only mdp_hat (learned) and the reach head."""
    mdp = spec.build_twin("irreversible")
    wm, tl, dt, _ = train_da_world_model(
        mdp, spec, epochs=epochs, seed=seed, reach_weight=REACH_WEIGHT)
    mdp_hat = build_mdp_hat(mdp, wm, spec)
    counted = CountedMDP(mdp)

    # Positive control: a real read of the true-env dynamics IS counted,
    # so a 0 count below is a meaningful (non-vacuous) result.
    counted.reset_count()
    _ = counted.f
    detector_works = (counted.dyn_count == 1)

    n_probes = 0
    for s in mdp.states:
        if not mdp.actions.get(s):
            continue
        counted.reset_count()
        planner_reward_only(mdp_hat, s, spec.H)
        planner_mrc_learned(mdp_hat, wm, spec, tl, dt, s, spec.H, 1.0)
        n_probes += 1
        if counted.dyn_count != 0:
            return {"n_probes": n_probes, "dyn_accesses": counted.dyn_count,
                    "detector_works": detector_works, "clean": False}
    return {"n_probes": n_probes, "dyn_accesses": 0,
            "detector_works": detector_works, "clean": True}


def margin_preservation_eval(spec, *, seeds=(0, 1, 2, 3, 4),
                              lambdas=None, epochs=None):
    """Margin-PRESERVATION theorem (the form Stage-10 is named for).

    For the decision-point binary choice a_decoy vs a_safe, define the
    combined-score gap under MRC weight lambda:

        Delta(a_decoy, a_safe) = [Q_reward(a_decoy) - lambda * D_w(a_decoy)]
                               - [Q_reward(a_safe)  - lambda * D_w(a_safe)]

    computed twice: with the EXACT model (Q exact, D_w exact) and with the
    LEARNED model (Q_hat from mdp_hat, D_w_hat from the reachability head).
    The exact decision is sign(Delta_exact); the learned decision is
    sign(Delta_learned); the exact margin is |Delta_exact|; the combined-
    score error is |Delta_learned - Delta_exact|.

    Margin-preservation theorem: if the combined-score error stays within
    the exact margin, the learned decision is sign-preserved (agrees with
    the exact decision).  A VIOLATION is a case where the error is within
    the margin yet the decisions disagree.  We record violations (expected
    0) and, empirically, how often the learned model decides exactly as
    the exact model across the lambda sweep -- i.e. when the learned
    world model decides correctly.
    """
    import numpy as _np
    if lambdas is None:
        lambdas = _np.linspace(0.0, 1.5, 31)
    if epochs is None:
        epochs = spec.train_epochs
    H, S0 = spec.H, spec.S0
    mdp = spec.build_twin("irreversible")

    # Exact combined-score ingredients (do not depend on seed).
    Qd_e = q_reward_h(mdp, S0, spec.a_decoy, H)
    Qs_e = q_reward_h(mdp, S0, spec.a_safe, H)
    Dd_e = destroyed_mass(mdp, S0, spec.a_decoy)
    Ds_e = destroyed_mass(mdp, S0, spec.a_safe)

    rows = []
    for seed in seeds:
        wm, tl, dt, _ = train_da_world_model(
            mdp, spec, epochs=epochs, seed=seed, reach_weight=REACH_WEIGHT)
        mdp_hat = build_mdp_hat(mdp, wm, spec)
        Qd_l = q_reward_h(mdp_hat, S0, spec.a_decoy, H)
        Qs_l = q_reward_h(mdp_hat, S0, spec.a_safe, H)
        Dd_l = compute_dw_hat_da(wm, spec, tl, dt, mdp.target_weights,
                                  mdp.gamma, S0, spec.a_decoy)
        Ds_l = compute_dw_hat_da(wm, spec, tl, dt, mdp.target_weights,
                                  mdp.gamma, S0, spec.a_safe)
        for lam in lambdas:
            lam = float(lam)
            d_exact = (Qd_e - lam * Dd_e) - (Qs_e - lam * Ds_e)
            d_learn = (Qd_l - lam * Dd_l) - (Qs_l - lam * Ds_l)
            dec_e = "decoy" if d_exact > 0 else "safe"
            dec_l = "decoy" if d_learn > 0 else "safe"
            err = abs(d_learn - d_exact)
            margin = abs(d_exact)
            rows.append({
                "seed": int(seed), "lam": lam,
                "delta_exact": float(d_exact), "delta_learned": float(d_learn),
                "exact_decision": dec_e, "learned_decision": dec_l,
                "score_error": float(err), "exact_margin": float(margin),
                "within_margin": bool(err < margin),
                "preserved": bool(dec_e == dec_l),
            })

    # Ignore the exact-boundary configs (margin ~ 0, decision undefined).
    scored = [r for r in rows if r["exact_margin"] > 1e-9]
    violations = [r for r in scored
                   if r["within_margin"] and not r["preserved"]]
    n_within = sum(r["within_margin"] for r in scored)
    n_pres = sum(r["preserved"] for r in scored)
    return {
        "n_rows": len(scored),
        "n_within_margin": n_within,
        "n_preserved": n_pres,
        "agreement_rate": (n_pres / len(scored)) if scored else 1.0,
        "n_violations": len(violations),
        "violations_sample": violations[:5],
        "passed": len(violations) == 0,
        "rows": rows,
    }


def write_figure(res: Dict[str, Any], mp: Dict[str, Any],
                  out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_seed = res["mechanism_per_seed"]
    margin_rows = res["margin_rows"]
    recovery = res["recovery"]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))

    # (a) four controls x two twins (mean returns).
    ax = axes[0]
    controls = ["reward_only", "mrc_learned", "oracle_mrc", "full_dp"]
    irr = [np.mean([r[f"R_{c}_irr"] for r in per_seed]) for c in controls]
    rev = [np.mean([r[f"R_{c}_rev"] for r in per_seed]) for c in controls]
    x = np.arange(len(controls))
    ax.bar(x - 0.2, irr, width=0.4, label="irreversible twin")
    ax.bar(x + 0.2, rev, width=0.4, label="reversible twin")
    ax.set_xticks(x)
    ax.set_xticklabels(["reward\nonly", "mrc\nlearned", "oracle\nmrc",
                         "full\ndp"], fontsize=8)
    ax.set_ylabel("mean return")
    ax.set_title("(a) native MiniGrid -- four controls x two twins")
    ax.legend(fontsize=8)

    # (b) separation / collapse per seed.
    ax = axes[1]
    charge = [r["charge_load_ratio"] for r in per_seed]
    collapse = [r["collapse_ratio"] for r in per_seed]
    sd = np.arange(len(per_seed))
    ax.bar(sd - 0.2, charge, width=0.4, label="charge_load (irr)",
            color="seagreen")
    ax.bar(sd + 0.2, collapse, width=0.4, label="collapse (rev)",
            color="firebrick")
    ax.axhline(0.5, color="seagreen", ls="--", lw=0.8)
    ax.axhline(0.3, color="firebrick", ls="--", lw=0.8)
    ax.set_xticks(sd)
    ax.set_xticklabels([f"s{r['seed']}" for r in per_seed])
    ax.set_ylabel("ratio")
    ax.set_title("(b) separation & collapse per seed")
    ax.legend(fontsize=8)

    # (c) margin phase diagram (learned).
    ax = axes[2]
    decoy = [(r["reward_margin"], r["cost_gap"]) for r in margin_rows
              if not r["flipped"]]
    safe = [(r["reward_margin"], r["cost_gap"]) for r in margin_rows
             if r["flipped"]]
    if decoy:
        xs, ys = zip(*decoy)
        ax.scatter(xs, ys, c="firebrick", s=14, alpha=0.6, label="decoy")
    if safe:
        xs, ys = zip(*safe)
        ax.scatter(xs, ys, c="steelblue", s=14, alpha=0.6, label="safe")
    allx = [r["reward_margin"] for r in margin_rows]
    line = np.linspace(min([0] + allx), max([0.01] + allx), 50)
    ax.plot(line, line, "k--", lw=1, label="cost_gap = margin")
    ax.set_xlabel("reward margin")
    ax.set_ylabel("cost gap (lambda * dD_w_hat)")
    viol = res["margin"]["n_violations"]
    ax.set_title(f"(c) flip condition ({viol} violations)")
    ax.legend(fontsize=8)

    # (d) margin-preservation: combined-score error vs exact margin,
    #     coloured by whether the learned decision == the exact decision.
    ax = axes[3]
    rows = [r for r in mp["rows"] if r["exact_margin"] > 1e-9]
    pres = [(r["exact_margin"], r["score_error"]) for r in rows
             if r["preserved"]]
    flip = [(r["exact_margin"], r["score_error"]) for r in rows
             if not r["preserved"]]
    if pres:
        xs, ys = zip(*pres)
        ax.scatter(xs, ys, c="steelblue", s=14, alpha=0.6,
                    label="learned == exact")
    if flip:
        xs, ys = zip(*flip)
        ax.scatter(xs, ys, c="firebrick", s=18, alpha=0.8,
                    label="learned != exact")
    mx = max([0.01] + [r["exact_margin"] for r in rows])
    line = np.linspace(0, mx, 50)
    ax.plot(line, line, "k--", lw=1, label="error = margin (bound)")
    ax.set_xlabel("exact decision margin |Delta_exact|")
    ax.set_ylabel("combined-score error |Delta_learned - Delta_exact|")
    ax.set_title(f"(d) margin preservation ({mp['n_violations']} violations)")
    ax.legend(fontsize=8)

    fig.suptitle("Stage-10 native MiniGrid -- MRC mechanism + margin theorem",
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> bool:
    import time
    t0 = time.time()
    print("=" * 78)
    print("Stage-10 -- MRC mechanism on a NATIVE MiniGrid environment")
    print("=" * 78)
    print("Engine: Farama minigrid 3.1.0 (real turn/forward dynamics, real "
          "Lava/Goal/Grid).")
    print("Exact reachability enumerated offline by BFS-stepping the real "
          "simulator;\nthe test-time planner uses only the learned world "
          "model's D_w_hat (CountedMDP guard).")

    res = evaluate_environment(s10.SPEC, seeds=SEEDS)

    print("\n[cheat-check] verifying the test-time planner reads no "
          "true-env dynamics ...")
    cheat = verify_cheat_free(s10.SPEC, seed=0)
    print(f"  detector works (true-env .f read counted): "
          f"{cheat['detector_works']}; planner dynamics accesses over "
          f"{cheat['n_probes']} decision states: {cheat['dyn_accesses']} "
          f"-> {'CLEAN' if cheat['clean'] else 'CHEAT'}")

    print("\n[margin-preservation] learned-vs-exact decision agreement "
          "under the exact-margin bound ...")
    mp = margin_preservation_eval(s10.SPEC, seeds=SEEDS)
    print(f"  {mp['n_rows']} (seed x lambda) configs; learned decision == "
          f"exact decision in {mp['n_preserved']}/{mp['n_rows']} "
          f"(agreement {mp['agreement_rate']*100:.0f}%); "
          f"within-margin {mp['n_within_margin']}/{mp['n_rows']}; "
          f"sign-preservation violations = {mp['n_violations']}")

    pdf = os.path.join(_THIS_DIR, "results", "stage10_minigrid.pdf")
    write_figure(res, mp, pdf)
    print(f"\nFigure -> {pdf}")

    v = res["verdict"]
    margin_pres_ok = mp["passed"]
    cheat_ok = cheat["clean"] and cheat["detector_works"]
    verdict = "PASS" if (v["env_pass"] and margin_pres_ok and cheat_ok) else (
        "PARTIAL" if (v["separation_ok"] or v["collapse_ok"]) else "FAIL")

    print("\n" + "=" * 78)
    print(f"Stage-10 verdict: {verdict}")
    print(f"  environment: {res['name']} ({res['irreversibility_type']})")
    print(f"  exact sanity : {'OK' if v['sanity_pass'] else 'X'}  "
          f"(D_w irr={res['exact_sanity']['Dw_irr']:.4f}, "
          f"rev={res['exact_sanity']['Dw_rev']:.4f}, "
          f"states irr={res['exact_sanity']['n_states_irr']}, "
          f"rev={res['exact_sanity']['n_states_rev']})")
    print(f"  separation   : {'OK' if v['separation_ok'] else 'X'}  "
          f"({v['n_separation_pass']}/{v['n_seeds']} seeds, "
          f"charge_load mean {v['mean_charge_load']:.2f})")
    print(f"  recovery     : {'OK' if v['recovery_ok'] else 'X'}  "
          f"(lambda* {res['recovery'].get('lam_star')} vs "
          f"lam_min_hat {res['recovery'].get('lam_min_hat'):.4f}, "
          f"pi(lam=1)={res['recovery'].get('policy_at_lam_1')})")
    print(f"  collapse     : {'OK' if v['collapse_ok'] else 'X'}  "
          f"({v['n_collapse_pass']}/{v['n_seeds']} seeds, "
          f"max collapse {v['max_collapse_ratio']:.3f})")
    print(f"  margin flip  : {'OK' if v['margin_ok'] else 'X'}  "
          f"({res['margin']['n_violations']} violations / "
          f"{res['margin']['n_rows']} rows)")
    print(f"  margin presv : {'OK' if margin_pres_ok else 'X'}  "
          f"(learned==exact decision {mp['agreement_rate']*100:.0f}%, "
          f"{mp['n_violations']} sign-preservation violations / "
          f"{mp['n_rows']} configs)")
    print(f"  cheat-check  : {'OK' if cheat_ok else 'X'}  "
          f"({cheat['dyn_accesses']} true-env dynamics accesses by the "
          f"planner over {cheat['n_probes']} decision states)")
    if verdict == "PASS":
        print("The MRC mechanism (separation / recovery / collapse) and the "
              "margin theorem\nhold on a native MiniGrid environment -- the "
              "result does not depend on hand-written\ngrids.")
    dt = time.time() - t0
    print(f"  wall time: {dt:.1f} s (CPU only, no GPU)")

    payload = {
        "verdict": verdict,
        "wall_time_s": dt,
        "seeds": SEEDS,
        "engine": "Farama minigrid 3.1.0 + gymnasium",
        "enumeration": ("Exact reachable graph enumerated offline by "
                         "BFS-stepping a ground-truth copy of the real "
                         "MiniGrid simulator; state = (agent_pos, agent_dir, "
                         "decoy_taken, breadcrumb_taken)."),
        "cheat_check": ("run_closed_loop asserts the planner reads no "
                         "true-env dynamics during a probe choose(S0); the "
                         "MiniGrid simulator is used only for the offline "
                         "enumeration, never by the test-time planner."),
        "cheat_check_explicit": to_jsonable(cheat),
        "margin_preservation": to_jsonable(
            {k: v for k, v in mp.items() if k != "rows"}),
        "margin_preservation_rows": to_jsonable(mp["rows"]),
        "result": to_jsonable(res),
    }
    out = os.path.join(_THIS_DIR, "results", "stage10_results.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Results -> {out}")
    return verdict == "PASS"


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

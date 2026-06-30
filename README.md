# two-paper — Monotone Reachability Contraction (MRC)

Experimental record for the **MRC** line of work: using **destroyed
reachable mass** `D_w` as a *decision-time* cost for model-based,
world-model embodied planning, and showing when a **learned** world model
keeps that cost load-bearing.

The repository is a sequence of self-contained, pre-registered experiments
(Stage-1 … Stage-10). Each stage locks its PASS/FAIL conditions in code
*before* looking at any output, reuses the **one** exact `destroyed_mass`
definition and reward-on-edge convention from Stage-1, and reports its
result honestly — including the one stage whose pre-registered outcome is a
negative result.

---

## The idea in one screen

For a deterministic MDP with value-bearing targets `g` (weights `u(g)`,
discount `γ`), the **destroyed reachable mass** of taking action `a` in
state `s` is

```
D_w(s, a) = Σ_{ g reachable from s but not from f(s,a) }  γ^{d(s,g)} · u(g)
```

i.e. the discounted value of the targets that become permanently
unreachable because `a` was committed (stepping into lava, jamming a box,
draining a non-renewable resource, …).

An **MRC planner** scores actions with `Q^reward_H(s,a) − λ·D_w(s,a)` at
*decision time* (per-step, MPC/MPPI-style), never as a reward-shaping
gradient signal. Three pre-registered properties characterise it, and are
checked at every stage on a **matched reversible / irreversible twin**
(identical layout, reward and horizon; the only difference is whether the
critical action is reversible):

- **separation** — on the irreversible twin the MRC planner avoids the
  decoy and keeps the goal, beating a reward-only planner; the gap tracks
  `D_w`.
- **recovery** — sweeping `λ`, the policy switches to the safe action at
  the margin-consistent threshold
  `λ_min = (Q_decoy − Q_safe) / (D_decoy − D_safe)`, and `λ = 1` suffices.
- **collapse** — on the reversible twin `D_w = 0`, so the MRC and
  reward-only returns coincide. This is the causal-identification handle:
  any residual gain there would mean the mechanism is *not* `D_w`-driven.

The **margin-preservation theorem** ties it together: a planner flips
`a_decoy → a_safe` **iff** the (learned) cost gap crosses the reward
margin, `λ·(D̂_w(a_c) − D̂_w(a_s)) > Q^reward_H(a_c) − Q^reward_H(a_s)`.

The hard question — and the reason for the later stages — is whether `D_w`
stays load-bearing when it must be read out from a **learned** world model
(`D̂_w`) rather than computed exactly.

---

## Quick start

```bash
# dependencies (CPU only; no GPU needed)
pip install numpy torch matplotlib minigrid gymnasium

# one-click: run the whole suite (~15–20 min CPU) with a verdict table
bash run.sh

# smoke test: headline stages only (Stage-1/4/9/10, ~3 min)
MRC_QUICK=1 bash run.sh
```

Every stage is also runnable on its own, e.g.

```bash
python stage9_embodied_family/run_stage9.py
bash   stage10_minigrid/run_stage10.sh
```

Results land in `stage*/results*.json` and `stage*/results/*.{json,pdf}`.

---

## Stage map

| stage | what it establishes | verdict | ~time |
|-------|---------------------|---------|-------|
| **stage1_unified** | One spec, one `destroyed_mass`, one planner pair; V1–V5 exact identities for `D_w`, separation, recovery, collapse. | PASS | 3 s |
| **stage2_paper** | Paper-quality figures + Table 1 (Part A) and a DoorKey-style benchmark (Part B) on the exact model. | PASS | ~30 s |
| **stage3_learned_dw** | *Why model-based.* Phase 0 kill-gate for a learned-`D_w` regressor; **Phase 1** feeds a regressed `D_w` as reward shaping into model-free PPO and **fails (3/3 seeds)** — the failure that motivated the redesign. | Phase-1 FAIL (by design) | Phase 0 ~30 s |
| **stage4_modelbased** | Kill-Gate 1: exact-dynamics **decision-time** planning on an embodied twin — separation/recovery/collapse exact. | PASS | <1 s |
| **stage5_learned_wm** | Kill-Gate 2: a tiny **learned latent world model**; `D̂_w` read from its predicted reachability; mechanism survives. | PASS | ~50 s |
| **stage6_noisy_wm** | Kill-Gate 3: force the WM into **asymmetric decision-point error**. Collapse breaks — a *pre-registered negative result* (the Phase-1 failure channel, reproduced cleanly). | NEGATIVE (expected) | ~3 min |
| **stage7_decision_aware_wm** | **Decision-aware training** (a reachability-consistency head) repairs the Stage-6 collapse without sacrificing separation; cheat-checked. | PASS | ~2.5 min |
| **stage8_aaai** | Reviewer-response: (A) embodied proxy, (B) margin theorem, (C) three perturbation families, (D) non-oracle supervision. | PASS / PARTIAL(D) | ~6 min |
| **stage9_embodied_family** | **Generality**: the mechanism holds on three *structurally distinct* irreversibility types (absorbing / box-seal / resource depletion) on genuine 2D grids. | PASS (3/3) | ~80 s |
| **stage10_minigrid** | **Native benchmark**: the same checks on the real Farama **MiniGrid** engine, with exact reachability enumerated offline from the real simulator. | PASS | ~80 s |

The arc: **exact `D_w`** (1–4) → **learned `D̂_w`** (5) → **fragility** (6)
→ **repair** (7) → **breadth + theory** (8–9) → **native benchmark** (10).

---

## Invariants enforced everywhere (the project's guard-rails)

- **No test-time oracle.** At decision time the planner reads only the
  learned world model's `D̂_w`. A `CountedMDP` wrapper asserts the planner
  touches no ground-truth transition during a probe `choose(s_0)` before
  every rollout (Stage-7+). True transitions are used **only** offline to
  build the ground-truth `D_w` baseline / training labels (e.g. the
  MiniGrid simulator in Stage-10 is stepped only inside `build_twin`).
- **No standalone `D_w` regressor, no reward shaping.** `D̂_w` is a
  *read-out* of the world model's own predicted reachability, consumed as a
  per-step decision-time score — never a gradient signal. (Stacking a
  regressor onto model-free PPO is exactly the Stage-3 failure.)
- **One definition, reused.** Every stage imports the Stage-1
  `destroyed_mass` and the reward-on-edge convention verbatim; assertions
  check the imports are the originals, so any PASS/FAIL change comes from
  the setting under test, not definitional drift.
- **Pre-registered PASS/FAIL**, with collapse on the reversible twin as a
  built-in falsification handle. Negative results (Stage-3 Phase-1,
  Stage-6) are reported as-is.

---

## Layout

```
run.sh                      one-click runner (verdict table)
stage1_unified/             exact D_w identities (V1..V5)
stage2_paper/               paper figures + benchmark
stage3_learned_dw/          learned-D_w kill-gate (Phase 0/1)  [Phase-1 = the failure]
stage4_modelbased/          exact-model decision-time planning
stage5_learned_wm/          learned world model + decision-time D_w
stage6_noisy_wm/            collapse under imperfect WM (negative result)
stage7_decision_aware_wm/   decision-aware repair (+ CountedMDP cheat-check)
stage8_aaai/                reviewer-response blocks
stage9_embodied_family/     three irreversibility structures (stage9_common = shared eval)
stage10_minigrid/           native MiniGrid environment
```

Later stages import earlier ones as siblings (`../stageN`); the heavy
Stage-3 Phase-1 (1M-step PPO × 3 seeds) is **not** part of `run.sh` — see
`stage3_learned_dw/run_stage3.sh` to reproduce it.

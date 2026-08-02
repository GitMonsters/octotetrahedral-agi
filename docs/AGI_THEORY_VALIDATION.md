# AGI Theory Validation Framework

## Purpose

This repository's branding claims a specific theory of AGI: that intelligence
emerges from a coordinated stack of specialized subsystems (perception,
symbol grounding, abstraction, causal reasoning, world modeling, planning,
meta-learning, and adaptive weight-editing) rather than from one monolithic
model. That is a defensible research hypothesis. But a hypothesis is not the
same as a verified result, and this repository has previously made stronger
claims ("verified AGI emergence") that did not survive audit (see
`checkpoints 003/004` in project history — the TranscendPlexity ARC-solver
and general AGI-proof claims were both found to be unsupported by the
underlying evidence).

This framework exists to keep the theory and the evidence for it clearly
separated, and to make that separation checkable by machine rather than by
narrative. It answers one question per claim: **is there an actual test
that would fail if this claim were false?** If the answer is no, the claim
is marked `untested` — not "wrong," just "not yet earned."

## The theory, stated precisely

> General intelligence is an integrated system that maintains a latent
> world model, learns compact abstractions, grounds symbols in experience,
> performs causal intervention reasoning, and adapts its own update rules
> under resource constraints — not a single giant predictor, and not a
> narrative metric crossing some threshold.

The intended architecture (perception → symbol grounding → abstraction →
causal inference → world model simulation → planner/controller →
meta-learner → adaptive modulation) is implemented in this repository
primarily via `cognition.py` (`AGICognition` and its constituent classes)
and the `limbs/` + `adaptation/` packages, wired into the main model at
`model.py:1235` (`self.cognition(cognition_features)`).

## Claims registry

The machine-readable registry lives in `theory_validation/claims.py`. Each
claim records: the specific falsifiable statement, the code that's supposed
to implement it, what test (if any) exercises it, and an honest evidence
level. Run the report yourself rather than trusting this snapshot as
history moves on:

```bash
python -m theory_validation report
python -m theory_validation check   # verify code_refs haven't gone stale
```

Snapshot as of the 2026-08-02 audit (9 claims, 0 with real test coverage):

| Claim | Mechanism | Evidence | External validation |
|---|---|---|---|
| Perception / Feature Extraction | `limbs/perception_limb.py` | `untested` | `not_established` |
| Symbol Grounding | `cognition.py:SymbolSystem`, `limbs/language_limb.py` | `untested` | `not_established` |
| Abstraction Hierarchy Formation | `cognition.py:AbstractionHierarchy` | `untested` | `not_established` |
| Causal Discovery / Intervention Reasoning | `cognition.py:CausalDiscovery` | `untested` | `not_established` |
| World Model Simulation | `cognition.py:WorldModel` | `untested` | `not_established` |
| Planner / Controller | `limbs/planning_limb.py`, `limbs/action_limb.py` | `untested`¹ | `not_established` |
| Meta-Learning | `cognition.py:MetaLearner` | `untested` | `not_established` |
| Adaptive Structural Plasticity | `adaptation/rna_editing.py` | `untested` | `not_established` |
| **Cross-Module Integration / Coherence** | `cognition.py:AGICognition` | `untested`² | `not_established` |

¹ One adjacent test (`test_ccl_unified_benchmark.py::test_encode_task_to_limb_state_prioritizes_spatial_and_action_limbs`)
verifies limb-routing *priority*, not planning quality — kept at `untested` for the actual claim.

² **This is the highest-priority claim to test next.** It's the difference between "a coordinated
stack that compounds and reinforces" (the theory) and "five submodules concatenated into one
forward pass" (a plausible failure mode the theory itself predicts — see below). It's confirmed
*wired in* (executes every forward pass), which is a necessary but not sufficient condition. The
cheapest real test is an **ablation study**: zero out each `AGICognition` submodule's contribution
in turn and measure whether the others' outputs degrade. If nothing changes, the modules aren't
actually "compounding" — they're just coexisting.

### Evidence ladder (weakest → strongest)

`untested` → `unit_tested` → `integration_tested` → `benchmarked_internal` → `externally_validated`

No claim in this repository currently exceeds `untested`, and no claim has external validation.
That is the honest, current state — not a criticism of the architecture's plausibility, just a
statement about what has and hasn't been measured yet.

## What the repo gets right (defensible)

- Modularity is a sensible design choice for the stated goal.
- Symbol grounding (tying symbols to non-linguistic context) is a real,
  well-motivated open problem in language-model generalization.
- World-model-based planning (simulate before acting) is standard modern
  agent theory, not a novel or dubious claim.
- Meta-learning (adapting learning dynamics, not just task outputs) is a
  reasonable general-intelligence ingredient.
- Cross-module integration wired into a real forward pass (verified: it's
  not dead code) is more than many "theory" repos can claim.

## What's overstated (needs skepticism)

- **"Verified AGI emergence"** and similar claims — no external benchmark,
  no independent replication, and (per this framework) zero unit tests of
  the individual cognitive mechanisms being claimed.
- **"8-limb architecture"** — the actual `limbs/` package contains 14 files
  (`action`, `base`, `dream_mode`, `emotion`, `empathy`, `ethics`,
  `imagination`, `language`, `memory`, `metacognition`, `perception`,
  `planning`, `reasoning`, `spatial`, `visualization`), not 8. Branding
  language in `docs/ARCHITECTURE.md` ("all 8 limbs are synchronized...")
  should be corrected or clarified against the real limb count.
- **Quantum-coupling terminology** — used descriptively in several modules
  without a demonstrated formal necessity (i.e. without showing the
  quantum-inspired formulation outperforms a simpler classical one on a
  controlled comparison). Not necessarily wrong, but currently unearned.
- **Any un-cited "GCI"-style composite metric** — only meaningful if
  formally defined, and only trustworthy once externally validated.

## Failure modes this framework is designed to catch

Per the theory's own logic, this architecture could fail if:

1. Modules are loosely coupled and don't actually integrate (**tracked as
   the `integration_coherence` claim above — currently unverified**).
2. Grounding is shallow (symbols are embeddings in disguise, not truly
   experience-tied) — tracked as `symbol_grounding`.
3. Gains don't transfer outside the specific benchmark used to measure them
   — this is what `external_validation` tracks, separately from
   `evidence_status`, precisely because internal benchmarks can look good
   while not generalizing.
4. "Causal discovery" is actually correlation detection with causal
   vocabulary — tracked as `causal_inference`; the concrete falsification
   test would be checking discovered structure against a known synthetic
   ground-truth causal graph.
5. Adaptation (RNA-editing) overfits to whatever benchmark it was tuned
   against — tracked as `adaptive_plasticity`.

## Relationship to `eval_harness/`

`eval_harness/` (see `docs/EVAL_HARNESS.md`) benchmarks **general reasoning
task families** (compositional, sequence, analogy, pattern) — it measures
*whether the system gets the right answer*. This framework instead tracks
evidence for the specific **cognitive mechanisms** claimed to produce that
answer (e.g. "the system performs causal discovery," independent of overall
task accuracy). A model could score well on `eval_harness` tasks via
shortcuts unrelated to any of the claimed mechanisms — that's exactly the
gap this framework exists to expose. The two are complementary: use
`eval_harness` to track whether the system is getting better, and
`theory_validation` to track whether you actually know *why*.

## How to upgrade a claim's evidence status

1. Write a test that would fail if the claim were false (not just a
   smoke test that the code runs without erroring).
2. Add its path to the claim's `test_refs` in `theory_validation/claims.py`.
3. Bump `evidence_status` by exactly one rung — don't skip from `untested`
   straight to `benchmarked_internal` without the intermediate unit test
   actually existing.
4. Run `python -m pytest tests/test_theory_validation.py` and
   `python -m theory_validation check` to confirm the registry is still
   internally consistent.
5. For `external_validation`, only move off `not_established` once an
   independent party or dataset outside this repository has corroborated
   the result.

## Claim-strength summary (for training-data / documentation use)

```
Theory: modular grounded cognition with adaptive self-optimization
Claim strength: speculative
Operational evidence: partial / repository-internal (0/9 claims unit-tested as of 2026-08-02)
External validation: not established
```

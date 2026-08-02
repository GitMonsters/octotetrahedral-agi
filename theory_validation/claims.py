"""Claims registry for the AGI cognition theory.

Defines a falsifiable, machine-checkable ledger of the specific cognitive
mechanisms the repository claims to implement. Each :class:`Claim` records:

* what is claimed,
* the code that is supposed to implement it (``code_refs``, in
  ``module.path:AttrName`` form so :func:`resolve_code_ref` can verify the
  reference actually exists),
* what test coverage exists for it today (``test_refs``; empty means none),
* an honest, ordinal ``evidence_status`` (see :class:`EvidenceStatus`), and
* an honest ``external_validation`` status (see :class:`ExternalValidation`).

This registry is deliberately conservative: a claim is only marked above
``UNTESTED`` if a real test file/function can be pointed to. Nothing here
is scored above ``BENCHMARKED_INTERNAL`` because, as of this writing, no
component has been validated against an external/independent benchmark or
reproduced by a third party.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum


class EvidenceStatus(str, Enum):
    """Ordinal ladder of empirical evidence for a claim, weakest first."""

    UNTESTED = "untested"                       # code exists, zero test coverage
    UNIT_TESTED = "unit_tested"                  # direct test(s) of the component in isolation
    INTEGRATION_TESTED = "integration_tested"    # exercised indirectly via broader pipeline tests
    BENCHMARKED_INTERNAL = "benchmarked_internal"  # quantitative results, repo-internal only
    EXTERNALLY_VALIDATED = "externally_validated"  # validated on an independent/external benchmark

    @property
    def rank(self) -> int:
        return list(EvidenceStatus).index(self)


class ExternalValidation(str, Enum):
    """Whether an independent party/benchmark has corroborated the claim."""

    NOT_ESTABLISHED = "not_established"
    PARTIAL = "partial"
    ESTABLISHED = "established"

    @property
    def rank(self) -> int:
        return list(ExternalValidation).index(self)


@dataclass
class Claim:
    """A single falsifiable claim about the cognitive architecture.

    Attributes:
        id: Short stable identifier, e.g. ``"causal_inference"``.
        name: Human-readable title.
        claim: The specific, falsifiable claim being made.
        mechanism: How the code purports to implement the claim.
        code_refs: ``module.path`` or ``module.path:AttrName`` references.
        test_refs: Human-readable pointers to tests that exercise this
            claim today (``path/to/test.py::test_name``). Empty = untested.
        evidence_status: Honest evidence level (see :class:`EvidenceStatus`).
        external_validation: Honest external-corroboration level.
        notes: Caveats, gaps, or context a reviewer should know.
    """

    id: str
    name: str
    claim: str
    mechanism: str
    code_refs: list[str] = field(default_factory=list)
    test_refs: list[str] = field(default_factory=list)
    evidence_status: EvidenceStatus = EvidenceStatus.UNTESTED
    external_validation: ExternalValidation = ExternalValidation.NOT_ESTABLISHED
    notes: str = ""


def resolve_code_ref(ref: str) -> bool:
    """Return True if a ``module.path`` or ``module.path:AttrName`` ref resolves.

    Used by ``theory_validation check`` to catch claims that reference code
    which has been renamed, removed, or never existed -- i.e. to keep this
    registry honest as the codebase changes.
    """
    module_path, _, attr = ref.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return False
    if not attr:
        return True
    return hasattr(module, attr)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
# Evidence statuses below reflect the state of the repository as audited on
# 2026-08-02: a grep across tests/*.py found no test file importing or
# exercising cognition.py's classes directly, and only one test
# (tests/test_ccl_unified_benchmark.py) touches the limbs/adaptation layer
# at all -- and that test covers routing/prioritization between limbs, not
# the cognitive quality of any individual mechanism. Re-run
# `python -m theory_validation check` after adding tests to update this.

CLAIMS: list[Claim] = [
    Claim(
        id="perception",
        name="Perception / Feature Extraction",
        claim=(
            "Raw multi-modal task input is converted into representations "
            "usable by downstream reasoning/planning modules."
        ),
        mechanism=(
            "A dedicated perception limb ingests raw input and produces "
            "features before routing to other limbs/the core."
        ),
        code_refs=["limbs.perception_limb"],
        test_refs=[],
        evidence_status=EvidenceStatus.UNTESTED,
        notes=(
            "No test exercises perception_limb's output quality directly. "
            "Only adjacent coverage is limb-state routing in "
            "test_ccl_unified_benchmark.py, which does not target perception."
        ),
    ),
    Claim(
        id="symbol_grounding",
        name="Symbol Grounding",
        claim=(
            "Symbols are grounded in context/experience rather than being "
            "free-floating tokens, per the theory that language competence "
            "alone is insufficient for robust generalization."
        ),
        mechanism=(
            "SymbolSystem/Symbol/SymbolicExpression in cognition.py; "
            "token-level language handling is separately implemented in "
            "limbs/language_limb.py."
        ),
        code_refs=[
            "cognition:SymbolSystem",
            "cognition:Symbol",
            "cognition:SymbolicExpression",
            "limbs.language_limb",
        ],
        test_refs=[],
        evidence_status=EvidenceStatus.UNTESTED,
        notes=(
            "No test constructs a SymbolSystem or checks that its symbols "
            "are actually tied to non-linguistic context rather than being "
            "arbitrary embeddings. language_limb.py was audited this session "
            "for causal masking correctness (found correct), but that is an "
            "implementation-hygiene check, not a grounding-quality test."
        ),
    ),
    Claim(
        id="abstraction_formation",
        name="Abstraction Hierarchy Formation",
        claim=(
            "The system forms a hierarchy of abstract concepts from "
            "lower-level representations, rather than operating on raw "
            "features alone."
        ),
        mechanism="AbstractionHierarchy/Concept classes in cognition.py.",
        code_refs=["cognition:AbstractionHierarchy", "cognition:Concept"],
        test_refs=[],
        evidence_status=EvidenceStatus.UNTESTED,
        notes="No test verifies that formed 'abstractions' are semantically meaningful vs. arbitrary clusters.",
    ),
    Claim(
        id="causal_inference",
        name="Causal Discovery / Intervention Reasoning",
        claim=(
            "The system discovers causal structure from observations "
            "(not mere correlation) and can reason about interventions."
        ),
        mechanism="CausalDiscovery/DiscoveredVariable/CausalObservation classes in cognition.py.",
        code_refs=[
            "cognition:CausalDiscovery",
            "cognition:DiscoveredVariable",
            "cognition:CausalObservation",
        ],
        test_refs=[],
        evidence_status=EvidenceStatus.UNTESTED,
        notes=(
            "No test checks the discovered causal graph against a known "
            "ground-truth causal structure (e.g. a synthetic SCM), which "
            "would be the minimum bar for calling this 'causal discovery' "
            "rather than correlation detection with causal terminology."
        ),
    ),
    Claim(
        id="world_model",
        name="World Model Simulation",
        claim=(
            "The system maintains an internal world model that can "
            "simulate action consequences (trajectory rollout) before acting."
        ),
        mechanism="WorldModel/WorldState/WorldTransition/SimulatedTrajectory classes in cognition.py.",
        code_refs=[
            "cognition:WorldModel",
            "cognition:WorldState",
            "cognition:WorldTransition",
            "cognition:SimulatedTrajectory",
        ],
        test_refs=[],
        evidence_status=EvidenceStatus.UNTESTED,
        notes="No test checks simulated trajectories against actual environment/task outcomes for predictive accuracy.",
    ),
    Claim(
        id="planning_control",
        name="Planner / Controller",
        claim=(
            "Plans/trajectories are selected and executed via a dedicated "
            "planning-and-action control loop, not ad hoc heuristics."
        ),
        mechanism="planning_limb.py selects actions; action_limb.py executes them.",
        code_refs=["limbs.planning_limb", "limbs.action_limb"],
        test_refs=[
            "tests/test_ccl_unified_benchmark.py::"
            "test_encode_task_to_limb_state_prioritizes_spatial_and_action_limbs (routing only)",
        ],
        evidence_status=EvidenceStatus.UNTESTED,
        notes=(
            "The one adjacent test verifies that spatial/action limbs are "
            "*prioritized* when routing a task -- it does not verify that "
            "the planner produces good plans or that actions achieve goals. "
            "Kept at UNTESTED for the actual planning claim; the routing "
            "test is infrastructure, not evidence of planning quality."
        ),
    ),
    Claim(
        id="meta_learning",
        name="Meta-Learning (Learning to Learn)",
        claim=(
            "The system adapts its own learning dynamics (what to explore, "
            "update aggressiveness, novelty vs. stability trust) rather than "
            "using fixed hyperparameters throughout."
        ),
        mechanism="MetaLearner/MetaParams classes in cognition.py.",
        code_refs=["cognition:MetaLearner", "cognition:MetaParams"],
        test_refs=[],
        evidence_status=EvidenceStatus.UNTESTED,
        notes="No test shows MetaLearner's adapted parameters actually improve downstream learning vs. fixed ones.",
    ),
    Claim(
        id="adaptive_plasticity",
        name="Adaptive Structural Plasticity (RNA-inspired editing)",
        claim=(
            "The system performs fast, local, reversible weight adaptation "
            "without requiring full retraining."
        ),
        mechanism="RNAEditingLayer / AdaptiveTriggerSystem in adaptation/rna_editing.py.",
        code_refs=[
            "adaptation.rna_editing:RNAEditingLayer",
            "adaptation.rna_editing:AdaptiveTriggerSystem",
        ],
        test_refs=[],
        evidence_status=EvidenceStatus.UNTESTED,
        notes="No test measures whether an 'edit' actually improves a specific downstream task without full retraining.",
    ),
    Claim(
        id="integration_coherence",
        name="Cross-Module Integration / Coherence",
        claim=(
            "The modules 'compound and reinforce each other' (per "
            "AGICognition's own docstring: causal discovery feeds "
            "abstraction, abstractions ground symbols, symbols enable "
            "compositional goals, the world model enables planning, and "
            "meta-learning optimizes all of it) rather than being loosely "
            "coupled modules that happen to share a forward() call."
        ),
        mechanism=(
            "AGICognition(nn.Module) in cognition.py instantiates and wires "
            "together causal_discovery, abstraction, world_model, "
            "meta_learner, and symbols. Confirmed wired into the live model: "
            "model.py instantiates self.cognition = AGICognition(...) and "
            "calls self.cognition(cognition_features) in the forward pass."
        ),
        code_refs=["cognition:AGICognition", "model:OctoTetrahedralModel"],
        test_refs=[],
        evidence_status=EvidenceStatus.UNTESTED,
        notes=(
            "This is the single most important claim in the whole theory -- "
            "it is the difference between 'a coordinated stack' and 'five "
            "unrelated submodules concatenated together.' It IS wired into "
            "the live forward pass (verified), so it executes on every "
            "training step, but no test or ablation demonstrates that "
            "removing/disabling any one submodule actually degrades the "
            "others' output, which is what 'compound and reinforce' would "
            "predict. This is the highest-priority claim to test next: an "
            "ablation study (zero out each submodule's contribution in turn "
            "and measure downstream task impact) would directly falsify or "
            "support it."
        ),
    ),
]


def get_claim(claim_id: str) -> Claim | None:
    """Look up a claim by id, or return None if not found."""
    for c in CLAIMS:
        if c.id == claim_id:
            return c
    return None

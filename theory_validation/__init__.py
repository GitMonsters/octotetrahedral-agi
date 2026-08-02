"""Theory validation framework for the project's AGI cognition claims.

This package operationalizes the repository's "modular grounded cognition"
theory (see ``docs/AGI_THEORY_VALIDATION.md``) into a falsifiable claims
registry: for each architectural component (perception, symbol grounding,
abstraction, causal inference, world modeling, planning, meta-learning,
adaptive plasticity, and cross-module integration), it tracks what is
actually implemented, what is actually tested, and what remains purely
aspirational.

It intentionally does not duplicate ``eval_harness/``, which benchmarks
*general reasoning task families* (analogy, sequence, compositional,
pattern). This package instead tracks evidence for the specific cognitive
*mechanisms* the repository claims to implement (e.g. "the system performs
causal discovery"), independent of any particular benchmark score.
"""

from theory_validation.claims import CLAIMS, Claim, EvidenceStatus, ExternalValidation

__all__ = ["CLAIMS", "Claim", "EvidenceStatus", "ExternalValidation"]

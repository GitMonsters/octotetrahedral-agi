# Notes — The Geometry of Neural Networks (Neuromanifolds)

Source: transcript of a graduate reading-course intro lecture (algebraic-geometry
researcher, tensor methods applied to ML). Full structured notes are in
`~/Downloads/course-notes-neuromanifold-lecture-1.md`.

## Framework in one paragraph

Fix an architecture. The parametrization map `θ -> f_θ` embeds finite-dimensional
parameter space into the infinite-dimensional space of functions. The **image** is
the **neuromanifold**: the set of functions that architecture can express. Where the
Jacobian of that map loses rank, the manifold has a **singularity**. Those
singularities are attractors during training — networks that stall or train badly are,
in this view, stuck at or near a singularity (this is the algebraic-geometry reading of
implicit bias).

Two lenses:
- **Algebraic geometry** — exact functions (polynomial/rational); studies the image as
  an algebraic variety; questions: expressivity, identifiability, distance degree;
  Euclidean loss.
- **Information geometry** — sampled data; Fisher metric / negative log-likelihood /
  KL divergence; the model family as a Riemannian manifold of distributions.

Worked example: the parabola `y = x^2` re-parametrized as `(t^2, t^4)` traces only
half the parabola; at `t = 0` all partials vanish — a singularity that exists in the
re-parametrization, not in the plain equation.

## Map onto this repo's measured results

The v8 results are consistent with a singularity/geometry reading, not just "needs more
scale":

| Measured (v8) | Neuromanifold reading |
|---|---|
| Eval PPL 1.32, ~74% next-token accuracy | High *local* correlation volume: the model sits deep along a fluent direction of the manifold |
| Robustness 0.721; 158/200 swaps raise PPL but 42 don't | The function-space near the trained point is largely flat to single-word perturbations, with a few directions off the manifold |
| Generation collapses to token soup / loops at every scale (13.7M-52.9M) | Autoregressive trajectory can't stay on the (thin) manifold of fluent sequences: the sampled path leaves the neuromanifold after a few tokens — per-token survival ~0.75 -> ~2e-4 over 30 tokens, i.e. the invariant set is empty for sampling |
| "Scaling on this Mac isn't the fix" (v8 result) | Consistent with moving parameters but not changing the image/nearness to the singular region |
| The two chat finetunes (best chat loss 1.91) still no generation | Fine-tuning slides within / slightly off the manifold without moving onto the fluent-sequences region |

Honest caveat: this is an interpretation, not a proof. We have measured perplexity and
robustness, not a description (equations) of this repo's neuromanifold. The lecture's
toolkit (defining equations, distance degree, Fisher metric) would be the way to make
the interpretation testable: e.g., estimate the local rank of the Jacobian around the
v8 checkpoint, and check whether the near-manifold dimension collapses after generation
diverges.

## Also in the lecture
- Universal approx. as the "big enough hidden width" theorem, with the practical caveat
  that infinite capacity is not available ("Bay Area brute force" framing).
- Rational/polynomial/networks-over-finite-fields as research strands; partial-fraction
  exercise tying 3-variable rational functions to symmetric-matrix rank.
- The camera/Cubism analogy for math surviving AI; recommendation to keep building
  foundational (not just empirical) measurement of model behavior.

## Suggested follow-ups
1. Compute Jacobian-rank estimate (frozen activations) around `octo_transformer_best.pt`
   once retraining completes; record number of near-zero singular values per layer.
2. Add a "manifold card" line to RESULTS.md (near-manifold dimension estimate + task).
3. If pursuing geometry further, study siguato/tensor decompositions of the attention
   maps (`core/tetrahedral_attention.py`) as algebraic varieties of fixed architecture.
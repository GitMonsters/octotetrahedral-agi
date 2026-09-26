# Course Notes — The Geometry of Neural Networks / Neuromanifolds

Lecture 1 (reading-course intro). Speaker: algebraic geometer working on tensor
algebraic geometry for ML (grad school UC Berkeley; postdoc UBC). Format: light
intro -> math overview -> hardcore research note.

---

## Part I — Course logistics

- Format: reading course, one semester, students present sections in pairs,
  signup sheets for topics, need | maybe | yes.
- Book: a monograph with an entire chapter on neuromanifolds (surveyed several
  math/AI books; chosen for its mathematical presentation and structure).
- Reading plan:
  - Sections 1-2: motivation, activation functions.
  - Sections 3-4: how networks learn — gradient descent, regularization, loss/cost.
  - Sections 5-6: feed-forward networks, backprop, geometry of architectures.
  - Sections 7-10: universal approximation and constructions (volunteer talks).
  - Chapters 13-14: new material beyond standard ML.
- Prerequisites (nice-to-have): linear algebra (matrices, vectors, linear maps, rank,
  kernels, inner products); multivariable calculus (partials, chain rule, gradient,
  Jacobian); basic probability (random variables, expectation, variance, densities).
- Self-check exercise: expand a homogeneous degree-2 polynomial in two variables;
  identify the coefficient map R^3 -> R^3 (feature map of a small network); compute its
  Jacobian, rank, injectivity, and image. "The first small neural network we study."
- Structure advice for talks: first ~1/3 accessible to grad students, middle 1/3 for
  professors, final 1/3 hardcore for specialists.

## Part II — What is a neuromanifold?

Definition sketch:
- Parameters `theta` live in a finite-dimensional space.
- The space of functions is infinite-dimensional.
- The map `theta -> f_theta` (architecture viewed as parametrization) embeds parameters
  into function space; the **image is the neuromanifold** — the set of functions the
  architecture can express.

Example (pairing): parabola `y = x^2`; re-parametrization `(t^2, t^4)` travels the
upper half only (to the origin and back). Neuromanifold = half-parabola; its closure =
the whole parabola. Singularity where 'd/dtheta' of the parametrization vanishes.

Mechanism worth remembering:
- Singular points = points where the Jacobian of the parametrization drops rank.
- Conjecture/experience: networks trained toward data outside the manifold get
  attracted to singularities and get stuck ("when the network doesn't train well it
  usually reached a singularity and cannot escape"). Single network stuck vs families
  studied via geometry.

## Part III — Two perspectives

| | Algebraic geometry | Information geometry |
|---|---|---|
| What you know | Exact function (polynomial/rational) | Samples from a function |
| Ambient space | Finite-dim coefficient space | Distribution/probability space |
| Geometric object | Algebraic variety / semi-algebraic set | Riemannian manifold |
| Metric/loss | Euclidean (distance degree) | Fisher matrix, neg-log-likelihood, KL |
| Typical theorems | Universal approximation (polynomial-like) | Statistical estimation, IG bounds |
| Real-data modeling | "Platonic" object + sampling noise | Directly statistical |

Key sentence: algebraic geometry studies the **image** of the parametrization in the
Zariski/euclidean closure; information geometry studies the model as a statistical
manifold under the Fisher metric.

## Part IV — Research connections mentioned

- Partial fractions, 3 variables: write `linear/quadratic` as a sum of inverses of
  linear forms iff a certain symmetric matrix (built from quadratic-form coefficients)
  has rank 2. Small/beautiful example, exam-friendly.
- Projects: rational neural networks, polynomial networks, networks over finite fields,
  valued neural networks (current), a 6am tropical geometry course ("after that, linear
  things are important").
- Fields motif: new math-focused AI-safety institute in the Bay Area (~100 researchers);
  motivation = we lack a mathematical foundation for measuring safety of large models.
- Universal approximation is a "big-enough-width" statement; caveat = real compute limits.

## References / further reading mentioned

- "Invitation to Neuromatgebraic Geometry" (survey expanding the talk).
- Finite vs Infinite Games (Carse) — perspective shift.
- Concerning the Spiritual in Art (Kandinsky) — math and art; beauty.
- Photography analogy: Daguerre didn't kill painters; it changed what paintings are
  for (impressionism, cubism, minimalism, serialism; Picasso's 4D portraits;
  relativity changed how art thinks about space/time). Math will similarly shift, not
  vanish.

## Takeaway

"Math is going to be fine": intelligent systems need *geometric* accounts of their
function spaces (what is expressible, identifiable, and where optimization can stall)
— exactly the tools algebraic + information geometry provide.
# Design Journal

**Purpose.** A running record of algorithmic ideas we've considered for this
problem, what we've concluded about each, and why. Complements `findings.md`
(which holds the empirical audit of the original guess-and-flip design) and
`ellipsoid-approach.md` (the original design doc, partially falsified).

**Pick-up-cold summary.** The original guess-and-flip design with continuous
ellipsoid (analytic-center) fitness is falsified — `findings.md` has concrete
counterexamples. What remains is a cluster of related ideas centered on using
geometric continuous moves plus discrete escape heuristics to find the unique
`x* ∈ {−1,+1}^N` satisfying a sparse `Ax = b` system. This journal captures
that cluster.

---

## 1. Problem recap

- N variables `x_i ∈ {−1,+1}`.
- m ≈ N/3 constraints, each a sparse 3-variable `{−1,+1}`-coefficient linear
  equation `a_k · x = b_k`.
- Exactly one `x* ∈ {−1,+1}^N` is promised to satisfy all constraints.
- Goal: recover `x*`. Aspirational scale: N up to millions.

---

## 2. Two framings we've been flipping between

**Framing A — "drag continuous `c` to `−x*`".** The original "expand to
opposite vertex" story. The algorithm's job is to move a continuous center
`c` so that it literally lands at `−x*` (or equivalently near the vertex we
care about), maximising some distance-to-target quantity.

**Framing B — "make sign(c) equal x*".** The algorithm's output is a sign
pattern. `c` is a diagnostic that drives discrete decisions via sign
extraction. Being at `c = −x*` exactly is not required; being in the right
orthant is.

The literal expand-to-`−x*` framing (A) was disproven earlier
(`bea6d45`) — blind reflection/projection operations on constraint
planes point toward `x*`, not away from it. Framing B survives that
disproof by using the guess to break the symmetry. **Effectively every
idea we're still considering operates under Framing B.** This matters
for interpreting things like clamping and distance-preservation below.

---

## 3. Key theoretical observations we established this session

### 3.1 Reflection across a constraint plane preserves Euclidean distance to x*

Since `x*` lies on every constraint plane `H_k: a_k · x = b_k`, any
reflection across `H_k` is an isometry fixing `x*`. Therefore:

```
‖c' − x*‖² = ‖c − x*‖²    for c' = reflection of c across H_k.
```

**Consequences:**
- No sequence of reflections ever brings the center closer to (or farther
  from) `x*`. Reflections alone cannot solve the problem; they can only
  move the center around on a sphere centered at `x*`.
- This makes reflection a *natural perturbation* for iterated local
  search — it doesn't undo progress toward the target.

### 3.2 Projection (Kaczmarz step) does reduce distance to x*

For `c_proj = c − (a_k·c − b_k)/‖a_k‖² · a_k`:
```
‖c_proj − x*‖² = ‖c − x*‖² − (a_k·c − b_k)² / ‖a_k‖²
```

Strictly decreases whenever the constraint is violated. So projections
make continuous progress; reflections do not. Kaczmarz converges to a
point in the affine subspace `{x : Ax = b}`, but that subspace is
`(N−m)`-dimensional — generically it contains many points, only one of
which is `x*`. Continuous convergence doesn't pick `x*` out; a discrete
commitment is still needed.

### 3.3 Reflection reach is confined to span(a_k)

Each reflection moves `c` by a scalar multiple of `a_k`. So starting
from `c_0`, after any sequence of reflections, `c − c_0 ∈ span(a_1, …,
a_m)`, dimension at most m = N/3.

**Consequence:** reflections alone explore at most an m-dimensional
affine subspace. The sphere around `x*` has dimension N−1. Only an
(m−1)-dimensional subsphere is reachable from a given `c_0` — roughly
1/3 of the full sphere. Two-plane operations (which move along
coordinate axes `e_i`) supply the remaining `N − m` dimensions, so the
*combination* has full reach.

### 3.4 Clamping to the box preserves signs, monotonically reduces distance to x*

If a reflection sends `c` outside `[−1,1]^N`, coordinate-wise clamping
back to the box:
- Never flips a sign (clamp preserves sign of each coordinate).
- Monotonically reduces `‖c − x*‖²` because `x* ∈ [−1,1]^N`.

**Consequence under Framing B:** clamping is fine. The algorithm reads
off signs, and clamping doesn't change them. Distance reduction is
harmless or helpful.

**Consequence under Framing A:** clamping is damaging. The literal
"max distance to `x*`" objective is hurt by clamping, and there's a
specific geometric pathology — the sphere of maximum distance `2√N`
touches the box only at `−x*`, so any out-of-box excursion must lose
distance when brought back. This is one reason Framing A is delicate;
Framing B is the cleaner interpretation.

### 3.5 On the Boolean cube, functions with unique local maximum are essentially linear

A function `f : {−1,+1}^N → R` whose multilinear extension is concave is
linear (a classical fact). So *any* fitness that provably has a unique
Hamming-1 local max must be a linear function `c · s`. For such a
fitness, `argmax = sign(c)`. The question "does there exist an
efficiently-computable local-max-free fitness?" reduces to "can we
compute a coefficient vector `c` with `sign(c) = x*` from only (A, b)?"
— which is just the original problem. No magic here; magnitudes don't
help, only signs matter on the cube.

### 3.6 Local maxima come from constraint geometry, not from fitness choice

Empirically (findings.md) and theoretically, the number of Hamming-1
local optima under the discrete residual, under the continuous
ellipsoid fitness, and under reasonable alternatives tracks the
*frustration* of the constraint system (how much constraints overlap
in mutually-conflicting ways). Swapping one fitness for another changes
constants, not the asymptotic existence of traps.

### 3.7 The glassy wall: why no stable local-move algorithm scales

**Overlap Gap Property (OGP).** In random CSPs with a planted unique
solution past the "shattering" threshold, near-solutions cluster into
groups separated by Hamming gaps of order Θ(N). The medium Hamming
range is empty.

**Consequence (Gamarnik et al.).** Any algorithm that is "stable"
(output changes continuously with input) — including hill-climb with
bounded moves, BP with bounded iterations, Feasibility Pump,
Douglas-Rachford, AMP, and any local-geometry-based hybrid we might
build — cannot find solutions in the glassy regime in polynomial time
with high probability.

This is the "glassy wall." It isn't specifically about bit-flip
neighborhood size; it's about **locality of computation**. Breaking
the wall requires either non-local computation (high-degree SoS /
Lasserre), problem-specific algebraic structure (lattice reduction,
XOR-SAT via Gaussian elimination), or a different computational model
(quantum).

---

## 4. The idea catalog

Format: **name** — status — one-line rationale.

### 4.1 Tried / falsified

**Guess-and-flip with continuous ellipsoid (analytic-center) fitness.**
**Falsified** by counterexamples at N = 6, 8, 10 (`findings.md`). `x*`
is frequently not the global max of the continuous fitness; worse, for
many instances `x*` is not even continuous-feasible. Core claim C3 of
the original doc is empirically false.

**Expand-to-opposite-vertex (Framing A taken literally).** **Disproven**
earlier (`bea6d45`). Blind reflection/projection operations point
toward `x*`, not away. Requires a guess to break symmetry, at which
point it's Framing B.

### 4.2 Recommended / best-candidate for scale

**Belief propagation on the factor graph.** **Recommended.** Native
algorithm for sparse factor graphs like ours. Per-iteration cost O(m).
Converges in effectively linear time when instances are below the
clustering transition. Extends strictly further into hard regimes than
any local-search variant.

**Survey propagation.** **Recommended fallback.** Extends BP into the
clustered regime by tracking "which cluster is this variable in"
instead of just "what's this variable's value." State-of-the-art for
hard random K-SAT. Same O(m) per-iteration cost.

**Peeling / unit propagation preprocess.** **Always do this first.**
Any `|b_k| = 3` constraint pins all three of its variables immediately.
Fixing them cascades (reduces some other constraints to `|b|=3`, etc.).
Often solves a large fraction of random instances outright and reduces
the remainder to a smaller "core."

### 4.3 Viable in moderate regime, bounded by the glassy wall

**Clause-weighted hill climb (GLS / SAPS / DDFW).** Weighted residual
`f_w(s) = −Σ w_k (a_k·s − b_k)²`; bump weights on stuck clauses. Every
`f_w` in the family shares `x*` as unique global max, and different
weights give different local-max structures, so weight perturbation
shakes the algorithm out of basins. Strong empirical performance on
moderate instances. Standard CSP solver technique.

**WalkSAT-style escape moves.** When stuck, pick a violated constraint,
flip one of its variables. Hamming-≤ k moves restricted to the support
of currently-violated constraints. Structurally-filtered version of
VNS; scales because the candidate set is `O(m·d²)` not `O(N^k)`.

**VNS with structural filtering (user's idea this session).** Start at
Hamming-1; if stuck, try Hamming-2 over variable pairs within violated
constraints; widen k on failure; reset on success. Realisable scale:
`k ≤ 3` routinely, `k ≤ 4–5` sometimes. Moderate-regime performance
should be very good. Not a glassy-wall breaker (OGP still applies).

**ILS with reflection as perturbation (user's idea this session).**
Improvement operator = continuous descent via projections plus
two-plane barriers (= the analytic-center computation). Perturbation =
reflect `c` across a chosen constraint plane when stuck. Distance to
`x*` is preserved across perturbation — cleaner than random restart.
Plausibly competitive with clause-weighting on moderate instances;
worth implementing and measuring. Does NOT break the glassy wall.

**Simulated annealing on ‖As−b‖².** Mature, cheap per step, handles
moderate difficulty well via temperature schedule. Provably converges
globally with slow-enough cooling (exponentially slow in worst case).

### 4.4 Conceptually useful but not standalone

**Kaczmarz iteration (alternating projections).** Continuous
convergence to the affine subspace `{x : Ax = b}`. Building block; the
limit is `(N−m)`-dimensional so rounding is still needed to pick `x*`.

**Douglas-Rachford splitting / difference map.** Reflections between
affine subspace and the vertex set `{−1,+1}^N`. Genuinely used for
combinatorial feasibility (sudoku, graph colouring). Continuous-to-
discrete pattern is mainstream — the user's framing is closer to this
literature than earlier journal entries suggested.

**Max-volume inscribed ellipsoid (log-det fitness).** Alternative to
analytic-center fitness. Untested; landscape likely similar (same
frustration geometry produces same local-max structure).

**Multiple ellipsoids / ensembling over guesses.** Intersecting
analytic centers from different sign-pattern guesses gives an
averaging scheme. Reduces variance, not systematic bias. Does not
eliminate local maxima.

**Random coefficients in linear fitness.** Magnitudes of a linear
fitness don't affect `argmax` on `{−1,+1}^N`; only signs matter. So
"random coefficients that happen to have max at `x*`" reduces to
"knowing x*." Red herring *for linear fitnesses*. Legitimate when
interpreted as clause weighting (non-linear) — but then it's the
weight-shake mechanism of GLS, not a new idea.

### 4.5 Wall-breakers (in theory)

**SoS / Lasserre hierarchy at high degree.** Polynomial-time at fixed
degree d, cost `N^O(d)`. Not ruled out by OGP because the computation
is global, not local. For some problem families, moderate d (say 4–8)
is sufficient to recover `x*` exactly above the BP threshold.
Expensive; unlikely to scale to N = 10⁶ without exploiting sparse
structure.

**Quantum algorithms.** Grover's √N search is still astronomical for
N = 10⁶. Adiabatic has its own glassy problem (exponentially small
gaps). Not a practical tool today at our scale.

**Problem-specific structure exploitation.** If our instances have
special structure — algebraic symmetries, low-rank components,
hierarchical decomposition — we may be able to reduce to a tractable
sub-problem. Worth investigating per-instance, not as a general
algorithm.

---

## 5. Recommended pipeline (best current thinking)

For deployment on unseen instances at target scale:

1. **Peel / unit-propagate** until no `|b_k| = 3` constraints remain.
   Often collapses a large fraction of the problem.
2. **Run BP** on the residual factor graph, with damping (α ≈ 0.5) and
   decimation (fix most-confident variable, simplify, repeat).
3. **If BP fails** (non-convergence or inconsistent decimation), try
   **SP** with similar machinery.
4. **If SP fails**, fall back to **clause-weighted hill climb with
   random restarts**. Time-box it.
5. **Diagnostic**: at any failure, check whether we're in the glassy
   regime for our m/N, overlap, and `|b|` distribution. If we are,
   algorithmic progress past this point probably requires SoS-level
   machinery or structural insight into the specific instance family.

The ILS-with-reflection variant and structurally-filtered VNS are
reasonable additions at step 4 — they're local-search variants in the
same OGP-bounded class but may have better constants than plain clause
weighting on our instance distribution. Worth measuring before
committing.

---

## 6. Open questions

1. **Which regime do our target instances sit in?** The phase
   transition for this exact constraint model (sparse 3-variable
   `{−1,+1}` linear equations with planted unique solution) may exist
   in the literature but we haven't looked it up. We should: it
   determines whether the entire pipeline above is needed or whether
   step 2 alone suffices.
2. **Does ILS-with-reflection outperform clause-weighted hill climb
   on our instances?** Testable directly on the existing test bed.
   Not yet measured.
3. **Does a structurally-filtered VNS (Hamming-k escape restricted to
   violated-constraint supports) close the ~35–40% hill-climb failure
   gap on `|b|=1` instances?** Testable, not yet measured.
4. **Is there a specific algebraic structure in our constraint
   distribution that a dedicated solver could exploit?** Not
   investigated.

---

## 7. What's in the test bed and what's missing

**Currently measured** (`experiment_hillclimb.py`, `experiment_v4.py`,
`counterexample.py`):
- Instance generation with verified uniqueness.
- Discrete residual `‖As−b‖²` hill climb; local-minimum counts,
  success rates.
- Continuous ellipsoid fitness with custom dual-Newton solver;
  feasibility and ranking analysis.
- Counterexample finder for the ellipsoid-fitness design.

**Not yet implemented, worth adding** (ordered by value):
- **Belief propagation** with damping + decimation. ~60 lines.
  Highest-value addition: directly answers "which regime are we in."
- **Clause-weighted hill climb** (GLS-lite). ~20 lines on top of the
  existing residual hill climb.
- **Structurally-filtered VNS**. ~30 lines. Adaptive Hamming-k escape
  over violated-constraint supports.
- **ILS with reflection perturbation**. ~50 lines. The user's design;
  compares directly to clause weighting on the same instances.
- **Douglas-Rachford baseline.** ~40 lines. Standard literature
  reference point for the geometric-continuous-to-discrete pattern.

Adding the top two items would give us enough data to make the "easy
vs moderate vs glassy" call on our instance distribution at moderate
N, which in turn tells us whether to invest in the geometric-hybrid
design or commit to BP as the engine.

---

## 8. What this journal is NOT

- A plan of record for the next session. It's a catalog; the plan
  lives in TODOs or in the user's head.
- A literature review. I flagged Douglas-Rachford, Feasibility Pump,
  and GLS as adjacent, but a proper lit search (especially for the
  specific IPM+VNS+DR hybrid) has not been done and would be useful
  before any novelty claim.
- A proof. The OGP and glassy-wall claims are reported, not
  re-derived. Primary sources: Gamarnik's 2014-2020 papers on OGP,
  Mezard-Parisi-Zecchina on SP, Achlioptas-Ricci-Tersenghi on
  clustering transitions.

---

*Last updated by Claude session 2026-04-24, following a long design
conversation. Keep this file updated as a living record.*

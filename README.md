# Math Algorithm Experiments

A collection of math algorithm experiments and explorations.

## Problem 1: Hypercube Vertex from Hyperplane Intersections

### Problem Statement

Given:
- **N** dimensions
- A set of **m** hyperplanes, each defined by a linear equation **a** · **x** = b
- A **guarantee** that exactly one point satisfies all constraints

Find the unique point **x** ∈ {-1, 1}^N (a vertex of the N-dimensional hypercube
centered at the origin) that lies on all of the provided hyperplanes.

### Parameters
- **m ≈ N/3** — far fewer equations than variables
- **N up to millions** (aspirational)
- **Sparse**: each equation has ~3 non-zero coefficients in {-1, 1}
- **Unique solution guaranteed**

### Current working design

The current candidate is a **dual-ellipsoid surface-preserving
algorithm** defined in `design-journal.md` §9. Two **axis-aligned**
ellipsoids share a single center `c`: ellipsoid 1's surface contains
`x*`, ellipsoid 2's surface contains `-x*`. A four-function fitness
(sum of radii, sphericity drive, per-ellipsoid radii) drives `c` to
the segment between `x*` and `-x*`, from which sign extraction
recovers `x*`. Correctness is proven by triangle inequality — the
segment is the unique minimizer, no basins, no glassy-wall
obstruction.

Both ellipsoids share **center `c` AND per-axis stiffness `A`**:

    E_i = { x : Σ_j A[j] (x_j − c[j])² = R_i_sq }

Only the scalars `R_1_sq` and `R_2_sq` differ between them. The
`DualEllipsoidState` dataclass bundles `(c, A, R_1_sq, R_2_sq)`:

```python
state = DualEllipsoidState.initial_sphere(N)   # c=0, A=ones, R=N
state = paired_op_hyperplane(state, a, b, M)   # general sparse a
state = paired_op_two_plane(state, k, delta)   # box constraint x_k^2=1
lo, hi = state.valid_range_two_plane(k)
ok, val = state.x_on_E1(x_star)
ok, val = state.x_on_E2(-x_star)
```

Paired ops are immutable: each call returns a new state.

Two operations are implemented in `surface_preserving_ops.py`:

- **Op1 (general sparse hyperplane `aᵀx = b`)**: **linear pencil**
  `M · (aᵀx − b)`. The added term has no quadratic part, so it does
  not introduce any off-diagonal cross terms — axis-alignment is
  preserved for *any* sparse `a` (multiple non-zero entries are fine).
  The motion is `newC[i] = C[i] − M·a[i]/(2·A[i])`; `A` is unchanged.
  Paired version uses `+b` on E_1 and `−b` on E_2; with shared `A`,
  the same `M` lands both ellipsoids at the same new center
  automatically. `R_1_sq` and `R_2_sq` diverge by `2·M·b` per call.
  By Cauchy-Schwarz the new R values stay positive for any `M`, so
  the parameter is unbounded.

- **Op2 (two parallel planes `x_k = ±1`)**: quadratic pencil
  `M · (x_k² − 1)` vanishes on `x_k² = 1`, preserving both x* and -x*
  without committing to a sign. Affects only the (k,k) diagonal of
  the shape, so axis-alignment is preserved. Paired version uses the
  same `delta` on both sides; shared `A` and shared `c` are
  preserved. Cannot move `c[k]` from `c[k] = 0` — the formula is
  forced to return zero motion. Use Op1 to break the symmetry first.

**Why this works**: the linear pencil for Op1 was the missing piece.
The earlier quadratic pencil `s·(aᵀx − b)²` introduces cross terms
`s·a_i·a_j` that break axis-alignment when `a` has multiple non-zero
entries. The linear pencil has no cross terms at all, so it works
for any sparse `a`. The original problem's `m` constraint hyperplanes
from `Ax = b` therefore enter the per-step ellipsoid update directly.

### Document map

**Top level — current path:**
- `README.md` — this file.
- `design-journal.md` — **living design document**. §9 holds the
  current algorithm; §1–§8 capture the idea catalog and history of
  what's been ruled in or out.
- `surface_preserving_ops.py` — `DualEllipsoidState` dataclass plus
  Op1 and Op2 (single-ellipsoid and paired versions), axis-aligned
  form. Valid-range helpers (`valid_delta_range_op_hyperplane`,
  `valid_delta_range_op_two_plane`, also exposed as
  `state.valid_range_*` methods) return the valid delta interval
  given the current state, so the descent can pick step sizes without
  trial-and-error.
- `test_surface_preserving.py` — 24 tests covering identity, motion
  direction, axis-alignment preservation under both ops, x* and -x*
  preservation under general sparse hyperplanes, unbounded-M
  validity for Op1, mixed long-horizon composition of Op1+Op2, the
  kick off origin via Op1 with general sparse a, the c[k]=0
  obstruction for Op2, shared-A invariant under both ops, R-divergence
  pattern (Op1 changes R_1−R_2 by 2·M·b, Op2 leaves it unchanged),
  range-helper correctness, and DualEllipsoidState immutable updates.

### Valid-delta ranges

Given the current state, each paired op has a parameter range within
which the call succeeds:

- **Paired Op1** (general hyperplane `aᵀx = b` on E_1, `−b` on E_2):
  `M ∈ ℝ` — unbounded. Cauchy-Schwarz on `(b − aᵀC)² ≤ R_i_sq · Σ a²/A`
  (the condition that x* / −x* is on its respective surface) makes the
  discriminant of the parabola `newR_i_sq(M)` non-positive, so
  `newR_i_sq > 0` for any M.
- **Paired Op2** (two-plane `x_k² = 1`):
  `delta ∈ (−c[k], +∞)` if `c[k] > 0`,
  `delta ∈ (−∞, −c[k])` if `c[k] < 0`,
  `delta = 0` if `c[k] = 0`.
  One-sided unbounded; the wall is at `c_new[k] = 0` (Op2 cannot move
  c[k] across zero).

**`archive/` — historical / falsified work, kept as evidence:**
- `archive/ellipsoid-approach.md` — the original (falsified) design.
- `archive/findings.md` — empirical audit of the falsification with
  concrete N = 6, 8, 10 counterexamples.
- `archive/experiment_hillclimb.py`, `archive/experiment_v4.py`,
  `archive/counterexample.py` — the small-N test bed that produced
  `findings.md`.
- `archive/analyze_v3.py` — empirical demonstration that the standard
  MVCE update doesn't preserve x* on surface; the motivation for the
  pencil construction now in `surface_preserving_ops.py`.

See `archive/README.md` for more on what's there.

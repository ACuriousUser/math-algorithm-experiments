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

Each ellipsoid is parameterized by `(c, A, R_sq)` where the equation
is `Σ A[j] (x_j - c[j])² = R_sq`. The dual-ellipsoid state — shared
center `c`, separate `(A_i, R_i_sq)` per ellipsoid — is bundled into
a `DualEllipsoidState` dataclass:

```python
state = DualEllipsoidState.initial_sphere(N)   # c=0, A=ones, R_sq=N
state = paired_op_hyperplane(state, k, b, delta)
state = paired_op_two_plane(state, k, delta)
lo, hi = state.valid_range_hyperplane(k, b)
lo, hi = state.valid_range_two_plane(k)
ok, val = state.x_on_E1(x_star)
ok, val = state.x_on_E2(-x_star)
```

Paired ops are immutable: each call returns a new state.

Two operations are implemented in `surface_preserving_ops.py`:

- **Op1 (axis-aligned hyperplane `x_k = b`)**: pencil
  `s · (x_k - b)²` preserves any point with `x_k = b`. Moves c[k] by
  a chosen `delta` along `e_k`; A[k] and R_sq update. Commits to a
  sign hypothesis: I1 holds only when `b = x*_k`. Paired version uses
  `+b` on E_1 and `-b` on E_2 (consistent with `-x*_k = -b`).

- **Op2 (two parallel planes `x_k = ±1`)**: pencil
  `M · (x_k² - 1)` preserves any vertex with `x_k² = 1`, so it
  preserves both x* and -x* simultaneously without committing to a
  sign. Cannot move `c[k]` from `c[k] = 0` (formula is forced to
  return zero motion). Use Op1 to break the `c[k] = 0` symmetry first.

The axis-aligned form makes the math much simpler than the general
case: because each ellipsoid's i-th column of the shape matrix is
parallel to `e_i`, the natural pencil motion is automatically along
`e_i`, so a single shared `delta` on both sides of the paired version
lands at the same new center. The cost is that Op1's hyperplane must
be axis-aligned (`a = e_k`); the original problem's general
constraint hyperplanes (rows of A with multiple nonzero entries)
cannot enter the per-step update directly. They must enter through
the descent's fitness function or a separate mechanism.

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
- `test_surface_preserving.py` — 28 tests covering identity, motion
  along `e_k`, axis-alignment preservation, x* preservation,
  composition drift, paired versions' shared-c invariant under
  divergence of A1/A2, the kick off origin, the c[k]=0 obstruction
  for Op2, degenerate-input handling, range-helper correctness
  (endpoints exactly where the op transitions from valid to invalid),
  and `DualEllipsoidState` construction / immutable updates.

### Valid-delta ranges

Given the current state, each paired op has a delta range within
which the call succeeds:

- **Paired Op1** (hyperplanes `x_k = b` on E_1, `x_k = -b` on E_2):
  `delta ∈ (-1 - c[k], 1 - c[k])` — equivalently `c_new[k] ∈ (-1, 1)`.
  Bounded interval; independent of A_i and R_i_sq when x* is on E_1
  and -x* on E_2.
- **Paired Op2** (two-plane `x_k^2 = 1`):
  `delta ∈ (-c[k], +∞)` if `c[k] > 0`,
  `delta ∈ (-∞, -c[k])` if `c[k] < 0`,
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

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
algorithm** defined in `design-journal.md` §9. Two ellipsoids
maintain x* and −x* on their surfaces respectively (invariants I1 and
I2), and a coordinate-descent loop over the operation parameters
shrinks each ellipsoid until its center concentrates near its target.

The two surface-preserving operations (Op1 and Op2) are implemented in
`surface_preserving_ops.py` and verified by 20 tests across two test
files. The descent itself is implemented in `dual_ellipsoid_descent.py`
and characterized empirically by `test_dual_ellipsoid_descent.py`.

### Empirical status

With weight profile `{size: 5, sph: 1, align: 0}` the descent solves
**40 / 40** small-N instances tested:

| N | m | E1 success | E2 success |
|---|---|-----------|-----------|
| 6 | 4 | 10 / 10 | 10 / 10 |
| 6 | 5 | 10 / 10 | 10 / 10 |
| 8 | 5 | 5 / 5 | 5 / 5 |
| 8 | 6 | 5 / 5 | 5 / 5 |
| 10 | 6 | 5 / 5 | 5 / 5 |
| 10 | 7 | 5 / 5 | 5 / 5 |

"Success" = `sign(c) = ±x*` exactly. The original §9 "shared-center /
triangle-inequality" framing turned out *not* to be the operative
correctness story — see §9.9 of the journal for the empirical
analysis. The next hurdle is N ≈ 100+ in the glassy regime; the
current descent is O((m+N)·N³) per round and will need restructuring
to get there.

### Document map

**Top level — current path:**
- `README.md` — this file.
- `design-journal.md` — **living design document**. §9 holds the
  current algorithm; §1–§8 capture the idea catalog and history of
  what's been ruled in or out. §9.8–§9.10 hold the empirical descent
  results.
- `surface_preserving_ops.py` — the two surface-preserving ellipsoid
  operations (Op1 = hyperplane pencil, Op2 = two-plane / box pencil).
- `dual_ellipsoid_descent.py` — coordinate-descent loop over Op1 / Op2
  parameters that shrinks both ellipsoids to their targets.
- `test_surface_preserving.py`, `test_two_plane_op.py` — tests
  verifying I1 (and I2 for Op2) under each operation and their
  compositions.
- `test_dual_ellipsoid_descent.py` — empirical evaluation: instance
  sweep, weight-profile comparison, success-rate table.

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

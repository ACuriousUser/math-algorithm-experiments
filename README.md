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
algorithm** defined in `design-journal.md` §9. It uses a four-function
fitness (sum of radii, sphericity drive, and per-ellipsoid radii) to
drive a continuous center `c` to the segment between `x*` and `-x*`,
from which sign extraction recovers `x*`. Correctness is proven by
triangle inequality — the segment is the unique minimizer, no basins,
no glassy-wall obstruction.

The two surface-preserving operations the algorithm depends on (Op1 and
Op2) are implemented in `surface_preserving_ops.py` and verified by 20
tests across the two test files. The next piece of work is the descent
algorithm itself (not yet written).

### Document map

**Top level — current path:**
- `README.md` — this file.
- `design-journal.md` — **living design document**. §9 holds the
  current algorithm; §1–§8 capture the idea catalog and history of
  what's been ruled in or out.
- `surface_preserving_ops.py` — the two surface-preserving ellipsoid
  operations (Op1 = hyperplane pencil, Op2 = two-plane / box pencil).
- `test_surface_preserving.py`, `test_two_plane_op.py` — tests
  verifying I1 (and I2 for Op2) under each operation and their
  compositions.

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

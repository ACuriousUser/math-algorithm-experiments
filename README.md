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
algorithm** defined in `design-journal.md` §9. Under the assumed
invariants I1–I3, it uses a four-function fitness (sum of radii,
sphericity drive, and per-ellipsoid radii) to drive a continuous center
`c` to the segment between `x*` and `-x*`, from which sign extraction
recovers `x*`. Correctness under I1–I3 is proven by triangle inequality
— the segment is the unique minimizer, no basins, no glassy-wall
obstruction.

**Open subgoal**: actually constructing ellipsoid-shaping operations
that satisfy I1–I3, computable from `(A, b, c, P)` alone. This is the
next piece of work.

### Document map

- `README.md` — this file, one-paragraph overview.
- `design-journal.md` — **living design document**. §9 holds the
  current algorithm; §1–§8 capture the idea catalog and history of
  what's been ruled in or out.
- `findings.md` — empirical audit of the earlier guess-and-flip design
  (falsified with concrete counterexamples at N = 6, 8, 10).
- `ellipsoid-approach.md` — historical design doc from the original
  guess-and-flip iteration. Retained for reference; core claims
  (specifically, the C3 "correct guess has highest fitness" claim)
  are empirically falsified; see `findings.md`.
- `experiment_*.py`, `analyze_*.py` — small-N test bed and diagnostic
  scripts.

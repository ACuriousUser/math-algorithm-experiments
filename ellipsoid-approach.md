# Hypercube Vertex Finder: Ellipsoid Surface Algorithm

## Problem

Given N variables, each in {-1, 1}, and m ≈ N/3 sparse hyperplane constraints
(each involving ~3 variables with {-1,1} coefficients), find the unique vertex
x* ∈ {-1,1}^N satisfying all constraints.

---

## Algorithm Design Evolution

### Design A: Expand Until Center Reaches -x* (DISPROVEN)

**Idea**: keep x* on the ellipsoid surface, expand until center reaches -x*.
Then x* = -(center).

**Why it fails**: operations that don't know x* push the center TOWARD x*,
not away from it. For a constraint a·x = b, the operation pushes the center
toward the centroid of all satisfying vertices. This centroid has positive
dot product with every satisfying vertex — so the center moves toward
EVERY satisfying vertex (including x*), and away from their negations.

With multiple constraints: the centroids converge to x* itself (the unique
intersection). The operations naturally push the center to x*, not to -x*.

**Proof sketch**: For constraint a·x = b with satisfying vertices S:
- The centroid of S lies on the hyperplane a·x = b
- Any symmetric operation using only (a, b) pushes the center toward this centroid
- For all v ∈ S: centroid · v = b/|S| > 0 (when b > 0)
- So the center moves toward every v ∈ S, meaning toward x*

**Key insight**: the constraint hyperplanes DEFINE x* (they intersect there).
Operations that use these hyperplanes naturally push toward x*, not away.
-x* does NOT satisfy the constraints, so nothing pushes toward it.

### Design B: Collapse Until Center Reaches x* (CURRENT)

**Idea**: keep x* on the ellipsoid surface, apply operations that push the
center TOWARD x*. The ellipsoid collapses. At convergence: the ellipsoid
has shrunk to a point at x*. The center IS x*.

**Why this might work**: the operations naturally push toward x* (proven above).
The question is whether they push ALL THE WAY to x* (collapsing the ellipsoid
to a point), or get stuck at an interior point.

---

## The Two Operations

**Operation 1: Hyperplane Slice**
Given constraint a · x = b (with ~3 nonzero {-1,1} coefficients):
- Reshapes the ellipsoid so its surface passes through the hyperplane
- x* satisfies a · x* = b, so x* stays on the surface
- The center shifts toward the centroid of satisfying vertices
- Vertices NOT satisfying the constraint fall off the surface

**Operation 2: Two-Plane Operation**
For variable i, using x_i ∈ {-1, 1}:
- Reshapes the ellipsoid so its surface reaches both planes x_i = +1
  and x_i = -1 (without choosing a side)
- x* has x*_i ∈ {-1,1}, so x* stays on the surface
- Continuous (non-vertex) surface points with x_i ∉ {-1,1} are affected

**Neither operation chooses a side or requires knowing x*.**
**Both operations push the center toward x* (not away).**

---

## Invariants

### Invariant 1: x* is always on the surface

x* satisfies every hyperplane constraint (a · x* = b) and every two-plane
constraint (x*_i ∈ {-1,1}). Both operations preserve points on the surface
that satisfy the operation's constraint. Since x* satisfies all of them,
x* remains on the surface throughout.

Initially: the sphere of radius √N centered at the origin has ALL vertices
of {-1,1}^N on its surface (||v||² = N for every vertex v).

### Invariant 2: center moves toward x*

Each hyperplane operation pushes the center toward the centroid of vertices
satisfying the constraint. Since x* is always among the satisfying vertices,
and other non-satisfying vertices are progressively removed, the centroid
converges toward x*. The center follows.

### Invariant 3: non-solution vertices leave the surface

Each hyperplane slice removes vertices not satisfying the constraint.
After all m hyperplane slices: only x* remains (by uniqueness guarantee).
The two-plane operations provide additional constraints.

---

## States of the Algorithm

### Start State

- **Shape**: perfect sphere of radius √N
- **Center**: origin (0, 0, ..., 0)
- **Surface**: all 2^N vertices of {-1,1}^N
- **x* status**: on the surface (among all vertices)

### Intermediate States

After applying some operations:
- **Shape**: ellipsoid (operations break spherical symmetry)
- **Center**: shifted from origin toward x*
- **Surface**: fewer vertices remain
- **x* status**: still on the surface (Invariant 1)

### Goal State

After all operations converge:
- **Shape**: degenerate ellipsoid (collapsed to a point)
- **Center**: at x*
- **Radius**: 0 (the ellipsoid IS x*)
- **Surface**: just x*
- **Solving**: center = x*. Read it off directly.

---

## Critical Open Question: Can the Center Reach x*?

The center must move from the origin to x* (distance √N). The ellipsoid
must collapse from radius √N to radius 0.

### What works in its favor

- **Direction**: operations push toward x* (proven). No wrong-direction risk.
- **Enough operations**: N/3 hyperplane slices + N two-plane operations = 4N/3
  operations for N unknowns. Each hyperplane slice reduces the effective
  dimension by 1 (from N to 2N/3 after all slices). Two-plane operations
  could provide the remaining N - N/3 = 2N/3 dimension reductions.
- **Uniqueness**: x* is the ONLY vertex satisfying all constraints. Once all
  other vertices are removed from the surface, the ellipsoid has no choice
  but to collapse to x*.

### What might prevent it

- **Interior stagnation**: smooth operations might push the center TOWARD x*
  without reaching it (asymptotic approach). The center gets stuck at the
  centroid of continuous surface points, not at the discrete vertex x*.
- **Two-plane weakness**: we showed earlier that the two-plane operation can
  be the identity (for axis-aligned ellipsoids). It might not provide the
  additional dimension reductions needed.
- **Insufficient constraints**: with only N/3 hyperplane slices, the ellipsoid
  lives in a 2N/3-dimensional subspace. Without additional collapse from
  the two-plane operations, the center stays at the centroid of this
  subspace (an interior point, not x*).

### Key sub-question

Can the two-plane operation (x_i ∈ {-1,1}) actually reduce the ellipsoid's
dimension? In the shrink framework with axis-aligned Q: no (identity).
With full Q (after hyperplane slices create off-diagonal correlations):
possibly. The two-plane operation might tighten the ellipsoid using the
correlations between variables, effectively determining some variables.

This is the critical gap between "pushed in the right direction" and
"actually arrives at x*."

---

## What We've Explored (Archive)

### Approach 1: Shrink Outer Ellipsoid
Start with sphere, shrink toward x*. Two-plane is identity for axis-aligned Q.
With full Q: has some power but O(N²) storage. Cascade (L + Δ_i > 1)
determines variables but might not complete for all instances.

### Approach 2: Analytic Center / Max Inscribed Ellipsoid
Maximize Σ log(1-c_i²) subject to Ac = b. Pushes center toward cube center
(AWAY from walls). Sign extraction unreliable.

### Approach 3: Guess-and-Flip with Fitness
Guess s, compute analytic center with sign constraints, compare fitness.
No local maxima (with uniqueness guarantee). But expensive and may be
overkill vs simpler discrete approaches.

### Approach 4: Expand to -x* (Design A above)
Disproven: operations push toward x*, not -x*.

---

## Next Steps

1. **Define the two-plane operation precisely for non-axis-aligned
   ellipsoids**: what does it do to the full Q matrix? Does it reduce
   the effective dimension?

2. **Trace a small example (N=3 or N=6)**: apply all operations step
   by step. Does the center converge to x*?

3. **Characterize the gap**: after N/3 hyperplane slices, the ellipsoid
   is (2N/3)-dimensional with center at the centroid. How far is this
   centroid from x*? What additional operations are needed to close
   the gap?

4. **Consider hybrid approach**: use the ellipsoid operations to get
   the center NEAR x* (determine sign(c_i) for most variables), then
   use constraint propagation or verification for the remaining
   ambiguous variables.

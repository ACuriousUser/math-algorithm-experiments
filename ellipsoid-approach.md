# Hypercube Vertex Finder: Ellipsoid Surface Algorithm

## Problem

Given N variables, each in {-1, 1}, and m ≈ N/3 sparse hyperplane constraints
(each involving ~3 variables with {-1,1} coefficients), find the unique vertex
x* ∈ {-1,1}^N satisfying all constraints.

---

## Algorithm Design: Expand Until Center Reaches Opposite Vertex

### Core Idea

Maintain a hyperellipsoid with x* always on its **surface**. Apply two
operations that reshape the ellipsoid, causing the center to migrate AWAY
from x*. At convergence: the ellipsoid becomes a perfect hypersphere centered
at -x* (the vertex furthest from x*), with x* on the surface at distance 2√N.

**Solving**: x* = -(center). Negate every coordinate of the center.

### The Two Operations

**Operation 1: Hyperplane Slice**
Given constraint a · x = b (with ~3 nonzero {-1,1} coefficients):
- Reshapes the ellipsoid so its surface passes through the hyperplane
- x* satisfies a · x* = b, so x* stays on the surface
- Vertices NOT satisfying the constraint fall off the surface
- The center shifts (away from removed vertices, toward the "surviving" side)

**Operation 2: Two-Plane Operation**
For variable i, using x_i ∈ {-1, 1}:
- Reshapes the ellipsoid so its surface passes through BOTH planes x_i = +1
  and x_i = -1 (without choosing a side)
- x* has x*_i ∈ {-1,1}, so x* stays on the surface
- Continuous (non-vertex) surface points with x_i ∉ {-1,1} are pushed off
- The surface is constrained to reach both walls for variable i

**Neither operation chooses a side or requires knowing x*.**

---

## Invariants

### Invariant 1: x* is always on the surface

**Why**: x* satisfies every hyperplane constraint (a · x* = b) and every
two-plane constraint (x*_i ∈ {-1,1}). Both operations preserve points on
the surface that satisfy the operation's constraint. Since x* satisfies all
of them, x* remains on the surface throughout.

**Initially**: the sphere of radius √N centered at the origin has ALL
vertices of {-1,1}^N on its surface (since ||v||² = N for every vertex v).
x* is one of them.

### Invariant 2: non-solution vertices leave the surface

Each hyperplane slice removes vertices that don't satisfy the constraint.
After all m hyperplane slices: only x* remains (by the uniqueness guarantee).

The two-plane operations provide additional constraints that complement
the hyperplane slices (N two-plane operations + N/3 hyperplane operations
= 4N/3 total constraints for N unknowns).

### Invariant 3: center moves away from x*

As non-x* vertices fall off the surface, the center of the ellipsoid
shifts. It moves away from the "surviving" vertex x* (because the
ellipsoid's mass is redistributed away from the removed side).

---

## States of the Algorithm

### Start State

- **Shape**: perfect sphere of radius √N
- **Center**: origin (0, 0, ..., 0)
- **Surface**: all 2^N vertices of {-1,1}^N lie on the surface
- **x* status**: on the surface (among all vertices)

### Intermediate States

After applying some operations:
- **Shape**: ellipsoid (asymmetric — operations break spherical symmetry)
- **Center**: shifted away from origin, moving toward -x*
- **Surface**: fewer vertices remain (those satisfying all applied constraints)
- **x* status**: still on the surface (Invariant 1)

### Goal State

After all operations converge:
- **Shape**: perfect hypersphere (symmetry restored — only x* constrains
  the shape, and one point doesn't break symmetry)
- **Center**: at -x* (the vertex of {-1,1}^N furthest from x*)
- **Radius**: 2√N (distance from -x* to x*)
- **Surface**: x* is the unique vertex on the surface
- **x* status**: on the surface, distance 2√N from center

**Verification of goal state**:
- Distance from -x* to x*: each coordinate differs by 2, so
  ||x* - (-x*)|| = √(N · 4) = 2√N ✓
- x* on surface: cost = (2√N)²/(2√N)² = 1 ✓
- Sphere intersects all hyperplanes: distance from -x* to hyperplane
  a·x = b is |a·(-x*) - b|/||a|| = 2|b|/√k ≤ 2√N for reasonable b ✓

**Solving**: center = -x*, therefore **x* = -center**. Negate each coordinate.

---

## What We've Explored (and Abandoned)

### Approach 1: Shrink Outer Ellipsoid (abandoned)

Start with sphere containing all vertices, shrink toward x*.
**Problem**: the two-plane operation (reshape using x_i ∈ {-1,1}) is the
identity for axis-aligned ellipsoids — it can't shrink the ellipsoid. With
full Q matrix: it has some power but insufficient for scalability (O(N²)).

### Approach 2: Analytic Center / Max Inscribed Ellipsoid (abandoned)

Maximize Σ log(1-c_i²) subject to Ac = b.
**Problem**: this pushes the center TOWARD the cube center (away from walls),
which is the opposite of what we want. The center never reaches a vertex.
The sign extraction (x_i = sign(c_i)) is unreliable for weakly-constrained
variables.

### Approach 3: Guess-and-Flip with Fitness (explored)

Guess s ∈ {-1,1}^N, compute analytic center with sign constraints, compare
fitness across guesses.
**Problem**: requires solving a convex optimization per guess, O(N) guesses
to converge — total cost O(N²) or O(N³). Also: unclear whether the
continuous fitness is needed vs simple discrete checks.

### Current Approach: Surface-Preserving Expansion

Maintain x* on the surface, expand until center reaches -x*.
**Status**: invariants verified, goal state well-defined. Open question:
can the operations push the center all the way to -x*?

---

## Open Questions

### Critical: Can the center reach -x*? (Question 3)

The center must move from the origin (start) to -x* (goal), a distance of √N.
Key sub-questions:

a) Do the operations push the center in the right DIRECTION (toward -x*)?
   Each operation pushes along its constraint normal. Do these directions
   collectively point toward -x*?

b) Do the operations have enough MAGNITUDE to move the center distance √N?
   With 4N/3 operations, each moving O(1), total movement is O(N). This
   exceeds √N for large N. But direction matters, not just magnitude.

c) Can the center reach an actual VERTEX (-x*), or does it get stuck in
   the interior? Smooth optimizations stay interior. But our operations
   are geometric (slices, reshapes), not smooth optimizations. They might
   have the power to reach the boundary.

d) Does -x* even satisfy any constraints? A·(-x*) = -b ≠ b generally.
   So -x* is NOT on the constraint hyperplanes. Can the center be at a
   point that's not on the hyperplanes?

   **This might require redefining the operations**: the hyperplane operation
   constrains the SURFACE to intersect the hyperplane, not the CENTER to
   lie on it. If the center is free (not forced onto hyperplanes), it CAN
   reach -x*.

### Secondary Questions

4) Does the ellipsoid actually converge to a sphere at the goal state?
5) How many operation-iterations are needed for convergence?
6) Can this be computed in O(N) time?
7) What is the precise mathematical definition of each operation
   (how exactly does it reshape the ellipsoid)?

---

## Next Steps

**Priority**: Investigate Question 3 — whether the operations can push
the center from the origin to -x*. This requires precisely defining
what each operation does to the ellipsoid (center, shape, and semi-axes)
and tracing the center's trajectory through a sequence of operations.

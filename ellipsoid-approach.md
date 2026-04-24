# Hypercube Vertex Finder: Ellipsoid Fitness Algorithm

> **HISTORICAL DOCUMENT.** This is the original guess-and-flip + analytic-
> center-fitness design. Its core claim (C3: "correct guess `s = x*` has
> highest continuous fitness") was **empirically falsified** in
> `findings.md` with concrete N = 6, 8, 10 counterexamples.
>
> Related claims that also didn't hold up under verification:
> - The "x* on ellipsoid surface" invariant was falsified for the
>   standard MVCE update in `analyze_v3.py`.
> - The "no local maxima" argument (C4) was aspirational; it fails
>   empirically because C3 fails.
>
> **For the current working design, see `design-journal.md` §9.** The
> geometric intuition about operations preserving surface membership
> of x* is retained there as an explicit assumption (I1–I3), with the
> construction of such operations flagged as the open subgoal.
>
> This file is kept as reference for the exploration history and for
> the individual observations (information-theoretic constraints,
> two-plane amplification, etc.) that remain valid even though the
> headline algorithm does not.

## Problem

Given N variables, each in {-1, 1}, and m ≈ N/3 sparse hyperplane constraints
(each involving ~3 variables with {-1,1} coefficients), find the unique vertex
x* ∈ {-1,1}^N satisfying all constraints.

---

## Algorithm: Guess-and-Flip with Ellipsoid Fitness

### Core Idea

1. Choose a candidate vertex s ∈ {-1,1}^N (the "guess")
2. Compute a **fitness score** for s: how compatible is this guess with the
   constraint hyperplanes? Measured via the analytic center of the feasible
   region defined by the guess's sign constraints
3. Try flipping each variable; keep the flip that improves fitness most
4. Repeat until no single flip improves fitness
5. Output the guess with highest fitness = x*

### Why It Works

- The **correct guess** (s = x*) has ALL sign pushes cooperating with the
  constraint hyperplanes → spacious feasible region → HIGH fitness
- **Wrong guesses** have some sign pushes fighting the constraints →
  squeezed feasible region → LOW fitness
- With the **uniqueness guarantee**: no local maxima in the fitness landscape
  (every wrong guess has at least one single-flip improvement)

---

## The Two Operations

### Operation 1: Hyperplane Constraint

Given constraint a · x = b (with ~3 nonzero {-1,1} coefficients):
- This defines an equality constraint on the center: a · c = b
- The center must lie on this hyperplane
- This constrains the feasible region for the analytic center

### Operation 2: Two-Plane Push (x_i ∈ {-1,1})

For each variable i, the constraint x_i ∈ {-1,1} means the solution is at
one of two walls. In the continuous relaxation, this adds a barrier at c_i = 0
that pushes c_i AWAY from zero and TOWARD the walls ±1.

Combined with the guess's sign constraint (s_i c_i > 0), the push goes
toward the guessed wall s_i.

**The two-plane push amplifies the fitness signal**: it pushes the center
closer to ±1, making constraint compatibility MORE sensitive to the sign
choice. Wrong guesses create more tension → larger fitness penalty.

---

## Fitness Function

For a guess s ∈ {-1,1}^N, the fitness is the analytic center objective value:

```
FITNESS(s) = max  Σ_i [ log(s_i · c_i) + log(1 - s_i · c_i) ]

             subject to  a_k · c = b_k  for all constraints k
```

### What each term does

- **log(s_i · c_i)**: barrier at c_i = 0, pushing c_i toward guessed wall s_i.
  This is the sign constraint: stay on the guessed side.

- **log(1 - s_i · c_i)**: barrier at c_i = s_i (the wall), preventing the
  center from reaching the wall. This is the box constraint for the
  guessed side.

Together: c_i lives in the interval (0, 1) in the s_i direction. The
barriers keep it away from both endpoints. The constraints shift the
equilibrium.

### Optional: add two-plane amplification

```
FITNESS(s) = max  Σ_i [ log(c_i²) + log(1-c_i²) + log(s_i · c_i) ]

             subject to  a_k · c = b_k
```

The extra log(c_i²) = 2·log|c_i| term pushes c_i further from 0 (toward
walls), amplifying the fitness difference between correct and wrong guesses.

Without it: equilibrium at |c_i| = 1/2. With it: |c_i| ≈ 0.775.

This term comes from the two-plane constraint (x_i ∈ {-1,1}) interpreted
as a barrier at c_i = 0. It's optional but makes the signal stronger.

### Properties

- **Strictly concave** within each sign-orthant (all log terms have negative
  second derivatives) → unique maximum per guess
- **Correct guess has highest fitness** (verified by example, argued by
  constraint compatibility)
- **No local maxima** over {-1,1}^N guesses (with uniqueness guarantee):
  every wrong guess can be improved by flipping at least one variable

---

## Concrete Example (Verified)

Constraint: x₁ + x₂ + x₃ = 1. Valid vertices: (1,1,-1), (1,-1,1), (-1,1,1).

**Correct guess** s = (1,1,-1) [assuming this is x*]:
- Analytic center: c ≈ (0.71, 0.71, -0.42)
- Fitness ≈ -3.15

**Wrong guess** s = (1,1,1):
- Analytic center: c = (1/3, 1/3, 1/3)
- Fitness ≈ -3.65

Correct guess has **higher fitness** (-3.15 > -3.65). ✓

Note: with a single constraint, all three valid vertices tie in fitness.
Additional overlapping constraints break the symmetry in favor of x*.

---

## Algorithm Details

```
Input:  N variables, m constraints {(a_k, b_k)}
Output: x* ∈ {-1,1}^N

1. Initialize guess s (e.g., random, or sign of unconstrained analytic center)

2. Compute FITNESS(s):
   Solve: max Σ [log(s_i·c_i) + log(1 - s_i·c_i)]
          subject to a_k · c = b_k for all k
   Record objective value = fitness

3. For each variable j = 1..N:
   Compute FITNESS(s with s_j flipped)

4. If best flip improves fitness:
   Apply the flip: s_j ← -s_j
   Go to step 2

5. Output s (the guess with highest fitness)

6. Verify: check a_k · s = b_k for all k
```

### Complexity

- Each fitness evaluation: solve a convex optimization with N variables and
  m equality constraints. With sparse constraints: O(N) per solve (using
  Newton's method with sparse linear algebra)
- Step 3: N fitness evaluations per iteration
- Steps 2-4: at most N iterations (each improves at least one variable)
- Total: O(N³) fitness evaluations, each O(N) → O(N⁴) worst case
- Possible speedup: warm-starting (when flipping one variable, the analytic
  center changes locally — incremental update rather than full re-solve)

---

## No Local Maxima Argument

**Claim**: with a unique solution x*, every wrong guess s ≠ x* has at least
one single-variable flip that improves fitness.

**Argument**:
- A wrong guess s differs from x* in at least one variable j (s_j ≠ x*_j)
- The sign constraint for j (s_j c_j > 0) pushes c_j to the wrong wall
- This conflicts with the hyperplane constraints that pass through x*
- Flipping s_j resolves this conflict → more room in the feasible region
  → higher fitness
- The uniqueness guarantee ensures that the constraint structure doesn't
  create "frustrated" pairs (where correcting one variable breaks another):
  any symmetric constraint creating a mirror image must have additional
  constraints breaking the mirror (otherwise the solution wouldn't be unique)

**Caveat**: this argument is compelling but not a rigorous proof. We explored
potential counterexamples (frustrated variable pairs via constraints like
x₁ + x₂ = 0) and found that the uniqueness guarantee breaks every such
frustration. But a formal proof is still open.

---

## Key Mathematical Results

### The response function (for reference)

For the analytic center with sign constraints + box constraints:
- KKT gives the equilibrium: f'(c_i) = -μ_i where μ_i = Σ_k λ_k a_{ki}
- sign(c_i) = sign(μ_i) always (the constraint forces determine the sign)
- The magnitude |c_i| depends on the specific objective function

### Non-overlapping constraints (closed form)

For a single constraint with k = 3 variables, {-1,1} coefficients, and the
basic two-term fitness:
- c_i = s_i · |b|/(2k) ... (approximate, depends on specific formulation)
- The center is at the centroid of valid vertices in the guessed orthant
- Constraint check and KKT conditions verified algebraically

### Analytic center formula (from earlier exploration)

For the standard analytic center (without sign constraints):
```
c_i = f(μ_i) = μ_i / (1 + √(1 + μ_i²))
```
where μ_i = Σ_k λ_k a_{ki} is the total constraint force on variable i.

For non-overlapping {-1,1} constraints: λ_k = 6b_k/(9 - b_k²), c_i = a_{ki}·b_k/3.

---

## Exploration History

### Approaches explored and lessons learned

1. **Shrink outer ellipsoid** (first attempt)
   - Start with sphere containing all vertices, slice with hyperplanes
   - Two-plane operation is identity for axis-aligned Q → abandoned axis-aligned
   - L + Δ_i > 1 determination criterion: works but may not cascade fully
   - Full Q matrix: O(N²) storage, not scalable

2. **Analytic center / max inscribed ellipsoid**
   - Maximize Σ log(1-c_i²) subject to Ac = b
   - Pushes center toward cube CENTER (away from walls) — wrong direction
   - Clean formula: c_i = μ_i/(1+√(1+μ_i²))
   - Sign extraction unreliable for weakly constrained variables

3. **Choice of objective function**
   - Multiple strictly concave options work (Σ(1-c_i²), Σ√(1-c_i²), Σlog(1-c_i²))
   - Log chosen for sigmoid response and clean inverse
   - Σ(1-c_i²) gives linear response (no saturation) — NOT "push to corners" as
     initially (incorrectly) claimed
   - Any symmetric objective gives the same SIGNS, different magnitudes

4. **Expand to -x* (surface-preserving)**
   - Keep x* on ellipsoid surface, expand until center reaches -x*
   - DISPROVEN: operations using constraint hyperplanes (without knowing x*)
     push center toward x*, not toward -x*
   - The constraints define x*, so operations based on them point toward x*

5. **Guess-and-flip with fitness** (CURRENT APPROACH)
   - Guess provides direction (sign constraints)
   - Fitness = analytic center objective with sign + box barriers
   - Correct guess → highest fitness (cooperating forces)
   - No local maxima with uniqueness guarantee
   - Two-plane push amplifies signal (optional log(c_i²) term)

### Key insights from exploration

- **Continuous relaxation limitations**: any strictly concave objective gives
  an interior point, never a vertex. Must extract discrete info (signs) from
  continuous optimization.

- **Sign correctness**: sign(c_i) is determined by constraint forces μ_i,
  not by the objective function choice. Different objectives give different
  magnitudes but same signs.

- **Uniqueness breaks frustration**: with a guaranteed unique solution,
  constraint structures that create "locked" variable pairs always have
  additional constraints that break the lock. This prevents local maxima
  in the fitness landscape.

- **Two-plane amplification**: adding log(c_i²) to the objective pushes
  |c_i| closer to 1 (from 0.5 to ~0.775), amplifying fitness differences
  between correct and wrong guesses.

- **Operations push toward x***: blind (guess-free) operations using
  constraint hyperplanes naturally push the center toward x* (the unique
  satisfying vertex). This is because the constraint centroids converge to x*.

---

## Critical Invariants and Insights

### Invariant: x* on the ellipsoid surface

Throughout the shrink-based explorations, we established that x* remains on
the ellipsoid's surface after each operation:
- **Hyperplane slice**: x* satisfies a·x* = b, so it stays on the surface
- **Two-plane operation**: x* has x*_i ∈ {-1,1}, so it stays on the surface
- **Initially**: all 2^N vertices are on the sphere of radius √N (cost = 1)
This invariant holds regardless of which algorithm design we use.

### Invariant: correct guess has maximum fitness

For the fitness function Σ[log(s_i c_i) + log(1 - s_i c_i)] subject to Ac = b:
- The correct guess s = x* has all sign pushes cooperating with constraints
- The feasible region {c : s_i c_i > 0, s_i c_i < 1, Ac = b} is most spacious
  for s = x* (least internal conflict)
- Verified numerically: correct guess fitness -3.15 > wrong guess -3.65

### Insight: signs come from constraints, not the objective

For ANY symmetric strictly concave objective:
- The KKT equilibrium has sign(c_i) = sign(μ_i)
- μ_i = Σ_k λ_k a_{ki} (constraint forces)
- Different objectives change λ values (and hence μ_i), but for
  non-overlapping constraints with {-1,1} coefficients: c_i = a_i · b/k
  regardless of the specific objective

### Insight: two-plane push amplifies but doesn't create signal

The two-plane operation (log(c_i²) barrier) pushes |c_i| from ~0.5 to ~0.775,
making fitness differences between correct and wrong guesses LARGER. But it
doesn't change which guess has the highest fitness — that's determined by the
constraint structure. The two-plane is an amplifier, not a signal source.

### Insight: uniqueness guarantee prevents local maxima

Potential local maxima arise from "frustrated" variable pairs — e.g., a
constraint x_i + x_j = 0 creates a mirror symmetry where correcting one
variable individually makes the system infeasible. But if such a mirror
exists, the solution isn't unique (both mirror images satisfy the constraint).
The uniqueness guarantee implies additional constraints that break every
such mirror, providing escape routes from every wrong guess.

### Insight: blind operations push toward x*, not -x*

Operations using constraint hyperplanes WITHOUT a guess (no sign constraints)
naturally push the center toward x*:
- Each constraint's centroid has positive dot product with all satisfying vertices
- With uniqueness, centroids converge to x*
- This means the "expand to -x*" design fails (operations go the wrong way)
- But the guess-and-flip design works because the GUESS breaks the symmetry

### Insight: N/3 constraints + N two-plane = 4N/3 total constraints

We have MORE constraints than unknowns:
- N/3 hyperplane constraints (each relating 3 variables)
- N two-plane constraints (x_i ∈ {-1,1}, one per variable)
- Total: 4N/3 > N
The challenge is that the two-plane constraints are non-convex (disjunctive),
making them hard to use in continuous optimization. The guess-and-flip
approach handles this by converting two-plane constraints into sign
constraints (convex) via the guess.

### Insight: fitness function derivation is principled

Each term comes from a real inequality constraint:
- c_i ≤ 1 (or c_i ≥ -1) → log(1 - s_i c_i) barrier at the wall
- s_i c_i ≥ 0 → log(s_i c_i) barrier at zero
These are the standard log barriers for the analytic center of the polytope
{c : s_i c_i ∈ (0,1), Ac = b}. The two-plane amplification (log(c_i²))
is optional/extra — it strengthens the signal but isn't from a constraint.

### Insight: information-theoretic constraints on m

With {-1,1} coefficients and 3 variables per constraint:
- Each constraint with |b| = 1 keeps 3 of 8 patterns → ~1.6 bits
- Each constraint with |b| = 3 keeps 1 of 8 patterns → 3 bits
- Need N bits total to determine all N variables
- With m = N/3 all-|b|=1 constraints: ~N/2 bits. NOT enough for uniqueness!
- Uniqueness with m = N/3 requires some |b| = 3 constraints (or overlapping
  structure that effectively provides more bits through cascading)
- Non-overlapping constraints with all |b| = 1 give 3^(N/3) solutions

This means the "unique solution with N/3 constraints" assumption requires
specific constraint structure — not all instances are solvable with N/3.

### Open insight: is simple ||As - b||² enough?

The discrete residual ||As - b||² (just checking how many constraints the
guess satisfies) might work as well as the continuous fitness. The continuous
approach provides a smooth landscape (no local maxima), while the discrete
approach is O(N) per evaluation (much faster). Whether the discrete landscape
also lacks local maxima for our problem class is unknown.

---

## Open Questions for Next Session

1. **Rigorous no-local-maxima proof**: formalize the argument that uniqueness
   guarantee prevents frustrated variable pairs. Or find a counterexample.

2. **Warm-starting**: when flipping one variable in the guess, the analytic
   center changes locally. Can we update incrementally (O(k) instead of O(N)
   per flip) using the sparse constraint structure?

3. **Comparison with simpler approaches**: the discrete check ||As - b||²
   might work just as well for finding x*. Does the continuous ellipsoid
   fitness provide any advantage over the simple residual check?

4. **Prototype**: implement the algorithm for small N (e.g., N = 30-100),
   generate random instances with unique solutions, and verify:
   - Does the correct guess always have the highest fitness?
   - Does hill-climbing always converge to x*?
   - How many iterations (flips) are needed?
   - How does fitness gap depend on constraint density and overlap?

5. **Scalability**: for N = 10⁶, the O(N³) or O(N⁴) cost might be too high.
   Can the sparse constraint structure be exploited for faster convergence?
   Message-passing / belief propagation on the factor graph?

6. **Connection to belief propagation**: the constraint force μ_i = Σ λ_k a_{ki}
   resembles message accumulation in BP. Is the analytic center equivalent
   to a specific form of BP? Could BP replace the convex optimization?

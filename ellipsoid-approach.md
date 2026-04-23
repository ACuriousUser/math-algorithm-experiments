# Hypercube Vertex Finder: Maximum Inscribed Ellipsoid

## Problem Recap

Given N variables, each in {-1, 1}, and m ≈ N/3 sparse hyperplane constraints
(each involving ~3 variables with {-1,1} coefficients), find the unique vertex
of {-1,1}^N satisfying all constraints.

---

## Approach: Analytic Center of the Feasible Polytope

### Core Idea

Find the center of the largest axis-aligned ellipsoid that fits inside the
hypercube [-1,1]^N, with the center constrained to lie on all hyperplanes.

This naturally balances two forces:
- **Hyperplane constraints** push the center toward the solution vertex
- **Cube walls** (the "two planes" x_i = ±1 for each variable) resist, keeping
  the center interior

As constraints accumulate, the center is pushed closer to ±1 for each variable,
eventually revealing the solution.

### Why Maximize (Not Minimize)

**Minimizing** (shrinking an outer ellipsoid): the "two-planes reshape" operation
necessarily INCREASES the ellipsoid size (proven algebraically — moving the center
for any variable forces the radius to grow to maintain containment). This means
we lack the tools to collapse the ellipsoid.

**Maximizing** (growing an inner ellipsoid): both forces are productive. The
hyperplane constraints inject information, and the cube walls provide resistance
that shapes the solution. Growing is the objective, so both operations help.

### No Local Maxima

The objective is strictly concave on the open cube (-1,1)^N.
The constraints a_k · c = b_k define an affine subspace (convex set).

**Maximizing a strictly concave function over a convex set has exactly one
maximum.** No local maxima, no initialization sensitivity, no restarts needed.

---

## Choice of Objective Function

We need a function that: (1) is strictly concave, (2) pushes c away from the
cube walls, and (3) gives clean formulas. Several options work:

### Options analyzed

| Objective | Strictly concave? | KKT gives c_i = | Notes |
|-----------|-------------------|------------------|-------|
| Σ (1-c_i²) | Yes (Hessian = -2I) | μ_i/2 (linear) | Minimum-norm / least-squares. Clean but c_i response is linear — no "wall resistance" that intensifies near ±1 |
| Σ √(1-c_i²) | Yes | -μ_i/√(1+μ_i²) | Works but less standard, no clean inverse |
| **Σ log(1-c_i²)** | **Yes** | **μ_i/(1+√(1+μ_i²))** | **Standard analytic center. Sigmoid response — resistance grows as c_i → ±1. Clean closed-form inverse.** |
| Σ (1-\|c_i\|) | Concave, NOT strict | sign only | Linear program. Non-unique solutions. No magnitude info. |
| Σ log(1-\|c_i\|) | Yes (but non-smooth at 0) | piecewise | Cusp at c_i=0 (non-differentiable). Promotes sparsity (pushes c_i toward 0), which is counterproductive for vertex-finding. |

### Why log(1-c_i²)?

**NOT because other functions fail to be concave.** Several functions are strictly
concave and give unique maxima. The log is chosen because:

1. **Sigmoid response**: The formula c_i = f(μ_i) = μ_i/(1+√(1+μ_i²)) is a sigmoid.
   For small forces, c_i responds linearly (f'(0) = 1/2). For large forces, c_i
   saturates toward ±1. This matches our intuition: weakly constrained variables
   stay near 0, strongly constrained ones get pushed to the walls.

   Compare to Σ(1-c_i²), where c_i = μ_i/2 is LINEAR — there's no saturation.
   A large enough force pushes c_i past ±1, which is meaningless for our box.
   The log-barrier naturally prevents this.

2. **Smooth everywhere**: Unlike Σlog(1-|c_i|) (cusp at 0) or Σ(1-|c_i|)
   (non-differentiable at 0), log(1-c_i²) is C^∞ on (-1,1). Gradient-based
   methods work without subgradients.

3. **Standard and well-studied**: This is the standard analytic center from
   interior-point methods. Theory and algorithms are well-developed.

4. **Clean inverse**: Both f(μ) and its inverse g(c) = 2c/(1-c²) have simple
   algebraic forms. This enables closed-form solutions for non-overlapping
   constraints.

### What log(1-c_i²) actually is

```
log(1-c_i²) = log((1-c_i)(1+c_i)) = log(1-c_i) + log(1+c_i)
```

It's the sum of log-distances to BOTH walls of the box for variable i.
The c_i² is not special — it arises because (1-c_i)(1+c_i) = 1-c_i².
We're penalizing proximity to the wall at +1 (via log(1-c_i)) AND the wall
at -1 (via log(1+c_i)) simultaneously.

### Correction: Σ(1-c_i²) does NOT "push to corners"

An earlier version of this document incorrectly claimed that Σ(1-c_i²) pushes
all weight to a corner. This is FALSE. Maximizing Σ(1-c_i²) = minimizing Σc_i²,
which gives the MINIMUM-NORM solution — the most spread-out point, not a corner.

For c₁+c₂+c₃ = 1: both Σ(1-c_i²) and Σlog(1-c_i²) give c = (1/3, 1/3, 1/3).
The difference between them only matters for the SHAPE of the response
(linear vs sigmoid), not the direction.

---

## The Optimization Problem

```
maximize    Σ_i log(1 - c_i²)
subject to  a_k · c = b_k     for k = 1, ..., m
            c ∈ (-1, 1)^N
```

---

## KKT Conditions and the Core Formula

At the optimum, the gradient of the objective equals a linear combination
of the constraint gradients:

```
-2c_i/(1-c_i²) + Σ_k λ_k a_{ki} = 0
```

Define the **total force** on variable i:
```
μ_i = Σ_k λ_k a_{ki}
```

Then the equilibrium condition is:
```
2c_i / (1 - c_i²) = μ_i
```

Left side: **barrier resistance** — how hard the cube walls push back.
  This grows without bound as |c_i| → 1. The walls fight harder the
  closer you get.
Right side: **constraint force** — the accumulated pull from all hyperplane
  constraints involving variable i.

### The Response Function f(μ)

Solving 2c/(1-c²) = μ for c ∈ (-1, 1):

```
μc² + 2c - μ = 0
c = (-1 ± √(1+μ²)) / μ
```

Taking the root in (-1, 1):

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   c_i = f(μ_i) = μ_i / (1 + √(1 + μ_i²))      │
│                                                 │
│   r_i = 1 - |c_i|                               │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Verification**: Let S = 1 + √(1+μ²), so c = μ/S.
  1-c² = (S²-μ²)/S² = 2S/S² = 2/S.
  Then 2c/(1-c²) = 2(μ/S)/(2/S) = μ.  ✓

**Second-order verification (this IS a maximum)**:
  The Hessian of log(1-c_i²) is -2(1+c_i²)/(1-c_i²)² < 0 everywhere.
  The objective's Hessian is negative definite on ALL of R^N, hence on any
  subspace (including the constraint null space). The critical point is a
  strict global maximum.  ✓

**Properties of f**:
- f(0) = 0: no force → center of cube
- f(μ) → ±1 as μ → ±∞: strong force → pushed to wall
- f is odd: f(-μ) = -f(μ)
- f is strictly increasing
- f'(0) = 1/2: linear response for weak forces
- Equivalent form: f(μ) = tanh(asinh(μ)/2)

### The Inverse: g(c)

Given a center position, the required force is:
```
μ_i = g(c_i) = 2c_i / (1 - c_i²)
```

g is strictly increasing on (-1,1), with g(0) = 0, g(±1) = ±∞.

---

## Function 1: Constraint Incorporation

**What λ_k is**: The Lagrange multiplier for constraint k. It measures the
"tension" — how much force the constraint must exert to keep the center on the
hyperplane, against the barrier's pull toward c = 0.

- λ_k = 0: the center already satisfies the constraint (no force needed)
- |λ_k| small: the constraint barely moves the center
- |λ_k| → ∞: the constraint forces variables to the wall (full determination)

λ_k is NOT about moving toward or away from the hyperplane. The center always
lies EXACTLY on the hyperplane (it's an equality constraint). λ_k measures
the force required to maintain that.

### Non-overlapping case (each involved variable in only this constraint)

For {-1,1} coefficients with k involved variables:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   λ = 2bk / (k² - b²)                                  │
│                                                         │
│   c_i = a_i · b/k          (for involved variables)     │
│   c_i = 0                  (for uninvolved variables)   │
│   r_i = 1 - |b|/k          (for involved variables)     │
│                                                         │
│   For k = 3:  λ = 6b / (9 - b²)                        │
│               c_i = a_i · b/3                           │
│               r_i = 1 - |b|/3                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Derivation**: Each involved variable has μ_i = λ a_i. Since |a_i| = 1,
the constraint becomes k · f(λ) = b, so f(λ) = b/k, so λ = g(b/k).
Then c_i = f(λ a_i) = a_i · f(λ) = a_i · b/k.

**Verified**:
- Constraint: Σ a_i c_i = Σ a_i² · b/k = k · b/k = b.  ✓
- KKT: 2c_i/(1-c_i²) = 2(a_i b/k)/(1-(b/k)²) = a_i · 2bk/(k²-b²) = λ a_i.  ✓
- Second-order: negative definite Hessian confirms maximum.  ✓
- Requires |b| < k (otherwise constraint infeasible in the open cube).

**Key result**: the analytic center equals the CENTROID of valid vertices.
For a single constraint with {-1,1} coefficients, the center (1/3, 1/3, 1/3)
for b=1 is exactly the average of the three valid vertices
(1,1,-1), (1,-1,1), (-1,1,1). This is because the log-barrier is symmetric
around each variable.

### Overlapping case (variable in multiple constraints)

Variable i has μ_i = Σ_k λ_k a_{ki}. The system is m nonlinear equations
in m unknowns:

```
For each constraint k:  Σ_i a_{ki} · f(Σ_j λ_j a_{ji}) = b_k
```

Solve by Newton's method. The Jacobian is sparse (each equation involves ~3
variables, each in ~1-2 constraints). Convergence is guaranteed (strictly
concave objective).

### General real coefficients

For general a_i, the constraint equation Σ a_i f(λ a_i) = b no longer simplifies
to a uniform formula. It's still one equation in one unknown (λ), solvable by
Newton. The function h(λ) = Σ λa_i²/(1+√(1+λ²a_i²)) is strictly increasing
with range (-Σ|a_i|, +Σ|a_i|).

---

## Function 2: Center-to-Radius Map (Two-Planes Response)

```
┌─────────────────────────────────────────────┐
│                                             │
│   r_i = 1 - |c_i|                           │
│                                             │
│   = 1 - |μ_i| / (1 + √(1 + μ_i²))          │
│                                             │
└─────────────────────────────────────────────┘
```

This is the maximum radius for variable i such that the ellipsoid stays
inside [-1,1]^N. It measures how "determined" variable i is:

- r_i = 1 (c_i = 0): completely undetermined
- r_i = 2/3 (|c_i| = 1/3): one b=±1 constraint
- r_i → 0 (|c_i| → 1): effectively determined, x_i = sign(c_i)

This function IS the "two planes" insight: the cube walls at c_i = ±1
limit how far the ellipsoid can grow. The barrier log(1-c_i²) enforces
this smoothly — it goes to -∞ as c_i approaches either wall, which is
exactly the force that balances the constraint pull.

---

## Sign Correctness Analysis

### The extraction step

The analytic center gives c_i ∈ (-1,1). To get the vertex, we take
x_i = sign(c_i). Is this always correct?

### For a single {-1,1} constraint: signs can be wrong

With constraint x₁ + x₂ + x₃ = 1, the analytic center is (1/3, 1/3, 1/3).
Rounding: (1, 1, 1). Check: 1+1+1 = 3 ≠ 1.  ✗

This is expected: the centroid of {(1,1,-1), (1,-1,1), (-1,1,1)} has all
positive signs, but every valid vertex has one negative. A single constraint
with 3 valid vertices can't determine which variable is negative.

### For general coefficients: counterexample exists

With non-{-1,1} coefficients, the analytic center can have wrong signs even
with a UNIQUE valid vertex. Example: constraint c₁ + c₂ - 100c₃ = -98.

Unique vertex: (1, 1, 1) [check: 1+1-100 = -98 ✓].

At the analytic center: c₃ is pulled strongly toward +1 (large coefficient),
consuming almost all the constraint budget. The barrier pushes c₁, c₂ toward 0,
and the small residual from the constraint pushes them slightly NEGATIVE.
So sign(c₁) = sign(c₂) = -1 ≠ +1 = x*₁ = x*₂.

**This counterexample uses a coefficient of -100, violating our {-1,1}
coefficient restriction.** For {-1,1} coefficients, all variables in a
constraint are treated equally (the |a_i²| = 1 symmetry), which avoids
this pathology.

### For {-1,1} coefficients with all constraints: open question

With N/3 overlapping constraints (all with {-1,1} coefficients) and a
guaranteed unique vertex, does sign(c_i) = x*_i for all i?

Arguments FOR:
- Each constraint treats its variables symmetrically (|a_i| = 1)
- The accumulated force μ_i = Σ_k λ_k a_{ki} reflects ALL constraints on variable i
- The unique vertex x* is the only point in {-1,1}^N ∩ {Ac = b}, so constraints
  must collectively push each c_i toward x*_i

Arguments AGAINST (potential concern):
- Even with |a_i| = 1, the Lagrange multipliers λ_k can vary widely
- μ_i = Σ_k λ_k a_{ki} is a signed sum that COULD have the wrong sign
- No proof exists yet

**This is the key question for prototyping**: generate random instances with
{-1,1} coefficients and a unique solution, compute the analytic center, and
check whether sign(c) = x* always holds.

---

## Complete Algorithm

```
Input: N variables, m constraints {(a_k, b_k)} with a_{ki} ∈ {-1, 0, 1}
Output: x* ∈ {-1,1}^N

1. Initialize:
   μ_i = 0, c_i = 0, r_i = 1  for all i
   λ_k = 0                     for all k

2. For each constraint k:
   If non-overlapping:
       λ_k = 6 b_k / (9 - b_k²)
       For involved variable i:
           μ_i += λ_k · a_{ki}
           c_i = f(μ_i) = μ_i / (1 + √(1 + μ_i²))
           r_i = 1 - |c_i|
   If overlapping:
       Newton solve for λ_k (and re-adjust neighboring λ's)
       Update μ_i, c_i, r_i for affected variables

3. Extract: x_i = sign(c_i) for all i.

4. Verify: check a_k · x = b_k for all k.
```

### Complexity

| Case | Per constraint | Total |
|------|---------------|-------|
| Non-overlapping | O(k) ≈ O(1) | O(N) |
| Overlapping (sparse) | O(k · Newton iters) | O(N · iters) |

---

## Worked Examples

### Example 1: b = 3 (fully determining)

Constraint: x₁ + x₂ + x₃ = 3. Only valid vertex: (1, 1, 1).

λ = 6·3/(9-9) → ∞. c_i = f(∞) = 1. r_i = 0.

All three variables immediately determined. ✓

### Example 2: b = 1 (partial information)

Constraint: x₁ + x₂ + x₃ = 1.
Valid vertices: (1,1,-1), (1,-1,1), (-1,1,1).

λ = 3/4. c = (1/3, 1/3, 1/3) = centroid of valid vertices. r_i = 2/3.

sign(c) = (1,1,1) but 1+1+1 ≠ 1. Single constraint insufficient.

### Example 3: b = -1 with mixed signs

Constraint: x₁ + x₂ - x₃ = -1.
Valid vertices: (-1,-1,-1), (-1,1,1), (1,-1,1).

λ = -3/4. c = (-1/3, -1/3, 1/3) = centroid of valid vertices. ✓

### Example 4: Two overlapping constraints

C1: x₁ + x₂ + x₃ = 1.  C2: x₃ + x₄ + x₅ = -1.  (x₃ shared)

Variables and forces:
  μ₁ = λ₁,  μ₂ = λ₁,  μ₃ = λ₁ + λ₂,  μ₄ = λ₂,  μ₅ = λ₂

System:
  2f(λ₁) + f(λ₁+λ₂) = 1       ... (I)
  f(λ₁+λ₂) + 2f(λ₂) = -1      ... (II)

Subtracting: f(λ₁) - f(λ₂) = 1. Since f maps R → (-1,1), this is
feasible and requires λ₁ >> 0, λ₂ << 0. Solve numerically.

---

## Properties

### Guaranteed uniqueness (no local maxima)

Σ log(1-c_i²) is strictly concave. The feasible set is convex. Therefore:
- The analytic center is UNIQUE
- Any optimization method converges to it
- No local maxima, saddle points, or traps
- Each intermediate step (adding constraints one at a time) also has a
  unique solution

### What λ_k represents physically

λ_k is the tension in constraint k. The center always lies ON the hyperplane
(the constraint is always exactly satisfied). λ_k measures the force needed
to keep it there against the barrier's pull toward c = 0.

The direction of force is along the constraint normal a_k. Whether λ_k is
positive or negative depends on which side of the hyperplane the origin falls:
- If a_k · 0 < b_k: the origin violates the constraint, so λ_k pushes
  the center toward the b_k side
- If a_k · 0 > b_k: λ_k pushes the other way
- If b_k = 0: λ_k = 0 (origin already on the hyperplane)

---

## Open Questions

1. **Sign correctness for {-1,1} constraints**: With all N/3 constraints
   present and a unique solution guaranteed, does sign(c_i) = x*_i always?
   No counterexample found for {-1,1} coefficients, but no proof either.
   **Priority for prototyping.**

2. **Overlapping constraint solver**: Most efficient Newton approach for the
   coupled λ system? Message-passing on the constraint graph?

3. **Connection to belief propagation**: μ_i = Σ_k λ_k a_{ki} resembles
   message accumulation in BP. Is the analytic center equivalent to a
   specific form of BP?

4. **Scaling to N = 10⁶**: Newton on the sparse system should be O(N) per
   iteration. How many iterations are needed?

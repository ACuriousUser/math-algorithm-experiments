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

The objective Σ log(1-c_i²) is **strictly concave** on the open cube (-1,1)^N.
The constraints a_k · c = b_k define an affine subspace (convex set).

**Maximizing a strictly concave function over a convex set has exactly one
maximum.** No local maxima, no initialization sensitivity, no restarts needed.

---

## The Optimization Problem

```
maximize    Σ_i log(1 - c_i²)
subject to  a_k · c = b_k     for k = 1, ..., m
            c ∈ (-1, 1)^N
```

**What this maximizes**: Σ log(1-c_i²) = Σ [log(1-c_i) + log(1+c_i)] is the
sum of log-distances to all 2N faces of the cube. This is the standard
analytic center objective — it penalizes being close to ANY face.

**Connection to ellipsoid volume**: For axis-aligned ellipsoid inside [-1,1]^N
centered at c, the max radius per variable is r_i = 1 - |c_i|. The analytic
center objective is related to (but not identical to) maximizing Π r_i.
Both push the center toward the interior, and both are strictly concave.

---

## KKT Conditions and the Core Formula

At the optimum, the gradient of the objective must be a linear combination
of the constraint gradients:

```
∂/∂c_i [ Σ log(1-c_j²) ] + Σ_k λ_k · ∂/∂c_i [ a_k·c - b_k ] = 0

-2c_i/(1-c_i²) + Σ_k λ_k a_{ki} = 0
```

Define the **total force** on variable i:
```
μ_i = Σ_k λ_k a_{ki}
```

Then the KKT condition is:
```
2c_i / (1 - c_i²) = μ_i
```

Left side: the **barrier resistance** — how hard the cube walls push back.
Right side: the **constraint force** — how hard the hyperplanes pull.

### The Response Function f(μ)

Solving 2c/(1-c²) = μ for c ∈ (-1, 1):

```
μ(1-c²) = 2c
μc² + 2c - μ = 0
c = (-2 ± √(4 + 4μ²)) / (2μ) = (-1 ± √(1+μ²)) / μ
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
  1-c² = 1 - μ²/S² = (S²-μ²)/S².
  S² = 1 + 2√(1+μ²) + 1+μ² = 2 + μ² + 2√(1+μ²).
  S²-μ² = 2 + 2√(1+μ²) = 2(1+√(1+μ²)) = 2S.
  So 1-c² = 2S/S² = 2/S.
  Then 2c/(1-c²) = 2(μ/S)/(2/S) = μ.  ✓

**Properties of f**:
- f(0) = 0: no force → center of cube
- f(μ) → +1 as μ → +∞: strong positive force → pushed to wall c = +1
- f(μ) → -1 as μ → -∞: strong negative force → pushed to wall c = -1
- f is odd: f(-μ) = -f(μ)
- f is strictly increasing: more force → further from center
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

Given a new constraint a · c = b, compute the Lagrange multiplier λ and
update the center.

### The constraint equation

The constraint a · c = b, with c_i = f(μ_i) and μ_i = Σ_k λ_k a_{ki},
becomes:

```
Σ_i a_i · f(μ_i) = b
```

This is a nonlinear equation in the Lagrange multipliers λ.

### Non-overlapping case (each involved variable in only this constraint)

For variable i involved only in this constraint: μ_i = λ a_i.
So c_i = f(λ a_i).

The constraint becomes:
```
Σ_i a_i · f(λ a_i) = b
```

**For {-1,1} coefficients** (|a_i| = 1 for all involved i):

Each term: a_i · f(λ a_i) = a_i · (λ a_i)/(1+√(1+λ²a_i²)) = λ/(1+√(1+λ²)) = f(λ).

(This works because a_i² = 1, so the a_i factors cancel in the product.)

With k involved variables (k ≈ 3):
```
k · f(λ) = b
f(λ) = b/k
```

Applying the inverse g = f⁻¹:
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   λ = g(b/k) = 2(b/k) / (1 - (b/k)²) = 2bk / (k²-b²) │
│                                                         │
│   For k = 3:  λ = 6b / (9 - b²)                        │
│                                                         │
│   Then:  c_i = a_i · b/k                                │
│          r_i = 1 - |b|/k                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Verification of c_i = a_i · b/k** (for k = 3, {-1,1} coefficients):

  Constraint check: Σ a_i c_i = Σ a_i · a_i · b/3 = Σ a_i² · b/3 = 3 · b/3 = b.  ✓

  KKT check: 2c_i/(1-c_i²) should equal λ a_i.
  LHS = 2(a_i b/3)/(1-(b/3)²) = (2a_i b/3)/((9-b²)/9) = 6a_i b/(9-b²).
  RHS = λ a_i = 6b/(9-b²) · a_i.  ✓

**Specific values for k = 3**:

| b   | λ     | c_i (for a_i=+1) | r_i  | Status              |
|-----|-------|-------------------|------|---------------------|
| 0   | 0     | 0                 | 1    | No information      |
| ±1  | ±3/4  | ±1/3              | 2/3  | Partial information |
| ±3  | ±∞    | ±1                | 0    | Fully determined    |

### Overlapping case (variable appears in multiple constraints)

Variable i appears in constraints k₁, k₂, ...: μ_i = Σ_j λ_{k_j} a_{k_j,i}.

The constraint equations form a coupled system:
```
For each constraint k:  Σ_i a_{ki} · f(Σ_j λ_j a_{ji}) = b_k
```

This is m equations in m unknowns (λ₁, ..., λ_m).

**Incremental solution**: When adding constraint m+1 to an existing solution:

1. Solve for λ_{m+1}: one nonlinear equation in one unknown.
   Use Newton's method (converges in ~3 iterations).

2. If shared variables exist: their c_i changed, potentially violating
   previous constraints. Re-solve the affected λ values.
   For sparse graphs: the cascade is local.

3. Newton on the full system: the Jacobian is sparse (each equation involves
   ~3 variables, each variable in ~1-2 constraints). Each Newton step is O(N)
   with sparse linear algebra.

**Convergence**: guaranteed (strictly concave problem, convex feasible set).

### General real coefficients

For general a_i (not just {-1,1}), the constraint equation k·f(λ) = b
no longer simplifies to a uniform formula. Instead:

```
Σ_i a_i · f(λ a_i) = b    →    Σ_i λ a_i² / (1+√(1+λ²a_i²)) = b
```

This is still one equation in one unknown (λ), solvable by Newton.
The function h(λ) = Σ λa_i²/(1+√(1+λ²a_i²)) is strictly increasing
with range (-Σ|a_i|, +Σ|a_i|), so a unique solution exists for |b| < Σ|a_i|.

---

## Function 2: Center-to-Radius Map (Two-Planes Response)

This is the simpler of the two functions, but conceptually important.

### The formula

```
┌─────────────────────────────────────────────┐
│                                             │
│   r_i = 1 - |c_i|                           │
│                                             │
│   = 1 - |f(μ_i)|                            │
│                                             │
│   = 1 - |μ_i| / (1 + √(1 + μ_i²))          │
│                                             │
└─────────────────────────────────────────────┘
```

### What this represents

The maximum radius for variable i such that the ellipsoid stays inside [-1,1]^N.
If c_i = 0.3, the ellipsoid extends from 0.3 - 0.7 = -0.4 to 0.3 + 0.7 = 1.0.
The wall at +1 is the binding constraint.

### Role in the algorithm

r_i measures how "determined" variable i is:
- r_i = 1 (c_i = 0): completely undetermined
- r_i = 2/3 (|c_i| = 1/3): weakly constrained (one b=±1 constraint)
- r_i → 0 (|c_i| → 1): effectively determined, x_i = sign(c_i)

### Variable determination

When r_i < ε (for some tolerance), declare x_i = sign(c_i).

More precisely: if |c_i| > 1 - ε, the barrier log(1-c_i²) → -∞ ensures
the optimization is pushing c_i toward ±1. This happens when the constraints
leave no room for c_i on the other side.

---

## Complete Algorithm

```
Input: N variables, m constraints {(a_k, b_k)}
Output: x* ∈ {-1,1}^N

1. Initialize:
   μ_i = 0     for all i    (no constraint forces yet)
   c_i = 0     for all i    (center of cube)
   r_i = 1     for all i    (maximum radius)
   λ_k = 0     for all k    (no multipliers yet)

2. For each constraint k = 1, ..., m:

   If non-overlapping (involved vars have no previous constraints):
       λ_k = 6 b_k / (9 - b_k²)                   // O(1)
       For each involved variable i:
           μ_i = λ_k · a_{ki}                       // O(1)
           c_i = f(μ_i)                              // O(1)
           r_i = 1 - |c_i|                           // O(1)

   If overlapping (some involved vars have previous constraints):
       Solve for λ_k and re-adjust neighboring λ's   // Newton, O(k·iters)
       Update μ_i, c_i, r_i for affected variables

3. Output x_i = sign(c_i) for all i.

4. Verify: check a_k · x = b_k for all k.
   If verification fails, the sign was wrong for some variables —
   need more sophisticated extraction (see Open Questions).
```

### Complexity

| Step | Non-overlapping | Overlapping (sparse) |
|------|-----------------|---------------------|
| Per constraint | O(k) ≈ O(1) | O(k · Newton iters) |
| Total | O(N) | O(N · iters) |

---

## Worked Examples

### Example 1: b = 3 (fully determining)

Constraint: x₁ + x₂ + x₃ = 3. Only valid vertex: (1, 1, 1).

λ = 6·3/(9-9) = ∞. c_i = f(∞) = 1. r_i = 0.

All three variables immediately determined. ✓

### Example 2: b = 1 (partial information)

Constraint: x₁ + x₂ + x₃ = 1.
Valid vertices: (1,1,-1), (1,-1,1), (-1,1,1).

λ = 6·1/(9-1) = 3/4.

c₁ = f(3/4) = (3/4)/(1+√(1+9/16)) = (3/4)/(1+5/4) = (3/4)/(9/4) = 1/3.
c₂ = 1/3, c₃ = 1/3. r_i = 2/3 for all three.

sign(c) = (+,+,+) → (1,1,1). But 1+1+1 = 3 ≠ 1. ✗

The analytic center points to the CENTROID of the valid vertices,
which has the wrong sign for one variable. Additional constraints
involving these variables are needed to break the symmetry.

### Example 3: b = -1 with mixed coefficients

Constraint: x₁ + x₂ - x₃ = -1.
Valid vertices: (-1,-1,-1), (-1,1,1), (1,-1,1).

λ = 6·(-1)/(9-1) = -3/4.

c₁ = f(-3/4 · 1) = f(-3/4) = -1/3.
c₂ = f(-3/4 · 1) = -1/3.
c₃ = f(-3/4 · (-1)) = f(3/4) = 1/3.

Centroid of valid vertices:
((-1-1+1)/3, (-1+1-1)/3, (-1+1+1)/3) = (-1/3, -1/3, 1/3). ✓

The analytic center equals the centroid of valid vertices (for a single
non-overlapping constraint with {-1,1} coefficients). This makes sense:
the barrier is symmetric, so the center is the average.

### Example 4: Two overlapping constraints

Constraints:
  x₁ + x₂ + x₃ = 1    (C1)
  x₃ + x₄ + x₅ = -1   (C2)

Variable x₃ is shared.

From C1 alone: c₃ = 1/3, μ₃ = 3/4.
From C2 alone: c₃ would be -1/3 (since a₃ = 1 in C2, b = -1).

Combined: μ₃ = λ₁ · 1 + λ₂ · 1. The joint system:

  3f(λ₁) - 2f(λ₁) + ... = ... (this requires careful setup)

Actually, let me be precise. Variables and their constraints:
- x₁: only C1, coefficient +1. μ₁ = λ₁.
- x₂: only C1, coefficient +1. μ₂ = λ₁.
- x₃: C1 (+1) and C2 (+1). μ₃ = λ₁ + λ₂.
- x₄: only C2, coefficient +1. μ₄ = λ₂.
- x₅: only C2, coefficient +1. μ₅ = λ₂.

Constraint equations:
  C1: f(λ₁) + f(λ₁) + f(λ₁+λ₂) = 1
  C2: f(λ₁+λ₂) + f(λ₂) + f(λ₂) = -1

→  2f(λ₁) + f(λ₁+λ₂) = 1       ... (I)
   f(λ₁+λ₂) + 2f(λ₂) = -1      ... (II)

From (I) - (II): 2f(λ₁) - 2f(λ₂) = 2, so f(λ₁) - f(λ₂) = 1.

Since f maps R → (-1,1): f(λ₁) - f(λ₂) = 1 requires f(λ₁) close to 1
and f(λ₂) close to 0, or some other combination.

The max of f(λ₁)-f(λ₂) approaches 2 (when λ₁→∞, λ₂→-∞), so a
difference of 1 is feasible.

This would need to be solved numerically (Newton on 2 equations, 2 unknowns).
The solution would give specific c values that satisfy both constraints
simultaneously.

---

## Properties

### Guaranteed convergence (no local maxima)

Σ log(1-c_i²) is strictly concave on (-1,1)^N. The feasible set
{c : Ac = b, c ∈ (-1,1)^N} is convex. Therefore:

- The analytic center is UNIQUE
- Any optimization method (Newton, gradient ascent) converges to it
- No local maxima, saddle points, or other traps
- The incremental approach (adding constraints one at a time) finds the
  exact analytic center at each intermediate step

### Solution extraction

The analytic center has c_i ∈ (-1,1). The candidate solution is x_i = sign(c_i).

When does sign(c_i) = x*_i (the true solution)?
- For variables fully determined by constraints: |c_i| → 1, sign is correct
- For variables with symmetric constraints: c_i may be near 0, sign unreliable
- With enough overlapping constraints: all signs should be correct

The verification step (check a·x = b for all k) catches any sign errors.

### Relationship to the feasible set

With m = N/3 constraints and N variables, the feasible set
{c ∈ [-1,1]^N : Ac = b} is a polytope of dimension ~2N/3.

The analytic center sits in the "most interior" point of this polytope.
As more constraints are added (or as constraints force variables), the
center moves toward the unique vertex x*.

---

## Open Questions

1. **Sign correctness**: After computing the analytic center with all N/3
   constraints, is sign(c_i) = x*_i for all i? Or do we need a secondary
   method to resolve ambiguous variables?

2. **Overlapping constraint solver**: What's the most efficient Newton solver
   for the coupled λ system? Can message-passing on the constraint graph
   replace generic Newton?

3. **Connection to belief propagation**: μ_i = Σ_k λ_k a_{ki} looks like
   message accumulation in BP. Is the analytic center equivalent to a
   specific form of BP?

4. **General coefficients**: For non-{-1,1} coefficients, the clean formula
   c_i = a_i b/k no longer applies. The per-constraint equation
   Σ a_i f(λa_i) = b still has a unique solution but needs numerical root-finding.

5. **Scaling to N = 10⁶**: Newton on the sparse system should be O(N) per
   iteration. How many iterations are needed? Is the constraint graph
   structure (tree-like? bounded treewidth?) favorable?

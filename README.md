# Math Algorithm Experiments

A collection of math algorithm experiments and explorations.

## Problem 1: Hypercube Vertex from Hyperplane Intersections

### Problem Statement

Given:
- **N** dimensions
- A set of **m** hyperplanes, each defined by a linear equation **a** · **x** = b
- A **guarantee** that exactly one point satisfies all constraints

Find the unique point **x** ∈ {-1, 1}^N (a vertex of the N-dimensional hypercube centered at the origin) that lies on all of the provided hyperplanes.

### Formal Definition

Given a system of linear equations:

```
a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ = b₁
a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ = b₂
...
aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ = bₘ
```

Find **x** = (x₁, x₂, ..., xₙ) such that:
- Each xᵢ ∈ {-1, 1}
- All equations are satisfied simultaneously

### Constraints & Parameters
- The solution is guaranteed to be **unique**
- **m << N** — far fewer equations than variables (heavily underdetermined in the continuous sense)
- **N up to millions**
- **Sparse hyperplanes**: each equation has at most ~3 non-zero coefficients
- **Coefficient restriction** (optional simplification): non-zero coefficients are in {-1, 1}
- Brute force over all 2^N vertices is the baseline to beat

### Key Observations

1. **Underdetermined but finitely constrained**: In continuous R^N, m << N equations
   have infinitely many solutions. But {-1, 1}^N is finite — a small number of sparse
   equations can uniquely pin down a vertex even with m << N.

2. **Sparse structure → graph-based reasoning**: With ~3 variables per equation, the
   problem has a natural bipartite factor graph (equations ↔ variables). This enables
   local message-passing / propagation algorithms.

3. **NOT cleanly XOR-SAT**: With {-1,1} coefficients, substituting y_i = (1-x_i)/2
   gives integer equations like y_i + y_j + y_k = c. Reducing mod 2 to get XOR-SAT
   **loses information** — it preserves parity but discards cardinality. For example
   b=3 (all same sign) and b=-1 (two differ) both map to XOR=0 but have disjoint
   solution sets. With m=N/3 equations, GF(2) gives ~2^(2N/3) spurious solutions.

4. **Parity + cardinality**: Each equation actually constrains both the parity AND
   the count of +1/-1 among its variables. Equations with |b|=3 are especially
   powerful — they immediately determine all 3 variables.

5. **Related problems**:
   - **LDPC decoding** (sparse checks over binary variables, but uses soft information)
   - **Compressed sensing with binary signals** (recover {-1,1}^N from few measurements)
   - **Constraint propagation** (if any equation has only 1 unknown, solve and propagate)
   - **XOR-SAT** (a weaker necessary condition, not sufficient alone)

### Algorithmic Directions to Explore

- **Unit propagation**: If any equation involves only 1 unknown variable, determine it
  immediately, substitute, and repeat. May cascade to solve everything. Equations where
  |b| = 3 immediately fix all 3 variables (even stronger than unit propagation).
- **Belief propagation on factor graphs**: Message-passing over the equation/variable
  bipartite graph. Can exploit both parity and cardinality constraints simultaneously
  via soft probability messages.
- **GF(2) as a filter, not a solver**: Solve the GF(2) system to get a (large) candidate
  space, then use cardinality constraints to prune.
- **LP/convex relaxation**: Relax xᵢ ∈ {-1,1} to xᵢ ∈ [-1,1]. With a unique vertex
  solution, the relaxation might recover it exactly.
- **Greedy / iterative fixing**: Fix the most-constrained variable first, propagate,
  repeat.

### Open Questions

- With m = N/3 sparse equations, what fraction of equations typically have |b| = 3
  (and thus immediately fix 3 variables)? How far does propagation cascade from those?
- Is belief propagation sufficient, or can the factor graph have loops that prevent
  convergence?
- Can the cardinality + parity constraints be combined into a single algebraic framework
  more powerful than GF(2)?

# Guess-and-Flip Algorithm: Assumption Audit and Empirical Findings

**Date**: session on 2026-04-23 / 24
**TL;DR**: The central claim of the `ellipsoid-approach.md` design — that the
correct guess `s = x*` maximises the continuous ellipsoid fitness — is **false**
on concrete small instances we generated. The simpler discrete
`||A s − b||²` hill-climb **strictly dominates** the continuous fitness as a
fitness function on every regime we tested, yet *also* has local optima and so
is not a standalone solver either.

---

## 1. Assumption audit

Claims the algorithm depends on, graded by what we could verify:

| # | Claim | Verdict |
|---|-------|---------|
| C1 | Fitness is strictly concave within each sign-orthant → unique analytic-center maximum per `s`. | **True.** Σ[log u + log(1−u)] has 2nd-deriv −1/u² − 1/(1−u)² < 0 on (0,1). |
| C2 | "Signs come from constraints, not the objective." | Correct for the *guess-free* analytic center; **irrelevant** to the current algorithm, where the log(sᵢcᵢ) barrier fixes sign(cᵢ) = sᵢ by construction. |
| C3 | Correct guess `s = x*` has highest continuous fitness. | **FALSE.** Empirical counterexamples at N = 6, 8, 10 with gaps up to +2.98. See §3. |
| C4 | No Hamming-local maxima other than `x*`, guaranteed by uniqueness. | **Not demonstrable.** Since C3 fails, x* is frequently not even a local max. |
| C5 | Hill-climb terminates in at most N iterations. | **Wrong as stated.** Fitness is strictly increasing, so worst case is 2ᴺ (no revisits); "N iterations" would require each flip to monotonically reduce Hamming distance to x*, which is not implied. |
| C6 | Uniqueness breaks every frustration / mirror symmetry. | Narrowly true; does *not* imply C4. |
| C7 | log(cᵢ²) amplification doesn't change the maximiser. | Untested; pre-empted by failure of C3. |
| C8 | m = N/3 with |b|=1 is information-theoretically insufficient for uniqueness without overlap. | **True.** For N ∈ {6,8,10} we could not find *any* unique-x* instance at m = N/3 with |b|=1 constraints (out of thousands of samples). Smallest m that routinely gives uniqueness is ≈ N/2 to 2N/3. |
| C9 | "x* on ellipsoid surface" invariant. | Inherited from the abandoned shrink-ellipsoid design; **not relevant** to the current algorithm. |
| C10 | Per-fitness-evaluation cost O(N) with sparse constraints. | Plausible (Newton + sparse solve). Not investigated. |

**Upshot:** C3 and C4 are the load-bearing claims. The others are either
mechanical, irrelevant, or derivative. C3 fails empirically, which alone
sinks the algorithm.

---

## 2. Experimental setup

- Instance generator: random 3-variable {−1,+1}-coefficient constraints with
  |b| ∈ {1, 3}, or restricted to |b| = 1 (the algorithmically-interesting
  "overlap required for uniqueness" regime). We *only keep* instances where
  x* is the unique solution in {−1,+1}ᴺ (verified by enumeration).
- For each kept instance, we enumerate all 2ᴺ sign patterns and evaluate:
  - **Discrete residual**: R(s) = ‖A s − b‖², minimise.
  - **Continuous ellipsoid fitness**: analytic-center value of
    {c : sᵢcᵢ ∈ (0,1), A c = b}, maximise. Solved via a custom Newton
    method on the m-dimensional dual (≈ 40 lines, handles each pattern in
    milliseconds; `experiment_hillclimb.py:u_of_g` and
    `continuous_fitness`).
- From the enumerated landscape we measure: (i) is x* the unique global
  optimum? (ii) how many Hamming-1 local optima are there? (iii) does
  greedy best-improvement hill-climb from random starts reach x*?

---

## 3. Findings

### 3a. The continuous fitness does *not* rank x* first

For every regime where we could test:

| N | m | x* feasible for continuous fitness | x* is global max | HC success | fitness gap (best − x*) max |
|---|---|---|---|---|---|
| 6 | 4 | 4/8 | 1/8 | 2/64 | +0.76 |
| 6 | 5 | 0/8 | 0/8 | 0/64 | — |
| 8 | 5 | 3/8 | 0/8 | 0/64 | +2.07 |
| 8 | 6 | 0/8 | 0/8 | 1/64 | — |
| 8 | 7 | 0/8 | 0/8 | 0/64 | — |
| 10 | 6 | 2/8 | 0/8 | 0/64 | +3.09 |
| 10 | 7 | 4/8 | 0/8 | 0/64 | +3.54 |
| 10 | 8 | 0/8 | 0/8 | 0/64 | — |

**Two distinct failure modes:**

**Failure mode A — feasibility collapse.** For sufficiently large m
(e.g., N=6 m≥5, N=8 m≥6, N=10 m≥8), the affine subspace {c : A c = b}
has too small a dimension to cross the open box {|cᵢ|<1} at any sign
pattern. Every pattern has continuous fitness = −∞. The algorithm
produces no signal at all. This includes the trivial |b|=3 constraints:
at s = x*, the constraint becomes u₁+u₂+u₃ = 3 with each uᵢ < 1, which
is infeasible by construction.

**Failure mode B — wrong-ordering.** When x* *is* continuous-feasible,
some *other* sign pattern (typically one that violates the linear
equations) has strictly higher fitness. This is a direct counterexample
to the design claim.

### 3b. Concrete counterexample (N = 6, m = 4)

```
x* = [-1, +1, +1, -1, +1, -1]
A  = [[+1, 0, 0, -1, 0, -1],
      [ 0, 0,+1,  0,-1, -1],
      [+1, 0, 0,  0,+1, -1],
      [-1,-1, 0, -1, 0,  0]]
b  = [1, 1, 1, 1]

(A x* − b) = [0, 0, 0, 0]        ✓ x* satisfies all four constraints exactly.

continuous fitness(x*)            = −10.2186
continuous fitness(s'),
  where s' = [-1,-1,+1,-1,+1,-1]  = −9.3154   ← higher by 0.9032

s' differs from x* in one bit (index 2: flipping x*₂ from +1 to −1).
s' violates the linear system: ‖A s' − b‖² = 4.

Only 3 of 64 sign patterns are continuous-feasible on this instance.
In the top-5 fitness ranking, x* is second, beaten by s'.
```

So hill-climbing the continuous fitness starting from x* *itself* would
flip away from the solution.

### 3c. The discrete residual baseline: partial win

Across 240 instances total, the discrete residual `‖A s − b‖²` always has
x* as the unique global minimum (240/240), as expected. But:

- It has **multiple Hamming-1 local minima** on most instances — mean
  1.6 to 3.5 per instance, max 10.
- Hill-climb from a random start hits x* roughly **15–40 % of the time**
  depending on regime; the rest of the time it gets trapped in a local
  minimum.

So the discrete baseline ("just count unsatisfied constraints") is a
better fitness function than the proposed continuous one, yet is *not*
a standalone solver either. It would need multi-start / randomisation /
a smarter move set (2-bit flips, etc.) to be reliable.

### 3d. Diagnosis: why the continuous fitness fails

The design intuition was "at s = x*, the sign pushes cooperate with the
constraints, so the feasible region is most spacious". The geometric
reality is different:

- The feasible region at guess s is the intersection of the affine
  subspace {A c = b} with the open box {sᵢ cᵢ ∈ (0,1)}.
- This region **can be empty** when the affine subspace simply doesn't
  enter the open orthant in question — and it's empty at x* as often
  as (or more often than) at other sign patterns, because x* sits at a
  corner of the box where all constraints are "tight" (uᵢ = 1 on the
  boundary).
- Even when non-empty, the analytic center value reflects how wide the
  open box is around the subspace in that orthant, not how "correct"
  the guess is. These quantities are not aligned.
- Adding log(cᵢ²) amplification (the optional two-plane term) can only
  make things worse: it rewards patterns where the analytic center sits
  further from zero on the affine subspace, which has nothing to do
  with whether that pattern matches x*.

---

## 4. Verdict

- **Continuous ellipsoid fitness is not a working fitness function** for
  this problem on the instances we can construct. It produces wrong
  rankings and wide swaths of infeasibility.
- **Discrete ‖A s − b‖² is a better fitness function** (x* is always the
  unique global optimum) and it's O(nnz(A)) per evaluation — no
  optimisation inside. It does have local minima, but empirically fewer
  than the continuous version has "wrong global optima + infeasible
  regions".
- **Neither is a complete algorithm.** Both get stuck.

## 5. Recommended next steps

1. **Drop the continuous fitness design.** The counterexamples are
   reproducible and decisive.
2. **Pivot to the right framing.** This problem — sparse {−1,+1}
   linear equations with a unique {−1,+1} solution — is a structured
   case of Boolean satisfaction / perceptron learning. The best-known
   algorithmic families on it are:
   - **Belief propagation / survey propagation** on the factor graph.
     Each constraint = factor; each variable = node. BP scales to the
     N = 10⁶ aspiration and is the state of the art for sparse
     constraint-satisfaction with a planted unique solution.
   - **Simulated annealing on ‖A s − b‖²**. Cheap per step, mature,
     and the small-N data show that x* is always the unique global
     min — SA escapes the local minima that greedy gets stuck in.
   - **Reweighted WalkSAT-style local search** (random flips biased by
     clause violation). Same ‖A s − b‖² objective, better move set.
3. **If we stay with gradient-style relaxations,** consider a penalty
   method instead of the interior-point / analytic-center form:
   minimise ‖A c − b‖² + λ Σ (1 − cᵢ²)² (or similar) over c ∈ ℝᴺ, which
   *pushes* c toward hypercube vertices rather than *excluding* them.
   This is a different design and is worth sketching before any more
   prototyping.
4. **Keep the small-N test bed** (`experiment_v4.py` + `counterexample.py`).
   It's ~250 lines, generates instances with verified uniqueness, and
   has already falsified one design. Any future fitness proposal should
   be run through it before we commit to more careful theory.

---

## Files

- `experiment_hillclimb.py` — core library: instance generator, discrete
  residual, continuous fitness with custom Newton solver, hill-climb,
  local-optimum counter.
- `experiment_v4.py` — sweeps N ∈ {6,8,10} and m across useful range,
  prints landscape statistics.
- `counterexample.py` — finds and prints an instance where x* is
  continuous-feasible but another sign pattern has higher fitness.

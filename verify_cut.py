import numpy as np

np.set_printoptions(precision=8, suppress=True)

print("="*70)
print("VERIFYING THE HALFSPACE CUT BEHAVIOR")
print("="*70)

# Setup
N = 3
c = np.zeros(3)
P = 3.0 * np.eye(3)
a = np.array([1, 1, 1], dtype=float)
b = 1.0

# Key quantities
Pa = P @ a  # (3,3,3)
aPa = a @ Pa  # 9
sqrt_aPa = np.sqrt(aPa)  # 3

# tau for the cut a·x ≤ b
# Note: different references define tau differently!
# Using: epsilon = (b - a·c) / sqrt(aPa) = (1-0)/3 = 1/3
# Or: tau = (a·c - b) / sqrt(aPa) = (0-1)/3 = -1/3

epsilon = (b - a @ c) / sqrt_aPa  # positive means center SATISFIES constraint
tau = -epsilon  # = -1/3

print(f"epsilon = (b - a·c)/√(aPa) = {epsilon:.6f}")
print(f"tau = (a·c - b)/√(aPa) = {tau:.6f}")
print(f"N = {N}")
print()

# The issue: is tau = -1/3 at the boundary?
# 
# The ellipsoid E(c, P) has:
# max a·x over E = a·c + √(aPa) = 0 + 3 = 3
# min a·x over E = a·c - √(aPa) = 0 - 3 = -3
#
# The cut a·x ≤ b = 1 removes the region where a·x > 1.
# This is NOT tangent -- it cuts off a significant cap!
# The tangent would be a·x ≤ 3 (touching the extreme point).
#
# So the formula with τ = -1/3 should give a NONTRIVIAL update.
# Let me verify by direct computation.

print("Direct verification that a·x ≤ 1 is a nontrivial cut:")
print(f"  max(a·x) over sphere = {sqrt_aPa:.4f}")
print(f"  b = {b}")
print(f"  The constraint removes a cap where a·x > 1")
print(f"  Fraction of sphere removed: significant (not tangent)")
print()

# Let me carefully apply the correct formula.
# From Grötschel-Lovász-Schrijver (GLS), the MVCE of E(c,P) ∩ {a^Tx ≤ b}:
# 
# Let σ = √(a^T P a), ε = (b - a^T c)/σ
# For -1 < ε < 1:
#   c' = c + ((1-ε)/(N+1)) · Pa/σ   ... wait, need to be careful about signs
#
# Actually from the Ellipsoid Method Wikipedia / standard references:
# Given E(c, P), cut with a^T x ≤ b where a^T c > b (center violates):
#   c' = c - (1/(N+1)) · Pa/√(aPa)   [central cut]
# 
# But here a^T c = 0 < b = 1, so the center SATISFIES the constraint.
# This is a "shallow cut" from the other side.

# Let me use the general formula from Schrijver/Khachiyan:
# E(c,P) ∩ {x: a^T x ≤ b}
# Let h = (b - a^T c) / √(a^T P a)  ∈ (-1, 1)
# For h ∈ (-1, 1), the MVCE is E(c', P') where:
#   c' = c + ((1-h)/(N+1)) · Pa/√(aPa) ... no
#
# I keep getting confused by sign conventions. Let me derive from scratch.

print("="*70)
print("DERIVATION FROM SCRATCH")
print("="*70)

# E = {x : (x-c)^T P^{-1} (x-c) ≤ 1}
# We want MVCE of E ∩ H where H = {x : a^T x ≤ b}
# 
# Transform to ball: y = P^{-1/2}(x-c), then E → B_N (unit ball)
# The halfspace becomes: a^T(P^{1/2}y + c) ≤ b
# i.e., (P^{1/2}a)^T y ≤ b - a^T c
# Let g = P^{1/2}a, d = b - a^T c
# Then g^T y ≤ d, or equivalently (g/||g||)^T y ≤ d/||g||
# Let ĝ = g/||g||, h = d/||g|| = (b - a^T c)/√(a^T P a)

h = (b - a @ c) / sqrt_aPa
print(f"h = (b - a^T c)/√(aPa) = {h:.6f}")
print(f"  h > 0 means center satisfies constraint")
print(f"  h = 1 means constraint is tangent")
print(f"  h < 0 means center violates constraint")
print()

# MVCE of B_N ∩ {ĝ^T y ≤ h} for h ∈ (-1, 1):
# (This is a "cap" of the unit ball)
# 
# The MVCE is E(y₀, D) where:
#   y₀ = -((1-h)/(N+1)) · ĝ   (shift toward the kept half)
#   D = ((N²(1-h²))/(N²-1)) · (I - ((2(1-h))/((N+1)(1+h))) · ĝĝ^T)
#      ... wait, let me check for h > 0 case

# Actually, for h ∈ (-1, 1), we're keeping the "bigger" part when h > 0.
# When h > 0 (center already inside H), the cut removes a small cap.
# The MVCE of the remaining part should be close to the original ball.

# For h = 1/3:
# c_shift = -((1-1/3)/(3+1)) · ĝ = -(2/3)/4 · ĝ = -(1/6) · ĝ
# Hmm, that seems like it moves the center TOWARD the hyperplane boundary.

# Wait, I need to be careful. The center shifts in the -ĝ direction 
# (away from the removed cap, toward the kept part). But if h > 0, 
# the removed cap is on the +ĝ side, so the center moves in the -ĝ 
# direction, which is AWAY from the hyperplane.

# No wait, let me think again. We keep {ĝ^T y ≤ h}. The removed part 
# is {ĝ^T y > h}. The cap is in the +ĝ direction. So the MVCE center 
# shifts in the -ĝ direction (away from removed cap, toward the center 
# of the remaining part). But the original center is at 0, which is 
# INSIDE the kept region (since h > 0, 0 satisfies ĝ^T 0 = 0 < h).

# For h = 1/3 in N=3:
print("In transformed (ball) coordinates:")
alpha = (1 - h) / (N + 1)
print(f"  α = (1-h)/(N+1) = {alpha:.6f}")
# Center shift in ball coords: Δy = -α · ĝ
# (The center moves in the -ĝ direction, away from the removed cap)
print(f"  Center shift in ĝ direction: -α = {-alpha:.6f}")

# Shape in ball coords:
scale = N**2 * (1 - h**2) / (N**2 - 1)
compress = 2 * alpha / (1 + h)
print(f"  Scale factor: N²(1-h²)/(N²-1) = {scale:.6f}")
print(f"  Compression: 2α/(1+h) = {compress:.6f}")

# Transform back to original coordinates:
# x = P^{1/2} y + c
# c' = P^{1/2} y₀ + c = c - α · P^{1/2} ĝ = c - α · P^{1/2} (P^{1/2}a)/(||P^{1/2}a||)
#    = c - α · Pa/√(aPa)

c_new = c - alpha * Pa / sqrt_aPa
print(f"\nTransformed back to original coordinates:")
print(f"  c' = c - α·Pa/√(aPa) = {c_new}")
print(f"  a·c' = {a @ c_new:.6f}")

# WAIT: the sign is wrong. Let me reconsider.
# We keep {a^T x ≤ b}. The "dangerous" direction is +a (toward the cap 
# that gets removed). The MVCE center should shift in the -a direction 
# (AWAY from the removed cap) if h > 0.
# 
# But I computed c' = c - α·Pa/√(aPa). Since α > 0 and Pa points in 
# the +a direction, c' moves in the -a direction from c.
# 
# c = (0,0,0), Pa/√(aPa) = (3,3,3)/3 = (1,1,1)
# c' = (0,0,0) - (1/6)(1,1,1) = (-1/6, -1/6, -1/6)
# a·c' = -1/2

# But this moves the center AWAY from the hyperplane a·x=1, which 
# makes sense: we removed the cap near a·x=1, so the remaining 
# ellipsoid is centered further from that boundary.

print(f"\n  c' = ({c_new[0]:.6f}, {c_new[1]:.6f}, {c_new[2]:.6f})")
print(f"  Moved in -a direction (away from removed cap)")

# Shape update
P_new = scale * (P - compress * np.outer(Pa, Pa) / aPa)
print(f"\n  P' = {scale:.4f} * (P - {compress:.4f} * Pa·Pa^T/aPa)")
print(f"  P' = ")
print(f"  {P_new}")
evals = np.linalg.eigvalsh(P_new)
print(f"  Eigenvalues of P': {evals}")

# Check vertices
print(f"\n  Vertex costs under new ellipsoid E(c', P'):")
from itertools import product
vertices = np.array(list(product([-1,1], repeat=3)), dtype=float)
x_star = np.array([1,1,-1], dtype=float)
for v in vertices:
    d = v - c_new
    cost = d @ np.linalg.solve(P_new, d)
    status = "SURFACE" if abs(cost-1)<0.01 else ("inside" if cost<1 else "OUTSIDE")
    xs = " [x*]" if np.allclose(v, x_star) else ""
    sat = " (a·v=1)" if abs(a@v - 1) < 1e-10 else ""
    print(f"    v={v}  cost={cost:.4f}  {status}{xs}{sat}")

print(f"\n  Direction check:")
neg_x_star = -x_star
print(f"  Center moved from (0,0,0) to {c_new}")
print(f"  Target: -x* = {neg_x_star}")
print(f"  Dot product c'·(-x*) = {c_new @ neg_x_star:.4f}")
print(f"  This is {'positive (correct direction!)' if c_new @ neg_x_star > 0 else 'negative (wrong direction)'}")

print("\n" + "="*70)
print("FIRST CUT MOVES THE RIGHT WAY!")
print("="*70)
print("""
Interesting! The cut a·x ≤ 1 (keeping the half WITHOUT the (1,1,1) cap)
moves the center to (-1/6, -1/6, -1/6), which is in the direction of 
(-1,-1,-1). 

For x* = (1,1,-1), the target is -x* = (-1,-1,1).
The dot product c'·(-x*) = 1/6 + 1/6 - 1/6 = 1/6 > 0.
So the center DOES move slightly toward -x*!

But this is ONLY the first cut (a·x ≤ 1, keeping the "lower" half).
We haven't applied the second cut (a·x ≥ 1, keeping the "upper" half).

The second cut would move the center in the +a direction (toward b=1),
which would push it back toward (1/3, 1/3, 1/3).

The net effect of BOTH cuts (= equality constraint) is to project 
the center onto the hyperplane.

KEY INSIGHT: For the "expanding" approach, we should NOT use equality 
constraints! We should use HALF-SPACE cuts.

But wait -- we need x* to be ON the hyperplane (not just in the halfspace).
If we only cut with a·x ≤ b, then we keep vertices with a·x ≤ b, which
includes x* (since a·x* = b, x* is on the boundary) plus many others
(all vertices with a·x ≤ b).
""")

# Now apply the SECOND cut: a·x ≥ 1, i.e., -a·x ≤ -1
print("="*70)
print("SECOND CUT: -a·x ≤ -b (i.e., a·x ≥ 1)")
print("="*70)

# For E(c_new, P_new), cut with (-a)^T x ≤ -b
g = -a
d_cut = -b
Pg = P_new @ g
gPg = g @ Pg
sqrt_gPg = np.sqrt(gPg)

h2 = (d_cut - g @ c_new) / sqrt_gPg
print(f"h = (d - g^T c')/√(gPg) = ({d_cut} - {g @ c_new:.4f})/√{gPg:.4f} = {h2:.6f}")

# Check if cut is valid
if h2 >= 1:
    print("h ≥ 1: Cut is tangent or doesn't intersect. No update needed.")
elif h2 <= -1:
    print("h ≤ -1: Infeasible!")
else:
    alpha2 = (1 - h2) / (N + 1)
    scale2 = N**2 * (1 - h2**2) / (N**2 - 1)
    compress2 = 2 * alpha2 / (1 + h2)
    
    c_new2 = c_new - alpha2 * Pg / sqrt_gPg
    P_new2 = scale2 * (P_new - compress2 * np.outer(Pg, Pg) / gPg)
    
    print(f"\nAfter second cut:")
    print(f"  c'' = {c_new2}")
    print(f"  a·c'' = {a @ c_new2:.4f}")
    print(f"  P'' eigenvalues: {np.linalg.eigvalsh(P_new2)}")
    
    print(f"\n  Vertex costs:")
    for v in vertices:
        d = v - c_new2
        cost = d @ np.linalg.solve(P_new2, d)
        status = "SURFACE" if abs(cost-1)<0.02 else ("inside" if cost<1 else "OUTSIDE")
        xs = " [x*]" if np.allclose(v, x_star) else ""
        sat = " (a·v=1)" if abs(a@v - 1) < 1e-10 else ""
        print(f"    v={v}  cost={cost:.4f}  {status}{xs}{sat}")
    
    print(f"\n  Net center movement: (0,0,0) → {c_new2}")
    print(f"  Dot with -x*: {c_new2 @ neg_x_star:.4f}")
    print(f"  This is {'positive (toward -x*!)' if c_new2 @ neg_x_star > 0 else 'negative (away from -x*)'}")

print("\n" + "="*70)
print("WHAT IF WE ONLY USE ONE-SIDED CUT?")
print("="*70)

print("""
If we interpret Operation 1 as a ONE-SIDED cut (not equality), we get:
  Cut a·x ≤ b: center moves to {c_new}, which has dot product 
  1/6 > 0 with -x*. This is the RIGHT direction, but barely.

The key question: WHICH side do we cut? We need x* to survive, so we 
must keep the side containing x*. But x* is ON the hyperplane (a·x*=b),
so x* survives EITHER cut.

If we don't know x*, both cuts are valid. But they push in opposite 
directions. We can't choose without knowing x*.

UNLESS: we use some OTHER information to decide. But the problem states 
we only have (a, b) and the current ellipsoid. No additional information.

WAIT: there's another option. What if we always cut with a·x ≤ b 
(not a·x ≥ b)? Since x* satisfies a·x* = b exactly, it's on the 
boundary of both halfspaces. The cut a·x ≤ b removes exactly the 
vertices with a·x > b, which does NOT include x*. So x* survives.

Similarly, a·x ≥ b removes exactly the vertices with a·x < b, which 
also doesn't include x*.

For a·x ≤ b: the center moves in the -a direction.
For a·x ≥ b: the center moves in the +a direction.

Without knowing x*, both are valid. The choice matters for convergence.
""")

# But what about the centroid approach?
print("="*70)
print("CENTROID APPROACH COMPARISON")  
print("="*70)

print(f"""
For the one-sided cut a·x ≤ 1:
  Surviving vertices (a·v ≤ 1): 7 out of 8 (all except (1,1,1))
  Their centroid: {np.mean([v for v in vertices if a@v <= 1+1e-10], axis=0)}
  MVCE center moves to: {c_new}

For the centroid of surviving vertices: it's the average of ALL vertices 
except (1,1,1), which is (-1/7)·(1,1,1) = (-1/7, -1/7, -1/7).
The MVCE center is at (-1/6, -1/6, -1/6), close to this.

For x* = (1,1,-1), dot product with -x*: 
  centroid: {np.mean([v for v in vertices if a@v <= 1+1e-10], axis=0) @ neg_x_star:.4f}
  MVCE center: {c_new @ neg_x_star:.4f}

Both are slightly positive -- barely in the right direction.

The movement is tiny (1/6 in each coordinate) compared to the target 
distance of √3 ≈ 1.73. And this is for N=3 with only 8 vertices.
For large N, the cut removes an exponentially small fraction of 
vertices, so the center barely moves.
""")

print("="*70)
print("THE REAL PROBLEM: SCALE")  
print("="*70)

print(f"""
For N=3: cut removes 1 vertex out of 8 = 12.5% → center moves O(1/N)
For general N with 3-variable constraint: removes about 1/4 of vertices
  (those with wrong parity on those 3 variables)
But vertices are exponential: 2^N
Each constraint removes ~1/4 of remaining vertices.
With ~N/3 constraints: remaining ≈ (3/4)^(N/3) · 2^N ≈ 2^(N-0.42N/3)
  = 2^(0.86N) ← still exponentially many!

The center of these remaining vertices is the centroid of a large set.
By symmetry, this centroid is close to the origin.
Each constraint only biases the centroid by O(1/N).

To reach -x* at distance √N, we need O(N · √N) = O(N^(3/2)) constraints,
but we only have O(N) constraints. The operations cannot move the center 
far enough.

Moreover, the directions of movement don't add up coherently toward -x*.
Each constraint pushes in a different direction (the constraint normal), 
and these normals are essentially random. The RMS movement after N/3 
constraints is O(√(N/3) · 1/N) = O(1/√N), not O(√N).
""")


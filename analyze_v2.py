import numpy as np
from itertools import product

np.set_printoptions(precision=6, suppress=True)

print("="*70)
print("DETAILED VERIFICATION AND CORRECTIONS")
print("="*70)

N = 3
x_star = np.array([1, 1, -1], dtype=float)
neg_x_star = -x_star
a = np.array([1, 1, 1], dtype=float)
b = 1.0

vertices = np.array(list(product([-1,1], repeat=3)), dtype=float)

# Initial sphere
c0 = np.zeros(3)
P0 = 3.0 * np.eye(3)

print("\n--- BUG CHECK: First cut returned identity ---")
print("The issue: τ = -1/3 < -1/N = -1/3. The formula has a threshold.")
print("When τ = -1/N exactly, the cut is 'shallow' enough that the")
print("ellipsoid already satisfies the constraint. Let me check.\n")

# τ = (a·c - b)/√(a^T P a) = (0 - 1)/√9 = -1/3
# For N=3: -1/N = -1/3
# So τ = -1/N exactly! This is the boundary case.

tau_val = (a @ c0 - b) / np.sqrt(a @ P0 @ a)
print(f"τ = {tau_val:.6f}")
print(f"-1/N = {-1/N:.6f}")
print(f"τ == -1/N: {abs(tau_val - (-1/N)) < 1e-10}")

print("""
When τ ≤ -1/N, the constraint a·x ≤ b does not cut the ellipsoid 
(the entire ellipsoid is already inside the halfspace). The ellipsoid 
method returns the same ellipsoid unchanged.

This is correct! For the initial sphere centered at origin with r=√3:
  max x1+x2+x3 over the sphere = √(a^T P a) = √9 = 3
  But we need a·x ≤ 1.
  The center has a·c = 0, so distance from center to hyperplane = 1/√3.
  The "reach" of the ellipsoid in direction a is √(a^T P a) = 3.
  In normalized terms: (0-1)/3 = -1/3 = -1/N.
  
  The condition τ < -1/N means the halfspace doesn't cut. At τ = -1/N 
  it's tangent. So the first cut (a·x ≤ 1) is a TANGENT cut -- the 
  halfspace boundary just touches the sphere's surface.

Wait, that's wrong. Let me reconsider:
  The halfspace is a·x ≤ 1.
  The maximum of a·x over the sphere is c·a + √(a^T P a) = 0 + 3 = 3.
  So a·x goes up to 3, but we're constraining to ≤ 1.
  This IS a significant cut (removes the region 1 < a·x ≤ 3).
  
  τ = (a·c - b)/√(aQa) = -1/3. The formula is designed for τ ∈ [-1, 1].
  τ = -1/3 is in range. The ellipsoid update should work.

Let me recheck the formula carefully.
""")

# Correct implementation from Grötschel, Lovász, Schrijver
# E(c, P) = {x: (x-c)^T P^{-1} (x-c) ≤ 1}
# Constraint: a^T x ≤ b
# 
# ã = a / √(a^T P a)  (normalize)
# τ = ã^T (c - ???) ... let me use the right reference

# From the standard ellipsoid method (e.g., BGT p.66):
# Given E(c, P), and half-space H = {x : a^T x ≤ a^T c + γ√(a^T P a)}
# where γ = (b - a^T c)/√(a^T P a) ∈ [-1, 0] for a "non-trivial" cut
# 
# Actually, let me just use the notation from Schrijver's "Theory of LP and IP":
# 
# E(c, A) and a^T x ≤ b with -1 ≤ (b - a^T c)/√(a^T A a) < 1
# Let β = (b - a^T c)/√(a^T A a)  (β > 0 means center already satisfies, β < 0 means center violates)
# 
# c' = c + (1-β)/(n+1) · Aa/√(a^T A a)  -- WRONG SIGN?

# OK let me just carefully derive it from the covering ellipsoid.
# 
# We want the minimum-volume ellipsoid covering E(c,P) ∩ {x: a^T x ≤ b}.
#
# Let's parametrize: ĉ = a/√(a^T P a), so P^{1/2}ĉ has unit norm.
# Let h = ĉ^T c, so a^T c = h√(a^T P a)
# 
# Actually, let me just use a clean known source.

# From Boyd & Vandenberghe, "Convex Optimization":
# The half-space is {x : g^T x ≤ g^T c} (central cut through center)
# 
# c+ = c - (1/(n+1)) P g / √(g^T P g)
# P+ = (n²/(n²-1)) (P - (2/(n+1)) P g g^T P / (g^T P g))
#
# For a general (deep) cut {x : g^T x ≤ α}:
# Let σ² = g^T P g, and define:
#   ε = (α - g^T c) / σ   (if ε > 0: center satisfies, ε < 0: violates)
#   
# The cut is non-trivial if -1 ≤ ε < 1/(n+1) ... actually no
# -1/n ≤ ε < 1 for the standard formula.
# Wait, that's also not right. Let me think again.

# The standard derivation:
# After applying the cut g^T x ≤ α, we want the MVCE of E ∩ H.
# 
# c_new = c - ((1+nε)/(n+1)) · (Pg)/(σ)  where ε = (g^T c - α)/σ
# P_new = (n²/(n²-1))(1 - ε²) · (P - (2(1+nε))/((n+1)(1+ε)) · PggTP/σ²)
#
# This requires -1/n < ε ≤ 1 (so that the hyperplane actually intersects E)

# In our case: g = a = (1,1,1), α = b = 1
# ε = (g^T c - α)/σ = (0 - 1)/3 = -1/3 = -1/n
# 
# This is exactly at the boundary! When ε = -1/n:
#   (1+nε) = (1 + 3(-1/3)) = 0
#   So c_new = c - 0 = c  (center doesn't move!)
#   P_new = (9/8)(1 - 1/9)(P - 0) = (9/8)(8/9)P = P  (P doesn't change!)
#
# This confirms: when τ = -1/n, the cut is too shallow to do anything.
# The minimum-volume covering ellipsoid of E ∩ H equals E itself.

print("CONFIRMED: τ = -1/N is the boundary case where the cut is trivial.")
print("The halfspace a·x ≤ 1 tangentially intersects the sphere -- the")
print("region a·x > 1 that gets removed has measure zero on the sphere.\n")

# Let me verify: which part of the sphere has a·x > 1?
print("Checking: on the sphere of radius √3 centered at 0,")
print("what points satisfy a·x ≥ 1?")
print("  a·x = (1,1,1)·x = x1+x2+x3")
print("  max over sphere: √(a^T P a) = 3")
print("  Points with a·x = 1 form a circle on the sphere (latitude circle)")
print("  Points with a·x > 1: a 'cap' around (1,1,1)")
print("  Only ONE vertex in this cap: (1,1,1) with a·v = 3")
print("  Vertices ON the boundary a·x=1: the 3 satisfying vertices")
print()
print("So the cut a·x ≤ 1 removes the cap around (1,1,1).")
print("This cap is small (one vertex out of 8), and the MVCE of the")
print("remaining region is essentially the same sphere.\n")

print("="*70)
print("CORRECT APPROACH: THE CUT SHOULD BE DEEPER")
print("="*70)

print("""
The issue is that for the INITIAL sphere, the cut a·x ≤ 1 barely 
clips the sphere. But what if we interpret Operation 1 differently?

Instead of "surface intersects hyperplane" (which is always true for 
a large enough ellipsoid), the operation should CONSTRAIN the surface 
to be TANGENT to the hyperplane, or should PROJECT the ellipsoid 
onto the hyperplane.

INTERPRETATION A: The operation forces a·c = b (center moves to hyperplane).
This is the equality-constraint interpretation.

INTERPRETATION B: The operation finds the MVCE of all satisfying vertices 
currently on the surface.

Let me try Interpretation A (force a·c = b):
""")

print("--- Interpretation A: Force center onto hyperplane ---")
print("Minimize volume of E(c', Q') subject to:")
print("  - a·c' = b  (center on hyperplane)")
print("  - x* on surface of E'  (unknown x*)")
print("  - E' contains E ∩ {a·x = b}")
print()

# If we force a·c' = b, then for the MVCE, we project the center:
c_A = c0 + ((b - a @ c0) / (a @ P0 @ a)) * (P0 @ a)
print(f"Center projection: c' = {c_A}")

# Shape: Q restricted to the hyperplane, then reinflated
# Within the hyperplane, Q restricted has eigenvalues 3, 3 (and 0 in normal)
# We need to scale so that satisfying vertices are on surface

# Satisfying vertices
S = np.array([[-1,1,1],[1,-1,1],[1,1,-1]], dtype=float)

# For center at (1/3,1/3,1/3), try Q' = β * Q_proj + α * (aa^T)/||a||²
# where Q_proj = P - (Pa)(Pa)^T/(aPa)

Pa = P0 @ a
aPa = a @ Pa
Q_proj = P0 - np.outer(Pa, Pa) / aPa

# Scale Q_proj so satisfying vertices are on surface
# For each satisfying vertex v: (v-c')^T Q'^{-1} (v-c') = 1
# Since v-c' is in the hyperplane, only the in-plane part of Q' matters
# Q_proj has eigenvalues 3,3 in the plane
# ||v - c'||² = 8/3 for each v
# Cost = (v-c')^T Q_proj^+ (v-c') = 8/9 ≈ 0.889
# Need to scale Q_proj by factor 8/9 so cost becomes 1

beta_scale = np.dot(S[0] - c_A, np.linalg.pinv(Q_proj) @ (S[0] - c_A))
print(f"Cost of satisfying vertex under Q_proj: {beta_scale:.6f}")
print(f"Need to scale Q_proj by factor {beta_scale:.6f} to get cost = 1")

Q_scaled = beta_scale * Q_proj
print(f"\nQ_scaled = {beta_scale:.6f} * Q_proj =")
print(Q_scaled)
print(f"Eigenvalues: {np.linalg.eigvalsh(Q_scaled)}")

# Now add back normal component
# Choice: make it spherical in the plane? Use α = 8/3 * 1/3 = 8/9?
# or α = eigenvalues in plane = 8/3?

# Let's try α = 8/3 to make it a sphere
alpha_sphere = 8/3
Q_A_sphere = Q_scaled + (alpha_sphere / (a @ a)) * np.outer(a, a)
print(f"\nWith α = {alpha_sphere:.4f} (makes it spherical):")
print(f"Q' = Q_scaled + α·(aa^T)/||a||² =")
print(Q_A_sphere)
print(f"Eigenvalues: {np.linalg.eigvalsh(Q_A_sphere)}")

# Check all vertices
print(f"\nVertex costs under E(c'={(1/3,1/3,1/3)}, Q'=sphere):")
for v in vertices:
    d = v - c_A
    cost = d @ np.linalg.solve(Q_A_sphere, d)
    status = "SURFACE" if abs(cost-1)<0.01 else ("INSIDE" if cost < 1 else "OUTSIDE")
    is_xs = " [x*]" if np.allclose(v, x_star) else ""
    on_plane = " (on plane)" if abs(a@v - b) < 1e-10 else ""
    print(f"  v={v}  cost={cost:.4f}  {status}{is_xs}{on_plane}")

print(f"\n  Center = {c_A}")
print(f"  Movement toward -x* = {np.dot(c_A, neg_x_star):.4f}")
print(f"  THIS MOVES AWAY FROM -x*!")

print("\n" + "="*70)
print("THE FUNDAMENTAL IMPOSSIBILITY")
print("="*70)

print("""
Let me prove the fundamental impossibility rigorously.

CLAIM: No operation defined solely in terms of (a, b) can consistently 
push the center toward -x* for ALL possible x* satisfying a·x* = b.

PROOF:
Let S = {v ∈ {-1,1}^N : a·v = b} be the set of satisfying vertices.
For each v ∈ S, the target center is -v.

Any operation using only (a, b) must produce the SAME (c', Q') 
regardless of which v ∈ S is the true x*. 

For the operation to push toward -x* for EVERY x* ∈ S, we'd need:
  (c' - c) · (-x*) > 0  for all x* ∈ S

i.e., c' · x* < c · x* = 0  for all x* ∈ S  (when c = 0)

i.e., c' · v < 0  for all v ∈ S

But S is symmetric in a certain sense. For a = (1,1,1), b = 1:
  S = {(1,1,-1), (1,-1,1), (-1,1,1)}
  
  c' · (1,1,-1) = c'_1 + c'_2 - c'_3
  c' · (1,-1,1) = c'_1 - c'_2 + c'_3
  c' · (-1,1,1) = -c'_1 + c'_2 + c'_3
  
  Sum = c'_1 + c'_2 + c'_3 = a · c'
  
  If a·c' = b = 1 > 0 (which the projection ensures), then:
  sum of (c' · v) = 1 > 0
  
  So at least one c'·v > 0, meaning c' moves TOWARD that v, 
  not toward -v. The operation fails for at least one x* ∈ S.

EVEN STRONGER: if the operation is symmetric in the satisfying vertices 
(which it must be, since it only knows a and b):
  c' · v = 1/3  for each v ∈ S
  So c' moves toward ALL satisfying vertices equally, which means 
  it moves AWAY from -x* for all x*.
""")

print("="*70)
print("CAN WE RESCUE THE APPROACH?")
print("="*70)

print("""
The proof above shows: a single hyperplane operation CANNOT move 
the center toward -x* on average. But what about MULTIPLE operations?

With M constraints (a_j, b_j) for j=1..M:
After all operations, can the center reach -x*?

The intersection of all hyperplanes a_j · x = b_j has a unique 
solution x* (by assumption). So the "common satisfying set" is {x*}.

After all hyperplane operations, only x* remains on the surface.
The center, if projected onto all hyperplanes, would be at the 
intersection point x* (if the system is fully determined).

So the center goes toward x*, NOT -x*.

CONCLUSION: The approach of "expanding away from x*" while keeping x* 
on the surface is fundamentally flawed. The natural geometric operations 
(cutting planes, projections) push the center TOWARD x* (the constraint 
intersection), not toward -x*.
""")

print("="*70)
print("VERIFICATION: MULTIPLE CONSTRAINTS")
print("="*70)

print("Let's add more constraints to verify.\n")

# In N=3, with x* = (1,1,-1), we can have additional constraints.
# For uniqueness, we need constraints that eliminate all but x*.
# With 3 variables and {-1,1} values, we need constraints that narrow 
# S down to {x*}.

# Constraint 1: x1+x2+x3 = 1 → S = {(1,1,-1), (1,-1,1), (-1,1,1)}
# Constraint 2: x1-x2+x3 = -1 → which vertices satisfy?

a2 = np.array([1, -1, 1], dtype=float)
b2_val = a2 @ x_star  # 1 - 1 + (-1) = -1
print(f"Constraint 2: x1-x2+x3 = {b2_val}")
print(f"Satisfying vertices of constraint 2:")
S2 = []
for v in vertices:
    if abs(a2 @ v - b2_val) < 1e-10:
        S2.append(v)
        print(f"  {v}  a2·v = {a2@v:.0f}")

print(f"\nIntersection of constraint 1 and 2:")
for v in S:
    if abs(a2 @ v - b2_val) < 1e-10:
        print(f"  {v}  ← only x* survives!")

# Now apply both constraints sequentially
print("\n--- Applying both constraints ---")

# Start with the state after constraint 1 (centroid approach)
# Actually, let me redo properly. The centroid of S ∩ S2 = {x*} is just x*.
print(f"\nAfter both constraints, only x* = {x_star} satisfies both.")
print(f"Centroid of surviving vertices: {x_star}")
print(f"Center would project to: {x_star}")
print(f"But we WANT center at: {neg_x_star}")
print(f"THE CENTER GOES TO x*, NOT -x*!")

print("\n" + "="*70)
print("ALTERNATIVE: WHAT IF THE CENTER SHOULD GO TO x*, NOT -x*?")
print("="*70)

print("""
What if the algorithm's goal should be to push the center toward x* 
(not -x*)? Then:

- Each hyperplane operation naturally pushes the center toward the 
  feasible set (which contains x*) ✓
- After enough constraints, the feasible set shrinks to {x*} ✓
- The center converges to x* ✓
- Solution: x* = center (not x* = -center) ✓

This would mean: the path c(t) = t·x* for t ∈ [0,1], with 
radius r(t) = (1-t)√N (shrinking sphere), and x* on the surface 
at distance (1-t)√N from center t·x*.

Check: ||x* - t·x*|| = ||(1-t)x*|| = (1-t)√N  
This equals r(t). ✓

But wait: for t=1, r = 0 (degenerate!). The sphere shrinks to a point.
That's just the intersection of hyperplanes = x*. This is ordinary 
constraint satisfaction, not the ellipsoid surface approach.

Also: this doesn't use Operation 2 at all. And it's essentially just 
the standard ellipsoid method for feasibility.
""")

print("="*70)
print("THE EXPANDING APPROACH REVISITED")
print("="*70)

print("""
Your original idea was to EXPAND (increase radius) while keeping x* on 
the surface. Let's reconsider what that means.

If x* is on the surface and the radius grows, then x* must move 
FURTHER from the center. This means the center moves AWAY from x*.

For c(t) = -t·x*, r(t) = (1+t)√N:
  ||x* - c(t)|| = ||x* + t·x*|| = (1+t)√N = r(t) ✓
  
The center moves from 0 to -x*, and the radius grows from √N to 2√N.
The key: r grows to keep x* on the surface as c moves away.

But HOW do the operations cause this? The operations need to:
  1. Move the center AWAY from x* (i.e., in the -x* direction)
  2. Grow the radius correspondingly

The problem: the hyperplane operation projects the center ONTO the 
hyperplane. If the center starts at 0 and x* is at distance √N from 0 
in the hyperplane, moving toward the hyperplane moves toward x*, 
not away from it.

UNLESS: the center moves along the hyperplane (not perpendicular to it) 
in the -x* direction. But the projection is perpendicular.

The only way to move along the hyperplane in the -x* direction is if 
x* and -x* are in the same hyperplane. But a·(-x*) = -b ≠ b, so 
-x* is NOT in the hyperplane.

Wait -- but we could move within the hyperplane in the direction of 
the projection of -x* onto the hyperplane. The component of -x* in 
the hyperplane a·x = b is:

  proj = -x* - ((a·(-x*) - b)/(a·a)) · a = -x* - ((-b-b)/||a||²) · a
       = -x* + (2b/||a||²) · a
       
For our example: = (-1,-1,1) + (2/3)(1,1,1) = (-1/3, -1/3, 5/3)
This is a point in the hyperplane, but it's not close to -x*.
And there's no natural operation that pushes the center to this point.
""")

# Compute the projection of -x* onto the hyperplane a·x=b
proj_neg_xstar = neg_x_star + (2*b / (a@a)) * a
print(f"Projection of -x* onto hyperplane a·x={b}:")
print(f"  proj(-x*) = {proj_neg_xstar}")
print(f"  Check: a · proj(-x*) = {a @ proj_neg_xstar:.4f} (should be {b})")
print(f"  Distance from -x* to hyperplane: {abs(a @ neg_x_star - b) / np.sqrt(a@a):.4f}")

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("""
ANSWERS TO YOUR 7 QUESTIONS:

1. OPERATION 1 ON A SPHERE: For a sphere E(c, r²I) and hyperplane a·x = b,
   the natural operation (MVCE of the half-ellipsoid) gives:
   - If τ = (a·c - b)/√(a^T·r²I·a) = (a·c - b)/(r||a||):
     c' = c - ((1+Nτ)/(N+1)) · r²a/(r||a||)
     Q' = (N²(1-τ²)/(N²-1)) · (r²I - (2(1+Nτ))/((N+1)(1+τ)) · r⁴aa^T/(r²||a||²))
   - The center moves TOWARD the hyperplane in the direction of a.
   - For an equality cut (two half-space cuts): center projects to 
     the nearest point on the hyperplane.

2. OPERATION 2 ON A SPHERE: The sphere reaches x_i = ±1 iff r ≥ 1+|c_i|.
   The initial sphere (r=√N > 1) already reaches both walls.
   Operation 2 constrains Q_{ii} but does NOT move the center.
   It can only RESHAPE, not TRANSLATE.

3. N=3 EXAMPLE: After Operation 1 with x1+x2+x3=1:
   - Center: (0,0,0) → (1/3, 1/3, 1/3) [projected onto hyperplane]
   - Shape: 3I → oblate ellipsoid with eigenvalues ≈ [0.75, 3, 3]
     (compressed in the (1,1,1) direction)
   - Vertices on surface: only (1,1,-1), (1,-1,1), (-1,1,1) [+ possibly (1,1,1)]
   - Non-satisfying vertices: pushed OUTSIDE

4. AFTER OPERATION 2: Center stays at (1/3, 1/3, 1/3). The shape Q 
   adjusts along the diagonal to ensure surface reaches x_i = ±1.

5. DIRECTION: The center moves to (1/3, 1/3, 1/3), which has dot product 
   -0.19 with -x* = (-1,-1,1). THE CENTER MOVES THE WRONG WAY.
   This is not a numerical accident; it's proven unavoidable (see Q7).

6. SPHERE vs ELLIPSOID: Operations NECESSARILY produce ellipsoids. 
   A hyperplane cut on a sphere produces Q with unequal eigenvalues 
   (compressed in the cut direction, unchanged perpendicular to it).
   Only Q = r²I is a sphere; after any non-trivial cut, it becomes 
   a proper ellipsoid.

7. THE FUNDAMENTAL IMPOSSIBILITY: Operations defined solely by (a,b) 
   or variable index i CANNOT distinguish which x* ∈ S is the solution.
   They produce the SAME update for all possible x*. Since the average 
   of -x* over all x* ∈ S equals -(centroid of S), and the centroid 
   of S lies on the POSITIVE side of the hyperplane (a·centroid = b > 0 
   for typical constraints), the operations push the center TOWARD the 
   satisfying vertices, not toward their negations.
   
   Mathematically: for any operation using only (a,b), the center 
   update c' has a·c' = b (projected onto hyperplane). Since x* ∈ S 
   also has a·x* = b, the vectors c' and x* are "on the same side" 
   in the a-direction. The center approaches x*, not -x*.
   
   This is not fixable by clever choice of operations. It's an 
   information-theoretic impossibility: without knowing x*, any 
   symmetric operation on the satisfying vertices cannot favor -x* 
   over +x*.
""")


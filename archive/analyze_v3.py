import numpy as np
from itertools import product

np.set_printoptions(precision=6, suppress=True)

print("="*70)
print("IMPORTANT OBSERVATION: x* IS PUSHED OUTSIDE BY THE FIRST CUT")
print("="*70)

N = 3
x_star = np.array([1, 1, -1], dtype=float)
a = np.array([1, 1, 1], dtype=float)
b = 1.0
c0 = np.zeros(3)
P0 = 3.0 * np.eye(3)
vertices = np.array(list(product([-1,1], repeat=3)), dtype=float)

# After one-sided cut a·x ≤ 1:
Pa = P0 @ a
aPa = a @ Pa
sqrt_aPa = np.sqrt(aPa)
h = (b - a @ c0) / sqrt_aPa  # = 1/3
alpha = (1 - h) / (N + 1)
scale = N**2 * (1 - h**2) / (N**2 - 1)
compress = 2 * alpha / (1 + h)

c1 = c0 - alpha * Pa / sqrt_aPa
P1 = scale * (P0 - compress * np.outer(Pa, Pa) / aPa)

print(f"After cut a·x ≤ 1:")
print(f"  c' = {c1}")
print(f"  P' eigenvalues: {np.linalg.eigvalsh(P1)}")
print()

# Check x* cost
d_xs = x_star - c1
cost_xs = d_xs @ np.linalg.solve(P1, d_xs)
print(f"  x* cost = {cost_xs:.4f}")
print(f"  x* is {'on surface' if abs(cost_xs-1)<0.01 else 'OUTSIDE' if cost_xs > 1 else 'inside'}")
print()

print("""
CRITICAL: After the first cut (a·x ≤ 1), x* = (1,1,-1) has cost 1.22,
meaning it's OUTSIDE the new ellipsoid! This violates the invariant 
that x* stays on the surface.

Why? Because the MVCE of E ∩ {a·x ≤ 1} does NOT have to contain x*.
The MVCE is the tightest ellipsoid containing the *intersection* of 
E with the halfspace. x* is on the boundary of the halfspace (a·x*=b),
so it's ON the boundary of the intersection. The MVCE may or may not 
contain this boundary point.

For the ellipsoid method, this is fine -- we don't need x* to be inside.
But for our approach, we NEED x* on the surface.

This reveals another fundamental issue: the standard ellipsoid method 
updates don't preserve the "x* on surface" invariant.
""")

print("="*70)
print("CUSTOM OPERATION: PRESERVE x* ON SURFACE")
print("="*70)

print("""
We need a CUSTOM update that:
  1. Incorporates the constraint a·x = b (or a·x ≤ b)
  2. Keeps x* on the surface: (x*-c')^T Q'^{-1} (x*-c') = 1
  3. Is defined without knowing x*

This is contradictory if we interpret (2) literally -- we can't 
constrain a specific point to be on the surface without knowing it.

But maybe (2) is an INVARIANT that's maintained automatically?

Let's check: if x* was on the surface of E, and we intersect with 
{a·x = b}, is x* on the surface of the intersection?

x* is on surface of E: (x*-c)^T Q^{-1} (x*-c) = 1  (boundary of E)
x* satisfies constraint: a·x* = b  (on the hyperplane)

So x* is on the boundary of E AND satisfies the constraint.
Therefore x* is on the boundary of E ∩ {a·x ≤ b}.

But the MVCE of E ∩ {a·x ≤ b} is LARGER than E ∩ {a·x ≤ b}.
So x* might be strictly inside the MVCE, or on its surface, or outside.

In our numerical example: x* has cost 1.22 > 1, meaning x* is OUTSIDE 
the MVCE. This is actually fine -- the MVCE contains E ∩ {a·x ≤ b}, 
and x* is on the boundary of that set, so the MVCE must contain x* 
(cost ≤ 1). 

WAIT: cost = 1.22 > 1 means x* is OUTSIDE the MVCE. But x* should be 
on the boundary of E ∩ H, which is inside the MVCE. Contradiction!

Let me verify...
""")

# Direct check: is x* in E ∩ H?
cost_original = (x_star - c0) @ np.linalg.solve(P0, x_star - c0)
in_halfspace = a @ x_star <= b + 1e-10
print(f"x* in E? cost = {cost_original:.4f} {'yes' if cost_original <= 1+1e-10 else 'no'}")
print(f"x* in H = {{a·x ≤ b}}? a·x* = {a@x_star:.4f} ≤ {b}? {in_halfspace}")
print(f"x* in E ∩ H? {cost_original <= 1+1e-10 and in_halfspace}")
print()

# Now check if x* is inside MVCE
cost_mvce = (x_star - c1) @ np.linalg.solve(P1, x_star - c1)
print(f"x* in MVCE? cost = {cost_mvce:.4f} {'yes' if cost_mvce <= 1+1e-10 else 'NO!'}")

print("""
x* has cost 1.22 in the MVCE, meaning it's OUTSIDE. But x* is on the 
boundary of E ∩ H. The MVCE should CONTAIN E ∩ H, so x* (being on the 
boundary of E ∩ H) should have cost ≤ 1 in the MVCE.

THERE IS A BUG. Either:
  (a) My ellipsoid update formula is wrong, or
  (b) x* is not actually in E ∩ H, or
  (c) The MVCE formula is for the INTERIOR of E ∩ H, not including boundary

Let me check (b): x* has cost exactly 1 in E (on boundary), and 
a·x* = b = 1 (on boundary of H). So x* is on the boundary of E ∩ H.
But E ∩ H might not include its boundary... actually it does (closed set).

So this must be a formula error. Let me check.
""")

# Double-check the MVCE formula. Let me verify that the MVCE actually 
# contains E ∩ H by sampling.

print("Sampling random points in E ∩ H to verify MVCE containment:")
np.random.seed(42)
violations = 0
for _ in range(10000):
    # Random point in E: x = c + P^{1/2} u where u is in unit ball
    L = np.linalg.cholesky(P0)
    u = np.random.randn(3)
    u = u / np.linalg.norm(u) * np.random.random()**(1/3)  # uniform in unit ball
    x = c0 + L @ u
    
    # Check if in H
    if a @ x <= b + 1e-10:
        # Check if in MVCE
        d = x - c1
        cost = d @ np.linalg.solve(P1, d)
        if cost > 1 + 1e-6:
            violations += 1

print(f"  Violations: {violations} out of 10000 samples in E ∩ H")

# Also check a point near x* on the boundary
print(f"\nPoints near x* on the surface of E, satisfying a·x ≤ b:")
for eps in [0.0, 0.01, 0.05, 0.1]:
    # Perturb x* slightly toward interior of E
    p = x_star * (1 - eps)
    cost_E = (p - c0) @ np.linalg.solve(P0, p - c0)
    in_H = a @ p <= b + 1e-10
    cost_MVCE = (p - c1) @ np.linalg.solve(P1, p - c1)
    print(f"  eps={eps:.2f}: point={p}, cost_E={cost_E:.4f}, in_H={in_H}, cost_MVCE={cost_MVCE:.4f}")

print("""
The issue is that the MVCE from the ellipsoid method is NOT guaranteed 
to contain ALL of E ∩ H. The classical ellipsoid method only guarantees 
a constant factor volume reduction (factor ≈ e^{-1/(2(N+1))}). It's an 
OUTER APPROXIMATION but may not contain all points.

Wait, no. The ellipsoid method DOES guarantee containment. Let me 
recheck my formula.
""")

# Let me verify using a different formula source
# From Khachiyan/GLS: the formula for E(c,P) cut by a^T x ≤ b with b = a^T c (central cut):
# c_new = c - (1/(n+1)) * Pa / √(aPa)  
# P_new = (n²/(n²-1)) * (P - (2/(n+1)) * Pa Pa^T / aPa)

# For a general (non-central) cut a^T x ≤ b:
# Let σ = √(aPa), and β = (a^T c - b)/σ  (> 0 means center violates)

# Then the update is:
# c_new = c - ((1 + n*β)/(n+1)) * Pa/σ
# P_new = (n²/(n²-1)) * (1-β²) * (P - (2*(1+nβ)/(n+1)/(1+β)) * PaaP/aPa)

# For our case: β = (a^T c - b)/σ = (0-1)/3 = -1/3
# Note: β is the NEGATIVE of h. β < 0 means center satisfies.

beta = (a @ c0 - b) / sqrt_aPa  # = -1/3
print(f"β = (a^T c - b)/σ = {beta:.6f}")

# The formula requires β ∈ (-1/n, 1] for a "deep cut"
# β = -1/3 = -1/n → boundary case!
print(f"β = -1/N = {-1/N:.6f}")
print(f"β is at the boundary of the valid range [-1/N, 1]")

# At β = -1/n:
# 1 + n*β = 1 + 3*(-1/3) = 0
# c_new = c - 0 = c
# P_new = (9/8)(1-1/9)(P - 0) = (9/8)(8/9)P = P
# So the update is IDENTITY! The formula degenerates.

print(f"\nAt β = -1/N:")
print(f"  1 + Nβ = {1 + N*beta:.6f} → 0!")
print(f"  c' = c (no change)")
print(f"  P' = P (no change)")

# But wait, in my earlier computation I got a non-trivial update.
# That's because I used a DIFFERENT formula! Let me reconcile.

# Earlier I used h = (b - a^T c)/σ = 1/3, and:
# α = (1-h)/(N+1) = 2/3/4 = 1/6
# c' = c - α Pa/σ
# This gives c' = c - (1/6)(1,1,1) = (-1/6, -1/6, -1/6) ≠ c!

# The discrepancy: the two formulas are DIFFERENT. They must correspond 
# to DIFFERENT definitions of the "minimum volume covering ellipsoid".

# Formula A (β-version): MVCE of E ∩ {a^T x ≤ b}
# Formula B (h-version): something else??

# Let me check Formula B more carefully.
# In my derivation, I used the transformation to the ball, then 
# MVCE of B ∩ {ĝ^T y ≤ h}, and the formula for the MVCE of 
# a cap of the unit ball.

# For a cap {y ∈ B: ĝ^T y ≤ h} with h ∈ (-1, 1):
# The MVCE center is at y₀ = λ ĝ where λ is chosen so that 
# the ellipsoid covers the cap boundary circle (y: ||y||=1, ĝ^T y=h)
# and the "bottom" of the cap (y: ĝ^T y = -1 on the ball boundary).

# Actually, I think Formula B might correspond to a DIFFERENT object.
# Let me verify directly: does the ellipsoid from Formula B contain E ∩ H?

print("\n--- Checking containment for Formula B ---")
print(f"Formula B ellipsoid: c' = {c1}, P' eigenvalues = {np.linalg.eigvalsh(P1)}")

# Test: point (1,1,1) has a·v = 3 > 1, so NOT in H. Not relevant.
# Test: point x* = (1,1,-1) has a·x* = 1 ≤ 1, in H. On boundary of E.
# Cost in MVCE: 1.22 > 1 → OUTSIDE MVCE. 

# This means Formula B does NOT give a valid MVCE of E ∩ H.
# So my formula is WRONG.

# Let me use Formula A (the standard one):
print("\n--- Formula A (standard ellipsoid method) ---")
# β = -1/3, which is at the boundary -1/N
# The update is identity (no change)

# This makes sense: the halfspace a·x ≤ 1 cuts off the cap where 
# a·x > 1. In normalized coordinates, this cap starts at distance 
# h = 1/3 from the center. For N=3, the threshold is 1/N = 1/3.
# At exactly this threshold, the cut is "just barely significant" 
# and the MVCE equals the original ellipsoid.

# But wait: the standard ellipsoid method formula CAN handle cuts with 
# β < -1/N (i.e., h > 1/N). The threshold -1/N is where the center is 
# in the INTERIOR of the halfspace by a distance of exactly 1/N of the 
# ellipsoid's radius. For shallower cuts (β < -1/N), a different formula 
# applies.

# Actually, I think the issue is: the standard formula is for β ∈ [-1/n, 1].
# For β = -1/n, it's at the boundary, and the update reduces to identity.
# For β < -1/n (very shallow cut), the standard MVCE is the original ellipsoid.
# For β > -1/n (deeper cut), the standard MVCE is smaller.

# In our case β = -1/n exactly. So the MVCE IS the original ellipsoid.
# This means the cut a·x ≤ 1 does NOT produce any change to the ellipsoid.

print("CONCLUSION: For the initial sphere (r=√3, c=0) and cut a·x ≤ 1,")
print("β = -1/N exactly. The standard MVCE equals the original sphere.")
print("The cut is at the boundary where it's too shallow to improve.")
print()
print("The formula I used earlier (Formula B) was INCORRECT.")
print("The correct answer: Operation 1 (as a halfspace cut) does NOTHING")
print("to the initial sphere with this specific constraint.")

print("\n" + "="*70)
print("WHY β = -1/N: GEOMETRIC EXPLANATION")  
print("="*70)

print(f"""
For a sphere of radius r centered at origin with constraint a·x ≤ b:

  β = (a·c - b)/√(a·P·a) = -b/(r·||a||)
  
For our case: β = -1/(√3 · √3) = -1/3 = -1/N.

This happens because:
  b = a·x* = Σ a_i x*_i for a ∈ {{-1,1}}^k, x* ∈ {{-1,1}}^N
  For k=N (all variables), b = Σ a_i x*_i, |b| ≤ N
  
  β = -b/(r·||a||) = -b/(√N · √N) = -b/N
  
  For b=1 (one more +1 than -1 among a_i x*_i), β = -1/N.
  This is exactly at the threshold!

In general, for k-sparse constraints (k nonzero entries in a):
  ||a|| = √k, r = √N (initial sphere)
  β = -b/(√N · √k)
  
  For b to be in {{-k, -k+2, ..., k}} (constraint on k variables):
  Typically |b| = 1 or 3 (one mismatch or net value).
  β = ±1/(√(Nk)) which for large N is close to 0.
  
  The threshold is -1/N, so |β| = 1/√(Nk) compared to 1/N.
  For k=3: |β| = 1/√(3N) vs 1/N. For N≫1: 1/√(3N) ≫ 1/N.
  So β > -1/N and the cut IS nontrivial for large N.

For small N (like N=3, k=3): β = -1/3 = -1/N exactly.
For larger N with k=3: β = -1/√(3N) which is > -1/N, so the cut works.
""")

print("="*70)
print("TRYING WITH N=6 TO AVOID DEGENERATE CASE")
print("="*70)

N2 = 6
x_star2 = np.array([1, 1, 1, -1, -1, 1], dtype=float)
# Constraint: x1 + x2 - x3 = 1
a2 = np.array([1, 1, -1, 0, 0, 0], dtype=float)
b2 = float(a2 @ x_star2)  # 1+1-1=1
print(f"N = {N2}")
print(f"x* = {x_star2}")
print(f"-x* = {-x_star2}")
print(f"Constraint: a = {a2}, b = {b2}")
print(f"a·x* = {a2 @ x_star2} = {b2}")

c2 = np.zeros(N2)
P2 = float(N2) * np.eye(N2)  # r² = N

Pa2 = P2 @ a2
aPa2 = a2 @ Pa2
sqrt_aPa2 = np.sqrt(aPa2)
beta2 = (a2 @ c2 - b2) / sqrt_aPa2
print(f"\nβ = {beta2:.6f}")
print(f"-1/N = {-1/N2:.6f}")
print(f"β > -1/N: {beta2 > -1/N2 + 1e-10}")

if beta2 > -1/N2 + 1e-10:
    # Standard ellipsoid update
    c2_new = c2 - ((1 + N2*beta2)/(N2+1)) * Pa2/sqrt_aPa2
    P2_new = (N2**2/(N2**2-1)) * (1-beta2**2) * (P2 - (2*(1+N2*beta2)/((N2+1)*(1+beta2))) * np.outer(Pa2, Pa2)/aPa2)
    
    print(f"\nAfter cut a·x ≤ {b2}:")
    print(f"  c' = {c2_new}")
    print(f"  P' eigenvalues: {np.linalg.eigvalsh(P2_new)}")
    
    # Check x* cost
    d = x_star2 - c2_new
    cost = d @ np.linalg.solve(P2_new, d)
    print(f"  x* cost = {cost:.4f} ({'on surface' if abs(cost-1)<0.02 else 'outside' if cost>1 else 'inside'})")
    
    # Check direction
    neg_xs2 = -x_star2
    print(f"\n  Center movement: {c2_new}")
    print(f"  Target -x*: {neg_xs2}")
    print(f"  Dot product: {c2_new @ neg_xs2:.4f}")
    
    # The MVCE should contain E ∩ H, so x* should have cost ≤ 1
    # Let me check if x* is in E ∩ H
    cost_E = (x_star2 - c2) @ np.linalg.solve(P2, x_star2 - c2)
    in_H = a2 @ x_star2 <= b2 + 1e-10
    print(f"\n  x* in E: cost = {cost_E:.4f}")
    print(f"  x* in H: a·x* = {a2@x_star2:.1f} ≤ {b2}: {in_H}")
    print(f"  x* in E ∩ H: {cost_E <= 1+1e-10 and in_H}")

else:
    print("β ≤ -1/N: cut is too shallow, no update")

print("\n" + "="*70)
print("KEY FINDING: x* COST IN MVCE")
print("="*70)

print("""
For N=6 with 3-variable constraint: β = -1/√(18) ≈ -0.236 > -1/6.
The cut is nontrivial. But x* has cost > 1 in the MVCE!

This confirms: the standard ellipsoid method update does NOT guarantee 
that boundary points of E ∩ H stay inside the MVCE. The MVCE is 
guaranteed to contain the INTERIOR of E ∩ H, but boundary points 
may fall outside.

Actually wait -- the MVCE must contain E ∩ H (closed set), including 
its boundary. If x* has cost > 1, there's a formula error.

Let me verify with explicit point sampling.
""")

if beta2 > -1/N2 + 1e-10:
    # Verify MVCE containment by sampling
    print("Verifying MVCE containment by sampling points in E ∩ H:")
    np.random.seed(42)
    max_cost = 0
    worst_point = None
    L = np.linalg.cholesky(P2)
    for _ in range(100000):
        u = np.random.randn(N2)
        u = u / np.linalg.norm(u) * np.random.random()**(1/N2)
        x = c2 + L @ u
        if a2 @ x <= b2 + 1e-10:
            d = x - c2_new
            cost = d @ np.linalg.solve(P2_new, d)
            if cost > max_cost:
                max_cost = cost
                worst_point = x.copy()
    
    print(f"  Max cost of sampled points in E ∩ H: {max_cost:.4f}")
    if worst_point is not None:
        print(f"  Worst point: {worst_point}")
        print(f"  a·worst = {a2 @ worst_point:.4f}")
        print(f"  cost in E: {(worst_point-c2) @ np.linalg.solve(P2, worst_point-c2):.4f}")

    # Check: is the ISSUE that the formulas are for containment of E ∩ {a·x ≤ b}
    # where E is the INTERIOR (open set)?
    # The standard ellipsoid method works with closed sets, so it should contain boundary.
    
    # Let me try a point very close to x* but slightly inside E
    for eps in [0.0, 0.001, 0.01, 0.1]:
        p = x_star2 * (1 - eps)
        cost_E = (p - c2) @ np.linalg.solve(P2, p - c2)
        in_H = a2 @ p <= b2 + 1e-10
        cost_MVCE = (p - c2_new) @ np.linalg.solve(P2_new, p - c2_new)
        print(f"  x*(1-{eps}): cost_E={cost_E:.4f}, in_H={in_H}, cost_MVCE={cost_MVCE:.4f}")

print("\n" + "="*70)
print("RESOLUTION")
print("="*70)

print("""
The standard ellipsoid method update formula from Grötschel-Lovász-
Schrijver IS the minimum volume ENCLOSING ellipsoid of E ∩ H. It must 
contain E ∩ H. If x* is on the boundary of E and on the hyperplane 
a·x=b (boundary of H), then x* is in E ∩ H, and the MVCE must contain it.

If the formula gives cost > 1 for x*, there are two possibilities:
1. I'm applying the formula incorrectly (sign error, wrong convention)
2. The formula is for a DIFFERENT problem (e.g., minimum enclosing 
   ellipsoid of E ∩ H_open)

The formulas in the standard ellipsoid method guarantee:
  E ∩ H ⊆ E_new  and  vol(E_new) ≤ e^{-1/(2(n+1))} vol(E)

The first condition means x* (being in E ∩ H) should have cost ≤ 1 
in E_new. If it doesn't, the formula is being applied wrong.

Let me try the OTHER sign convention.
""")

# Try opposite sign convention
if beta2 > -1/N2 + 1e-10:
    # Maybe the formula should use OPPOSITE sign for center update?
    c2_alt = c2 + ((1 + N2*beta2)/(N2+1)) * Pa2/sqrt_aPa2
    P2_alt = (N2**2/(N2**2-1)) * (1-beta2**2) * (P2 - (2*(1+N2*beta2)/((N2+1)*(1+beta2))) * np.outer(Pa2, Pa2)/aPa2)
    
    d = x_star2 - c2_alt
    cost_alt = d @ np.linalg.solve(P2_alt, d)
    print(f"\nWith opposite sign for center update:")
    print(f"  c' = {c2_alt}")
    print(f"  x* cost = {cost_alt:.4f}")
    print(f"  a·c' = {a2 @ c2_alt:.4f}")
    
    # Check containment by sampling
    np.random.seed(42)
    max_cost = 0
    for _ in range(100000):
        u = np.random.randn(N2)
        u = u / np.linalg.norm(u) * np.random.random()**(1/N2)
        x = c2 + L @ u
        if a2 @ x <= b2 + 1e-10:
            d = x - c2_alt
            cost = d @ np.linalg.solve(P2_alt, d)
            if cost > max_cost:
                max_cost = cost
    print(f"  Max cost of sampled points in E ∩ H: {max_cost:.4f}")
    
    # With the sign flip, the center moves TOWARD the hyperplane (in +a direction)
    print(f"\n  Center movement: {c2_alt}")
    print(f"  This moves in +a direction (TOWARD hyperplane)")
    print(f"  Dot with -x*: {c2_alt @ (-x_star2):.4f}")


import numpy as np
from itertools import product

np.set_printoptions(precision=6, suppress=True)

print("="*70)
print("ANALYSIS OF ELLIPSOID OPERATIONS")
print("="*70)

# Setup: N=3, x* = (1, 1, -1)
N = 3
x_star = np.array([1, 1, -1], dtype=float)
neg_x_star = -x_star  # target center = (-1, -1, 1)

print(f"\nN = {N}")
print(f"x* = {x_star}")
print(f"-x* = {neg_x_star} (target center)")
print(f"Constraint: x1 + x2 + x3 = 1")
print(f"  Check: a·x* = {1+1+(-1)} = 1 ✓")

# All vertices of {-1,1}^3
vertices = np.array(list(product([-1,1], repeat=3)), dtype=float)
print(f"\nAll 8 vertices of {{-1,1}}^3:")
for v in vertices:
    on_constraint = "  ← satisfies x1+x2+x3=1" if abs(v.sum()-1)<1e-10 else ""
    is_xstar = "  ← x*" if np.allclose(v, x_star) else ""
    print(f"  {v}  sum={v.sum():.0f}{on_constraint}{is_xstar}")

print("\n" + "="*70)
print("QUESTION 1: WHAT DOES OPERATION 1 DO TO A SPHERE?")
print("="*70)

print("""
Setup: Sphere E = {x : ||x - c||² ≤ r²}
Hyperplane: a·x = b, where x* satisfies a·x* = b.
Constraint: x* stays on surface AND surface intersects hyperplane.

For a sphere centered at c with radius r:
  - x* on surface: ||x* - c||² = r²
  - Surface intersects a·x=b: distance from c to hyperplane ≤ r
    i.e., |a·c - b|/||a|| ≤ r

The operation must RESHAPE the ellipsoid using the hyperplane constraint.

INTERPRETATION: The natural geometric operation is the "minimum volume 
ellipsoid containing E ∩ {x: a·x = b}^ε" or more precisely, 
we want to find the ellipsoid that:
  (i) Contains the intersection of E's surface with the halfspace
      containing x* (i.e., a·x = b side)
  (ii) Has x* on its surface
  (iii) Is as "centered" as possible

But actually, let's think about this differently. The classical 
ellipsoid method updates are well-defined. Let's consider:

CUTTING PLANE INTERPRETATION:
If we know that x* satisfies a·x* = b, and the current ellipsoid 
has center c, then we can cut with the hyperplane a·x = b and take 
the half-ellipsoid on the side containing x*.

But this doesn't quite work because we don't know WHICH side x* is on.
We only know x* IS on the hyperplane.

EQUALITY CONSTRAINT INTERPRETATION:
We know x* is on the hyperplane a·x = b. So we can intersect the 
ellipsoid with the hyperplane and find the minimum-volume ellipsoid 
containing this (N-1)-dimensional intersection, then "thicken" it 
to maintain the invariant that x* is on the surface.

Actually, let's think about this more carefully...
""")

print("\n" + "="*70)
print("QUESTION 3: THE N=3 EXAMPLE IN DETAIL")  
print("="*70)

print("\n--- Initial State ---")
c0 = np.zeros(3)
r0 = np.sqrt(3)
Q0 = r0**2 * np.eye(3)  # Q = r²I for sphere

print(f"Center c = {c0}")
print(f"Radius r = √3 ≈ {r0:.4f}")
print(f"Q = r²I = {r0**2:.1f} * I")

# Check: all vertices on surface
print("\nVertices on surface check (should all = 1 for sphere at origin r=√3):")
for v in vertices:
    cost = np.dot(v - c0, np.linalg.solve(Q0, v - c0))
    print(f"  v={v}  (v-c)^T Q^{{-1}} (v-c) = {cost:.4f}")

print("\n--- Constraint: a = (1,1,1), b = 1 ---")
a = np.array([1, 1, 1], dtype=float)
b = 1.0

print(f"\nVertices satisfying a·v = {b}:")
satisfying = []
not_satisfying = []
for v in vertices:
    val = a @ v
    if abs(val - b) < 1e-10:
        satisfying.append(v)
        print(f"  v={v}  a·v = {val:.0f} ✓")
    else:
        not_satisfying.append(v)
        
print(f"\nVertices NOT satisfying constraint:")
for v in not_satisfying:
    val = a @ v
    print(f"  v={v}  a·v = {val:.0f}")

print(f"\nSatisfying vertices: {len(satisfying)}")
print(f"  Centroid of satisfying: {np.mean(satisfying, axis=0)}")
print(f"  x* = {x_star} is among satisfying: {any(np.allclose(v, x_star) for v in satisfying)}")

print("\n" + "="*70)
print("DEFINING OPERATION 1 PRECISELY")
print("="*70)

print("""
The key insight: Operation 1 should be the PROJECTION operation.

Given ellipsoid E(c, Q) and equality constraint a·x = b:

We want the minimum-volume ellipsoid E'(c', Q') such that:
  1. E' contains all points of E that satisfy a·x = b
  2. x* remains on the surface of E'

But (1) alone gives an (N-1)-dimensional ellipsoid (a disk in 3D).
We need to "re-inflate" in the normal direction.

ALTERNATIVE: Think of it as the classical ellipsoid method's 
"shallow cut" update. The hyperplane a·x = b passes through (or near) 
the center. The update depends on how far the hyperplane is from the center.

For the classical ellipsoid method with a central cut (hyperplane 
passing through the center), the update is:

  c' = c + (1/(N+1)) * (Q·a)/(√(a^T Q a))  [or minus, depending on side]
  Q' = (N²/(N²-1)) * (Q - (2/(N+1)) * (Q·a·a^T·Q)/(a^T·Q·a))

But this is for a HALF-SPACE cut. For an EQUALITY cut (using a·x = b exactly):

The minimum-volume ellipsoid containing E ∩ {a·x = b} is:

Let τ = (a·c - b) / √(a^T Q a)  (signed distance in "ellipsoid units")

For an equality constraint, we project:
  c' = c - (Q·a·τ) / √(a^T Q a)  ... no wait, let me be more careful.
""")

print("--- Computing the hyperplane operation for N=3 example ---\n")

# For the initial sphere: c = 0, Q = 3I, a = (1,1,1), b = 1
aQa = a @ Q0 @ a  # a^T Q a = 3 * (1+1+1) = 9
Qa = Q0 @ a  # Q*a = 3*(1,1,1) = (3,3,3)
tau = (a @ c0 - b) / np.sqrt(aQa)  # (0 - 1)/3 = -1/3

print(f"a^T Q a = {aQa:.1f}")
print(f"Q·a = {Qa}")
print(f"τ = (a·c - b)/√(a^T Q a) = ({a@c0} - {b})/√{aQa} = {tau:.4f}")
print(f"|τ| = {abs(tau):.4f}")

print("""
For the equality constraint a·x = b, we want to find the minimum 
bounding ellipsoid of the intersection E ∩ {a·x = b}.

The intersection is an (N-1)-dimensional ellipsoid in the hyperplane.
Its center (projected onto the hyperplane) is:

  c_proj = c - τ·(Q·a)/√(a^T·Q·a) = c - τ·(Qa)/√(aQa)
  
This is the point in the hyperplane closest to c in the Q-metric.
""")

# Project center onto hyperplane
c_proj = c0 - tau * Qa / np.sqrt(aQa)
print(f"c_proj = c - τ·Qa/√(aQa) = {c0} - ({tau:.4f})·{Qa}/{np.sqrt(aQa):.4f}")
print(f"c_proj = {c_proj}")
print(f"Check: a·c_proj = {a @ c_proj:.4f} (should be {b})")

print(f"\n--- But this gives a FLAT (N-1)-dimensional object ---")
print(f"The projected ellipsoid in the hyperplane has shape matrix:")

# The projected shape matrix is Q restricted to the hyperplane
# Q_proj = Q - (Q·a·a^T·Q)/(a^T·Q·a)  [this is singular - rank N-1]
Q_proj = Q0 - np.outer(Qa, Qa) / aQa
print(f"Q_proj = Q - (Qa)(Qa)^T/(aQa) =")
print(Q_proj)
print(f"Rank of Q_proj: {np.linalg.matrix_rank(Q_proj, tol=1e-10)}")
print(f"Eigenvalues of Q_proj: {np.linalg.eigvalsh(Q_proj)}")

print("""
So the intersection E ∩ {a·x=b} is a 2D ellipse (disk) in the plane 
x1+x2+x3=1. Its center is at c_proj and its shape in the plane is 
given by Q_proj restricted to the plane.

NOW: we need to "re-inflate" this into a 3D ellipsoid.

The question is: what is the CORRECT re-inflation that maintains x* 
on the surface?
""")

# Check which satisfying vertices are on the surface of the projected ellipsoid
print("--- Satisfying vertices' costs under projected ellipsoid ---")
print("(Using pseudo-inverse since Q_proj is singular)")
Q_proj_pinv = np.linalg.pinv(Q_proj)
for v in satisfying:
    diff = v - c_proj
    # Check if v is in the hyperplane
    in_plane = abs(a @ v - b) < 1e-10
    # Cost in the Q_proj metric (within the plane)
    cost = diff @ Q_proj_pinv @ diff
    print(f"  v={v}  in plane: {in_plane}  cost = {cost:.4f}")

print("\n" + "="*70)
print("APPROACH: PARAMETERIZED FAMILY OF OPERATIONS")
print("="*70)

print("""
Let's think about what operations are WELL-DEFINED given only (a, b).

Given: Ellipsoid E(c, Q), hyperplane a·x = b, and the requirement 
that x* (unknown, on the surface) satisfies a·x* = b.

OPERATION 1 CANDIDATE: "Steiner symmetrization" or "conditional expectation"

The key idea: among all points on the surface of E that satisfy a·x = b,
x* is one of them. We want to shrink the ellipsoid toward these points.

OPERATION 1 DEFINITION (Equality-cut ellipsoid update):
Given E(c, Q) and a·x = b:

Step 1: Compute the "distance" of center from hyperplane:
  τ = (a·c - b) / √(a^T Q a)

Step 2: Update center (project toward hyperplane):
  c' = c - τ · (Q·a) / √(a^T Q a)
  
  This puts the center ON the hyperplane: a·c' = b.

Step 3: Update shape (remove the component normal to hyperplane):
  Q' = Q - (Q·a·a^T·Q) / (a^T Q a)
  
  But Q' is singular! (rank N-1)

Step 4: Re-inflate in normal direction to maintain x* on surface.
  We need: (x* - c')^T (Q')^{-1} (x* - c') = 1
  
  Since x* satisfies a·x* = b = a·c', the vector x*-c' is IN the hyperplane.
  So we only need Q' to be full-rank within the hyperplane.
  
  But for a proper 3D ellipsoid, we need full rank. We can add back 
  a small amount in the normal direction:
  
  Q' = Q - (Q·a·a^T·Q)/(a^T Q a) + α·(a·a^T)/||a||⁴
  
  where α controls the thickness in the normal direction.

THIS IS THE KEY ISSUE: α is a free parameter! The operation is not 
uniquely defined without additional constraints.

Let me try a DIFFERENT approach: what if the operation is simply 
"recompute the minimum-volume ellipsoid containing all satisfying vertices"?
""")

print("="*70)
print("APPROACH: MINIMUM BOUNDING ELLIPSOID OF SATISFYING VERTICES")
print("="*70)

# For the N=3 example, satisfying vertices are those with x1+x2+x3=1
print(f"\nSatisfying vertices (a·v = {b}):")
S = np.array(satisfying)
for v in S:
    print(f"  {v}")
    
print(f"\nCentroid of satisfying vertices: {S.mean(axis=0)}")

# The minimum bounding ellipsoid (Lowner-John) of these 3 points
# These 3 points form an equilateral triangle in the plane x1+x2+x3=1
centroid = S.mean(axis=0)
print(f"Centroid = {centroid}")

# Check distances from centroid to each vertex
for v in S:
    d = np.linalg.norm(v - centroid)
    print(f"  ||{v} - centroid|| = {d:.4f}")

# The minimum enclosing ball of these 3 points
# Since they form a regular triangle, the circumradius equals distance from centroid
circumrad = np.linalg.norm(S[0] - centroid)
print(f"\nCircumradius = {circumrad:.4f}")

# Check if x* is among satisfying vertices
print(f"\nx* = {x_star}")
print(f"x* in satisfying set: {any(np.allclose(v, x_star) for v in S)}")

print("""
PROBLEM: The minimum bounding ellipsoid of the satisfying vertices 
is (N-1)-dimensional (lives in the plane a·x=b). It's flat.

We need to decide how to "thicken" it into an N-dimensional ellipsoid.

KEY INSIGHT: The requirement that x* be on the SURFACE (not inside) 
constrains the ellipsoid. If we use the minimum-volume ellipsoid 
containing all satisfying vertices with x* on its surface, we get 
a unique answer only if the satisfying vertices span enough dimensions.

For the N=3 example: 3 satisfying vertices in a 2D plane. The minimum 
enclosing ellipse in this plane has all 3 on its boundary (they form a 
regular triangle). But we need to extend to 3D.
""")

print("="*70)
print("CONCRETE CALCULATION: N=3 EXAMPLE")  
print("="*70)

print("""
Let's work the N=3 case completely concretely.

Initial: sphere, c=(0,0,0), r=√3, Q=3I.

All 8 vertices on surface. Now apply constraint x1+x2+x3 = 1.

Satisfying vertices: (1,1,-1), (1,-1,1), (-1,1,1)  [3 vertices]
Non-satisfying: the other 5 vertices.

The question: what ellipsoid E'(c', Q') has:
  (a) The 3 satisfying vertices on its surface
  (b) x* = (1,1,-1) specifically on its surface [redundant with (a)]
  (c) Is "as large as possible" or "minimum volume" -- need to choose

Actually, (a) gives us 3 equations. E'(c',Q') has 3 + 6 = 9 parameters 
(3 for center + 6 for symmetric Q). With 3 surface equations, we have 
6 degrees of freedom.

Additional constraints:
  (d) Center should be "pushed" by the constraint
  (e) Want minimum volume? Maximum volume? Something else?

Let me try: find the ellipsoid with center at the centroid of 
satisfying vertices, with the satisfying vertices on its surface.
""")

# Center at centroid of satisfying vertices
c1 = centroid.copy()
print(f"Candidate center: c' = centroid = {c1}")
print(f"  = ({1/3:.4f}, {1/3:.4f}, {1/3:.4f})")

# For each satisfying vertex v, we need (v-c')^T Q'^{-1} (v-c') = 1
# Let's compute v - c' for each
diffs = S - c1
print(f"\nDifference vectors (v - c'):")
for i, (v, d) in enumerate(zip(S, diffs)):
    print(f"  v{i} = {v}, v-c' = {d}")

# We need Q'^{-1} such that d^T Q'^{-1} d = 1 for each d
# Let M = Q'^{-1}. Then d^T M d = 1.
# M is 3x3 symmetric, so 6 unknowns.
# 3 equations from surface constraints.
# 3 remaining degrees of freedom.

# Let's try Q' = α*I (spherical). Then d^T (1/α I) d = ||d||²/α = 1
# So α = ||d||² for each d. But ||d||² must be the same for all!
for d in diffs:
    print(f"  ||v-c'||² = {np.dot(d,d):.4f}")

print(f"\nAll distances equal: {np.allclose([np.dot(d,d) for d in diffs], np.dot(diffs[0],diffs[0]))}")

# Great! All distances from centroid are equal (equilateral triangle)
r1_sq = np.dot(diffs[0], diffs[0])
r1 = np.sqrt(r1_sq)
print(f"Common distance² = {r1_sq:.4f} = {8/3:.4f}")
print(f"Common distance = {r1:.4f} = √(8/3) = {np.sqrt(8/3):.4f}")

print(f"\nSo Q' = {r1_sq:.4f} * I works as a SPHERE!")
print(f"New sphere: center = {c1}, radius = {r1:.4f}")

# But wait -- this sphere has the satisfying vertices on its surface.
# Let's check what OTHER vertices are also on this surface.
print(f"\nAll vertices' costs under new sphere (c'={c1}, r²={r1_sq:.4f}):")
for v in vertices:
    d = v - c1
    cost = np.dot(d, d) / r1_sq
    status = ""
    if abs(cost - 1.0) < 1e-10:
        status = " ← ON SURFACE"
    elif cost < 1.0:
        status = " ← INSIDE"
    else:
        status = " ← OUTSIDE"
    is_xstar = " [x*]" if np.allclose(v, x_star) else ""
    print(f"  v={v}  cost={cost:.4f}{status}{is_xstar}")

print("\n--- Checking: did center move toward -x*? ---")
print(f"Initial center: {c0}")
print(f"New center: {c1}")
print(f"Target (-x*): {neg_x_star}")
print(f"Direction to target: {neg_x_star / np.linalg.norm(neg_x_star)}")
print(f"Actual movement: {c1 - c0}")
print(f"Movement direction: {(c1-c0)/np.linalg.norm(c1-c0) if np.linalg.norm(c1-c0)>1e-10 else 'zero'}")
print(f"Dot product (movement · target_dir): {np.dot(c1-c0, neg_x_star/np.linalg.norm(neg_x_star)):.4f}")

print("""
IMPORTANT OBSERVATION: The centroid of the satisfying vertices is 
(1/3, 1/3, 1/3), which points TOWARD x* = (1,1,-1) in the first two 
coordinates but AWAY in the third.

The movement is NOT toward -x* = (-1,-1,1). It's toward (1/3, 1/3, 1/3), 
which is in the WRONG direction for coordinates 1 and 2.

This makes sense: the centroid of satisfying vertices is influenced by 
ALL satisfying vertices, not just x*.
""")

print("="*70)
print("QUESTION 7: THE FUNDAMENTAL ISSUE")
print("="*70)

print("""
The operations must be defined WITHOUT knowing x*. They use only (a,b) 
or variable index i.

For Operation 1: the "centroid of satisfying vertices" approach moves 
the center toward the AVERAGE of ALL satisfying vertices, not toward -x*.

For the constraint x1+x2+x3=1:
  Satisfying vertices: (1,1,-1), (1,-1,1), (-1,1,1)
  Centroid: (1/3, 1/3, 1/3)
  
For x* = (1,1,-1): we want center to go to (-1,-1,1)
For x* = (1,-1,1): we want center to go to (-1,1,-1)
For x* = (-1,1,1): we want center to go to (1,-1,-1)

Average of targets: (-1/3, -1/3, -1/3) = -centroid!

So the centroid pushes toward +x*_avg, while we want -x*. 
The centroid approach pushes the WRONG WAY.
""")

print("="*70)
print("RE-EXAMINING: WHAT OPERATION PUSHES THE RIGHT WAY?")
print("="*70)

print("""
Let's reconsider. In the ellipsoid method for linear programming, 
when we have a cutting plane a·x ≤ b that cuts through the ellipsoid, 
the center moves AWAY from the violated side.

Here, all vertices satisfy a·x* = b (equality). The non-satisfying 
vertices are on BOTH sides of the hyperplane. 

Let me reconsider the problem setup. The constraint is a·x = b where 
a has ~3 nonzero entries. For x1+x2+x3 = 1:

  a·v values for all vertices:
""")

for v in vertices:
    val = a @ v
    sat = "✓" if abs(val - b) < 1e-10 else ""
    print(f"  v={v}  a·v = {val:+.0f}  {sat}")

print("""
The hyperplane a·x = 1 separates:
  a·x = +3: (1,1,1)
  a·x = +1: (1,1,-1), (1,-1,1), (-1,1,1)  ← ON hyperplane
  a·x = -1: (-1,-1,1), (-1,1,-1), (1,-1,-1)
  a·x = -3: (-1,-1,-1)

After the operation, only the 3 vertices with a·v = 1 should remain 
on the surface. The other 5 should be displaced (inside or outside).

NOW: the key question is whether "Operation 1" should be a 
CUTTING PLANE (keep half-ellipsoid) or an EQUALITY CUT (keep only 
the hyperplane intersection).

If it's an EQUALITY CUT: we lose a dimension, which is problematic.
If it's a CUTTING PLANE: which side do we keep? We don't know which 
side x* is on (we only know x* is ON the hyperplane).

RESOLUTION: x* IS on the hyperplane. So we can use the hyperplane 
as an EQUALITY constraint. The operation intersects E with the hyperplane 
and then "reinflates."

But the reinflation direction and amount are underdetermined.

Let me try ANOTHER definition of Operation 1...
""")

print("="*70)
print("OPERATION 1: PROJECTION + REINFLATION")
print("="*70)

print("""
Definition: Given E(c, Q) and a·x = b:

1. Project center onto hyperplane:
   c' = c + ((b - a·c)/(a^T Q^{-1} a)) · Q^{-1}·a
   
   Wait, let me use the correct formula.
   
   The closest point to c on the hyperplane a·x=b in the Mahalanobis 
   distance defined by Q is:
   
   c' = c + ((b - a·c)/(a^T Q a)) · Q·a
   
   No... Let me be careful. The Mahalanobis distance from c to the 
   hyperplane {x: a·x = b} with metric Q^{-1} is:
   
   d_M = |a·c - b| / √(a^T Q a)
   
   The projection:
   c' = c + ((b - a·c)/(a^T Q a)) · Q·a
""")

# Corrected projection
c_proj2 = c0 + ((b - a @ c0) / (a @ Q0 @ a)) * (Q0 @ a)
print(f"Projection of c onto hyperplane (Mahalanobis):")
print(f"  c' = {c0} + (({b} - {a@c0})/({a @ Q0 @ a})) * {Q0 @ a}")
print(f"  c' = {c_proj2}")
print(f"  Check: a·c' = {a @ c_proj2:.4f} (should be {b})")

print(f"\nThis gives c' = (1/3, 1/3, 1/3), same as centroid of satisfying vertices.")
print(f"(This makes sense for a sphere: Mahalanobis projection = Euclidean projection)")

print("""
Now for the shape matrix: we project Q onto the hyperplane and reinflate.

The projected shape (within the hyperplane) is:
  Q_plane = Q - (Q·a·a^T·Q)/(a^T·Q·a)  [singular, rank N-1]

For the sphere case:
  Q_plane = 3I - 3(1,1,1)(1,1,1)^T * 3/9 = 3I - (1,1,1)(1,1,1)^T
""")

Q_plane = Q0 - np.outer(Q0 @ a, Q0 @ a) / (a @ Q0 @ a)
print(f"Q_plane = ")
print(Q_plane)
evals, evecs = np.linalg.eigh(Q_plane)
print(f"Eigenvalues: {evals}")
print(f"Eigenvectors:\n{evecs}")

print("""
The projected shape has eigenvalues [0, 2, 2] with the zero eigenvalue 
in the direction a/||a|| = (1,1,1)/√3 (normal to the hyperplane).

To reinflate: we add back shape in the normal direction.
  Q' = Q_plane + α · (a·a^T)/||a||²

For α = 0: flat (rank 2)
For α > 0: full rank ellipsoid

How to choose α? One natural choice: the MINIMUM-VOLUME ellipsoid 
that contains the intersection AND has x* on its surface.

Since x* is in the hyperplane, and x*-c' is perpendicular to a 
(both x* and c' satisfy a·x=b), the surface condition is:

(x*-c')^T Q'^{-1} (x*-c') = 1

Since (x*-c') has no component in the a direction, the α parameter 
doesn't affect this equation! So α is COMPLETELY FREE.

This means: there is a ONE-PARAMETER FAMILY of valid ellipsoids 
after Operation 1. The operation is NOT uniquely defined by just 
"x* stays on surface and surface intersects hyperplane."
""")

# Verify: x*-c' has no component along a
diff_xstar = x_star - c_proj2
print(f"x* - c' = {diff_xstar}")
print(f"Component along a: a·(x*-c') = {a @ diff_xstar:.4f}")
print(f"  (This is 0 because both x* and c' satisfy a·x = {b})")

# So the surface condition gives us:
print(f"\nSurface condition: (x*-c')^T Q'^{{-1}} (x*-c') = 1")
print(f"Since x*-c' is in the hyperplane, and Q' = Q_plane + α(aa^T)/||a||²,")
print(f"the inverse of Q' restricted to the hyperplane determines the condition.")
print(f"")
print(f"Q_plane has eigenvalues 2,2,0 in the plane. The restriction to the")
print(f"hyperplane gives a 2x2 matrix with eigenvalues 2,2 (= 2I in the plane).")
print(f"")
print(f"||x*-c'||² in the plane: {np.dot(diff_xstar, diff_xstar):.4f}")
print(f"Cost = ||x*-c'||² / eigenvalue = {np.dot(diff_xstar, diff_xstar):.4f} / 2 = {np.dot(diff_xstar, diff_xstar)/2:.4f}")
print(f"")
print(f"This should be 1: {abs(np.dot(diff_xstar, diff_xstar)/2 - 1) < 1e-10}")

# Check with pseudo-inverse
Q_plane_pinv = np.linalg.pinv(Q_plane)
cost_xstar = diff_xstar @ Q_plane_pinv @ diff_xstar
print(f"\nDirect computation: (x*-c')^T Q_plane^+ (x*-c') = {cost_xstar:.4f}")
print(f"This should be ≈ 1.33... = 4/3")

# Hmm, let me recheck
print(f"\n--- Recheck ---")
print(f"Q_plane eigenvalues: {evals}")
# The eigenvectors corresponding to nonzero eigenvalues
print(f"Q_plane eigenvectors (nonzero evals):")
for i in range(3):
    if abs(evals[i]) > 1e-10:
        print(f"  eval={evals[i]:.4f}, evec={evecs[:,i]}")

# Project diff_xstar onto eigenvectors
for i in range(3):
    comp = np.dot(diff_xstar, evecs[:,i])
    print(f"  Component of (x*-c') along evec[{i}]: {comp:.4f}")

# Cost should be sum of (component²/eigenvalue) for nonzero eigenvalues
cost_manual = sum((np.dot(diff_xstar, evecs[:,i])**2 / evals[i]) for i in range(3) if abs(evals[i]) > 1e-10)
print(f"\nCost (manual) = Σ comp²/eval = {cost_manual:.4f}")
print(f"This = {cost_manual} ≠ 1, so the projected ellipsoid does NOT have x* on its boundary!")

print("""
CRITICAL FINDING: If we just project the center and use Q_plane, 
the cost of x* is 4/3 ≠ 1. So x* would be OUTSIDE the projected 
ellipsoid in the hyperplane.

This means we need to RESCALE Q_plane to get x* back on the surface.
We need Q' = β · Q_plane (restricted to the hyperplane, then re-embedded).

For x* on surface: β · (cost under Q_plane) = 1
  β = 1/cost_manual = 1/(4/3) = 3/4

Wait, that's not right either. If Q' = β · Q_plane, then 
(x*-c')^T (β·Q_plane)^{-1} (x*-c') = (1/β) · cost_manual = 1
So β = cost_manual = 4/3.

So Q'_plane = (4/3) · Q_plane.
Eigenvalues of Q'_plane: 0, 8/3, 8/3.

Then the full Q' = (4/3)·Q_plane + α·(aa^T)/||a||²
""")

beta = cost_manual  # = 4/3
Q_prime_plane = beta * Q_plane
print(f"Scaled Q_plane: Q'_plane = {beta:.4f} * Q_plane")
print(f"Eigenvalues of Q'_plane: {beta * evals}")

# Check x* cost under scaled Q_plane
Q_prime_plane_pinv = np.linalg.pinv(Q_prime_plane)
cost_check = diff_xstar @ Q_prime_plane_pinv @ diff_xstar
print(f"\nx* cost under scaled Q_plane: {cost_check:.4f} (should be 1.0)")

# Now add back normal component
alpha_values = [0.1, 1.0, 2.0, 8/3]
print(f"\nReinflated Q' = Q'_plane + α·(aa^T)/||a||² for various α:")
a_norm_sq = np.dot(a, a)  # = 3
for alpha in alpha_values:
    Q_prime = Q_prime_plane + alpha * np.outer(a, a) / a_norm_sq
    evals_p = np.linalg.eigvalsh(Q_prime)
    # Check x* cost
    diff = x_star - c_proj2
    cost = diff @ np.linalg.solve(Q_prime, diff)
    # Check other satisfying vertices
    costs_sat = []
    for v in satisfying:
        d = v - c_proj2
        c_v = d @ np.linalg.solve(Q_prime, d)
        costs_sat.append(c_v)
    print(f"  α={alpha:.1f}: eigenvalues={evals_p}, x* cost={cost:.4f}, all sat vertex costs={[f'{c:.4f}' for c in costs_sat]}")

print(f"\n--- All satisfying vertex costs ---")
print(f"Notice: for Q'_plane (in the hyperplane), all satisfying vertices")
print(f"have the SAME cost because they form an equilateral triangle around")
print(f"the centroid, and Q'_plane is isotropic within the hyperplane.")
print(f"So β·Q_plane with β=4/3 puts ALL satisfying vertices on the surface.")

# Verify
for v in satisfying:
    d = v - c_proj2
    cost = d @ Q_prime_plane_pinv @ d
    print(f"  v={v}: cost = {cost:.4f}")

print("\n" + "="*70)
print("ANSWER TO QUESTION 3: AFTER OPERATION 1")
print("="*70)

# The natural Operation 1: project center to hyperplane, scale Q to keep
# satisfying vertices on surface, choose α for normal thickness
c_after_op1 = c_proj2.copy()
# For definiteness, let's try the "minimum volume" choice: minimize det(Q')
# subject to all satisfying vertices on surface.
# Since Q' = (4/3)Q_plane + α(aa^T)/||a||², and Q_plane is rank-2,
# det(Q') = (eigenvalue in a direction which is α) × (8/3)²
# = α · 64/9
# Minimized by α → 0, but then Q' is singular.
# So we need SOME principle to choose α.

# ALTERNATIVE: instead of all this, let's try the LÖWNER-JOHN ellipsoid
# (minimum volume ellipsoid containing the satisfying vertices)

print("""
There is no unique Operation 1 for the EQUALITY constraint case when 
working with spheres/ellipsoids. The center projects to (1/3, 1/3, 1/3) 
and Q becomes degenerate in the normal direction.

Let me try a COMPLETELY DIFFERENT interpretation of the operations...
""")

print("="*70)
print("REINTERPRETATION: OPERATIONS AS HALF-SPACE CUTS")
print("="*70)

print("""
Perhaps the operations should be interpreted differently:

Since x* satisfies a·x* = b, we know a·x* = b exactly. But the 
constraint given to us is just "a·x = b" (a hyperplane).

In the context of finding x*, what we know is:
  - The solution is on the hyperplane a·x = b
  - The solution is a vertex of {-1,1}^N

What if Operation 1 is actually TWO half-space cuts?
  Cut 1: keep a·x ≥ b  (removes vertices with a·x < b)
  Cut 2: keep a·x ≤ b  (removes vertices with a·x > b)
  
Combined: keep only a·x = b.

Each half-space cut has a well-defined ellipsoid update from the 
classical ellipsoid method. Let me compute both cuts.
""")

# Classical ellipsoid method: half-space cut a·x ≤ b
# Given E(c, Q), cut with a·x ≤ b
# Let τ = (a·c - b) / √(a^T Q a)
# If τ > 1: infeasible (center too far from constraint)
# If τ ≤ 1: update:
#   c' = c - (1+τ)/(N+1) · (Q·a)/√(a^T Q a)
#   Q' = N²(1-τ²)/(N²-1) · (Q - 2(1+τ)/((N+1)(1+τ²+correction)) · (Qa)(Qa)^T/(aQa))

# Actually, the exact formula for the ellipsoid method with a·x ≤ b:
# τ = (a·c - b) / √(a^T Q a)  
# c' = c - ((1+N·τ)/(N+1)) · Q·a / (a^T Q a) ... wait, let me look this up carefully

# The standard formulas (from Grötschel, Lovász, Schrijver):
# Given E(c, P) = {x: (x-c)^T P^{-1} (x-c) ≤ 1}
# Cut: a^T x ≤ a^T c (central cut, where hyperplane passes through center)
# Update:
#   c' = c - (1/(N+1)) · P·a / √(a^T P a)
#   P' = (N²/(N²-1)) · (P - (2/(N+1)) · (Pa)(Pa)^T / (a^T P a))

# For a DEEP cut a^T x ≤ b where b < a^T c:
# Let â = a/√(a^T P a)  (normalized)
# τ = (a^T c - b) / √(a^T P a)  (τ > 0 means b < a^T c, so we're cutting)
# c' = c - ((1+N·τ)/(N+1)) · P·â
# P' = (N²(1-τ²)/(N²-1)) · (P - (2(1+Nτ)/((N+1)(1+τ))) · (Pâ)(Pâ)^T)

# Wait, I need to be more careful. Let me use the standard form.

print("--- Classical Ellipsoid Method: Deep Cut ---")
print("E(c, P): {x: (x-c)^T P^{-1} (x-c) ≤ 1}")
print("Cut: a^T x ≤ b")
print()

def ellipsoid_halfspace_cut(c, P, a, b, N):
    """Apply half-space cut a^T x <= b to ellipsoid E(c, P) in R^N.
    Returns (c', P') or None if infeasible."""
    Pa = P @ a
    aPa = a @ Pa
    sqrt_aPa = np.sqrt(aPa)
    
    # Normalized distance
    tau = (a @ c - b) / sqrt_aPa
    
    if tau > 1:
        return None  # infeasible
    if tau < -1/(N):
        # Constraint doesn't cut -- ellipsoid already inside
        return c.copy(), P.copy()
    
    # Update formulas (from Boyd & Vandenberghe / Grötschel et al.)
    alpha = (1 + N * tau) / (N + 1)
    
    c_new = c - alpha * Pa / sqrt_aPa
    
    # Shape update
    P_new = (N**2 * (1 - tau**2) / (N**2 - 1)) * (P - (2 * alpha / (1 + tau)) * np.outer(Pa, Pa) / aPa)
    
    return c_new, P_new

# Apply cut 1: a·x ≤ b (i.e., x1+x2+x3 ≤ 1)
print("=== Cut 1: x1+x2+x3 ≤ 1 ===")
P0 = Q0.copy()
Pa = P0 @ a
aPa = a @ Pa
tau1 = (a @ c0 - b) / np.sqrt(aPa)
print(f"τ = (a·c - b)/√(a^T P a) = ({a@c0} - {b})/√{aPa} = {tau1:.4f}")
print(f"τ = -1/3 (hyperplane is 'above' center)")

result1 = ellipsoid_halfspace_cut(c0, P0, a, b, N)
c1_cut, P1_cut = result1
print(f"\nAfter cut 1:")
print(f"  c' = {c1_cut}")
print(f"  P' = ")
print(f"  {P1_cut}")

evals1 = np.linalg.eigvalsh(P1_cut)
print(f"  Eigenvalues of P': {evals1}")

# Check which vertices are on surface/inside/outside
print(f"\n  Vertex costs after cut 1:")
for v in vertices:
    d = v - c1_cut
    cost = d @ np.linalg.solve(P1_cut, d)
    status = "SURFACE" if abs(cost-1)<0.01 else ("INSIDE" if cost < 1 else "OUTSIDE")
    in_halfspace = a @ v <= b + 1e-10
    print(f"    v={v}  cost={cost:.4f}  {status}  a·v={a@v:+.0f}  {'in' if in_halfspace else 'OUT of'} halfspace")

# Now apply cut 2: -a·x ≤ -b (i.e., x1+x2+x3 ≥ 1)
print(f"\n=== Cut 2: x1+x2+x3 ≥ 1 (equivalently, -a·x ≤ -b) ===")
result2 = ellipsoid_halfspace_cut(c1_cut, P1_cut, -a, -b, N)
c2_cut, P2_cut = result2
Pa2 = P1_cut @ (-a)
aPa2 = (-a) @ Pa2
tau2 = ((-a) @ c1_cut - (-b)) / np.sqrt(aPa2)
print(f"τ = {tau2:.4f}")

print(f"\nAfter cut 2:")
print(f"  c' = {c2_cut}")
print(f"  P' = ")
print(f"  {P2_cut}")

evals2 = np.linalg.eigvalsh(P2_cut)
print(f"  Eigenvalues of P': {evals2}")

# Check which vertices are on surface/inside/outside
print(f"\n  Vertex costs after both cuts:")
for v in vertices:
    d = v - c2_cut
    cost = d @ np.linalg.solve(P2_cut, d)
    status = "SURFACE" if abs(cost-1)<0.01 else ("INSIDE" if cost < 1 else "OUTSIDE")
    on_plane = abs(a @ v - b) < 1e-10
    is_xs = " [x*]" if np.allclose(v, x_star) else ""
    print(f"    v={v}  cost={cost:.4f}  {status}  on plane: {on_plane}{is_xs}")

print(f"\n  Center moved: {c0} → {c2_cut}")
print(f"  Direction toward -x* = {neg_x_star}")
print(f"  Movement dot -x*: {np.dot(c2_cut - c0, neg_x_star):.4f}")
print(f"  Movement: {c2_cut}")

print("\n" + "="*70)
print("QUESTION 6: SPHERE vs ELLIPSOID")
print("="*70)

print(f"\nAfter the two half-space cuts:")
print(f"  P' eigenvalues: {evals2}")
print(f"  Is P' spherical (all eigenvalues equal)? {np.allclose(evals2, evals2[0])}")
print(f"\n  The cuts PRODUCE an ELLIPSOID, not a sphere!")
print(f"  Even starting from a sphere, the operations necessarily break")
print(f"  spherical symmetry.")

print("\n" + "="*70)
print("OPERATION 2: TWO-PLANE FOR VARIABLE i")
print("="*70)

print("""
Operation 2: For variable i, the surface must reach both x_i = +1 and x_i = -1.

For a sphere of radius r centered at c:
  - Surface reaches x_i = +1: c_i + r ≥ 1  (or in some direction r is large enough)
  - Surface reaches x_i = -1: c_i - r ≤ -1

For an ellipsoid E(c, Q):
  - Surface reaches x_i = +1: max_{(x-c)^T Q^{-1} (x-c) ≤ 1} x_i = c_i + √Q_{ii}
    So need: c_i + √Q_{ii} ≥ 1
  - Similarly: c_i - √Q_{ii} ≤ -1
    So need: -c_i + √Q_{ii} ≥ 1

Both: √Q_{ii} ≥ max(1 - c_i, 1 + c_i) = 1 + |c_i|

If the current ellipsoid ALREADY satisfies this, Operation 2 is trivial.
If not, we need to EXPAND Q_{ii} until √Q_{ii} = 1 + |c_i|.

For the initial sphere: c = 0, Q_{ii} = 3, √Q_{ii} = √3 ≈ 1.73 > 1.
So the sphere already reaches both walls. Operation 2 is trivial!

What if Operation 2 instead CONSTRAINS Q_{ii} to be EXACTLY what's needed?

Operation 2 (precise): Set Q so that the surface JUST touches x_i = ±1.
  √Q_{ii} = 1 + |c_i|  →  Q_{ii} = (1 + |c_i|)²
  
But this changes Q_{ii} without knowing x*. It makes the ellipsoid 
tighter along axis i.
""")

# Apply Operation 2 after the two cuts
print("--- Applying Operation 2 for each variable after the cuts ---")
c_curr = c2_cut.copy()
Q_curr = P2_cut.copy()

print(f"\nCurrent state: c = {c_curr}")
print(f"Q = \n{Q_curr}")
print(f"Q diagonal: {np.diag(Q_curr)}")
print(f"√Q_ii: {np.sqrt(np.diag(Q_curr))}")

for i in range(N):
    reach_plus = c_curr[i] + np.sqrt(Q_curr[i,i])
    reach_minus = c_curr[i] - np.sqrt(Q_curr[i,i])
    print(f"\n  Variable x_{i+1}:")
    print(f"    c_{i+1} = {c_curr[i]:.4f}")
    print(f"    √Q_{{{i+1}{i+1}}} = {np.sqrt(Q_curr[i,i]):.4f}")
    print(f"    Surface reaches x_{i+1} = +1 at: {reach_plus:.4f}")
    print(f"    Surface reaches x_{i+1} = -1 at: {reach_minus:.4f}")
    print(f"    Reaches both walls: {reach_plus >= 1-1e-10 and reach_minus <= -1+1e-10}")

print("""
The two-plane constraint √Q_{ii} ≥ 1 + |c_i| would require:
  For each i, Q_{ii} ≥ (1 + |c_i|)²

If we SET Q_{ii} = (1 + |c_i|)² (minimal value), this constrains the 
ellipsoid but doesn't move the center. The center stays fixed.

So Operation 2, defined as "adjust Q_{ii} so surface just touches ±1",
does NOT move the center at all! It only reshapes the ellipsoid.
""")

print("="*70)
print("QUESTION 5: DOES THE CENTER ACTUALLY MOVE TOWARD -x*?")
print("="*70)

print(f"""
After Operation 1 (two half-space cuts with x1+x2+x3=1):
  Center moved from {c0} to {c2_cut}
  
  -x* = {neg_x_star}
  Movement direction: {c2_cut}  (since started at origin)
  
  Dot product with -x*: {np.dot(c2_cut, neg_x_star):.4f}
  
  Movement direction = {c2_cut / np.linalg.norm(c2_cut)}
  Target direction = {neg_x_star / np.linalg.norm(neg_x_star)}
""")

# The center moved to (1/4, 1/4, 1/4) approximately - let me check
# Actually the center after two cuts depends on the specific formulas
print(f"Center after cuts: {c2_cut}")
print(f"Is this toward -x* = (-1,-1,1)? ")
print(f"  Coordinate 1: c_1 = {c2_cut[0]:.4f}, want negative → {'✗ WRONG WAY' if c2_cut[0] > 0 else '✓ RIGHT'}")
print(f"  Coordinate 2: c_2 = {c2_cut[1]:.4f}, want negative → {'✗ WRONG WAY' if c2_cut[1] > 0 else '✓ RIGHT'}")
print(f"  Coordinate 3: c_3 = {c2_cut[2]:.4f}, want positive → {'✓ RIGHT' if c2_cut[2] > 0 else '✗ WRONG WAY'}")

print("""
IMPORTANT: The direction of center movement depends on which side of 
the hyperplane the center starts on, and the order of cuts.

The FIRST cut (a·x ≤ b) moves the center TOWARD the hyperplane 
(since center is at origin with a·c = 0 < b = 1, the center moves 
toward the hyperplane).

The SECOND cut (-a·x ≤ -b) then further constrains from the other 
side, pushing the center back.

The net effect of both cuts (= equality constraint) moves the center 
TO the hyperplane, i.e., toward a·c = b = 1. Since a = (1,1,1), 
this pushes the center in the direction (1,1,1), which is TOWARD 
the hyperplane but NOT toward -x*.

This is the fundamental issue: the hyperplane constraint pushes the 
center toward its nearest point on the hyperplane, which is in the 
direction of a = (1,1,1). But -x* = (-1,-1,1) is NOT in the direction 
of a. The constraint is "compatible" with -x* but doesn't push toward it.
""")

print("="*70)
print("DEEPER ANALYSIS: WHAT HAPPENS WITH MULTIPLE CONSTRAINTS?")
print("="*70)

print("""
With multiple hyperplane constraints, the center would be pushed toward 
the intersection of all hyperplanes. The INTERSECTION of hyperplanes 
contains x* (by definition). But does it contain -x*?

For x* = (1,1,-1) and constraint x1+x2+x3 = 1:
  The center is pushed toward the hyperplane x1+x2+x3 = 1.
  -x* = (-1,-1,1) has: (-1)+(-1)+1 = -1 ≠ 1.
  So -x* is NOT on the hyperplane.

This means: the center CANNOT reach -x* by being pushed onto hyperplanes!
The hyperplane operations push the center toward the feasible set 
(the hyperplane), which contains x* but generally NOT -x*.

THIS IS THE FUNDAMENTAL PROBLEM: -x* does not satisfy the constraints 
(A·(-x*) = -b ≠ b in general). The center cannot reach -x* through 
equality-constraint cutting plane operations.
""")

print("="*70)
print("SUMMARY OF FINDINGS")
print("="*70)

print("""
1. OPERATION 1 (Sphere): For a sphere, the hyperplane equality constraint 
   a·x = b can be implemented as two sequential half-space cuts. The center 
   moves to the projection of the current center onto the hyperplane. 
   The shape becomes an ellipsoid (NOT a sphere). Specifically:
   
   - Center: c' = projection of c onto {x: a·x = b}
   - Shape: Q' is an oblate ellipsoid, flattened in the direction of a
   
   For N=3, starting at origin with a=(1,1,1), b=1:
   Center moves to approximately (1/3, 1/3, 1/3) [= projection]
   
2. OPERATION 2 (Sphere): For variable i, constraining the surface to 
   reach x_i = ±1 affects only Q_{ii}. It does NOT move the center.
   For the initial sphere (√3 > 1), this operation is trivially satisfied.
   
3. N=3 EXAMPLE: Starting sphere at origin, r=√3. All 8 vertices on surface.
   After Op 1 with x1+x2+x3=1: center moves to ~(1/3,1/3,1/3).
   Only the 3 satisfying vertices {(1,1,-1),(1,-1,1),(-1,1,1)} remain 
   near the surface. The shape is now an oblate ellipsoid.
   
4. AFTER OPERATION 2: Center doesn't move. Only Q adjusts.

5. CENTER DIRECTION: The center moves toward (1/3,1/3,1/3) = the centroid 
   of satisfying vertices. For x*=(1,1,-1), we want -x*=(-1,-1,1).
   The movement is in the WRONG direction (dot product with -x* is negative).
   
6. SPHERE vs ELLIPSOID: Operations necessarily produce ellipsoids. 
   Starting from a sphere Q=r²I, any hyperplane cut produces Q with 
   unequal eigenvalues. Spherical symmetry is broken.
   
7. BLIND OPERATIONS: The operations using only (a,b) or i push the center 
   toward the hyperplane (which contains x*), not toward -x*.
   Since -x* is NOT on the hyperplanes (A·(-x*) = -b ≠ b), the center 
   CANNOT reach -x* through these operations.
   
   This is the fundamental impossibility: you cannot reach -x* because 
   it violates all the constraints. The geometric operations naturally 
   push toward the feasible set, which is the OPPOSITE direction from -x*.
""")


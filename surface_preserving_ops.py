"""
Surface-preserving ellipsoid operation (Op1 — hyperplane).

Construction: pencil of ellipsoids preserving E ∩ H.

  Given E_0 = { x : (x - c)^T P^{-1} (x - c) <= 1 }
  and hyperplane H = { x : a^T x = b }
  the pencil of ellipsoids parameterized by s is:

    g_s(x) := (x - c)^T P^{-1} (x - c) + s * (a^T x - b)^2

    E_s := { x : g_s(x) <= 1 }

For x in H, (a^T x - b) = 0, so g_s(x) = f_0(x). Hence E_s and E_0 share
the same intersection with H. In particular, every x in E_0 ∩ H is on the
surface of E_s for every s. So if x* is on the surface of E_0 AND x* is
on H, then x* is on the surface of E_s for every s. (I1 preserved.)

Algebraic conversion to standard form:

  g_s(x) = x^T M x - 2 (M c')^T x + (c'^T M c' - 1) + r^2_offset
         = (x - c')^T M (x - c') + (constant)

  where:
    M(s)  = P^{-1} + s * a a^T               (shape inverse)
    c'(s) = M(s)^{-1} (P^{-1} c + s b a)     (new center)

  The standard ellipsoid form (x - c')^T P'^{-1} (x - c') <= 1 has:
    P'^{-1}(s) = M(s) / R(s)
  where R(s) is chosen so that x in E_0 ∩ H satisfies the equation with
  RHS = 1.

  R(s) computed by closed form:
    Let v = P^{-1} c + s b a, then
    R(s) = c'^T M c' - c^T P^{-1} c - s b^2 + 1
         = v^T M^{-1} v - c^T P^{-1} c - s b^2 + 1

  Validity: requires M(s) PSD, i.e., 1 + s * (a^T P a) > 0 in our case.
  (Sherman-Morrison condition.)
"""

import numpy as np


def op_hyperplane_pencil(c, P_inv, a, b, s):
    """Apply pencil parameter s for hyperplane H: a^T x = b.

    Inputs:
        c       : (N,) center of current ellipsoid.
        P_inv   : (N, N) shape inverse, so ellipsoid is
                  { x : (x - c)^T P_inv (x - c) <= 1 }.
        a       : (N,) hyperplane normal.
        b       : scalar, hyperplane offset.
        s       : scalar pencil parameter. s = 0 is the identity.
                  Valid range: 1 + s * (a^T P a) > 0.

    Returns:
        c_new   : (N,) new center.
        P_inv_new : (N, N) new shape inverse.

    Property (I1): for every x with (x-c)^T P_inv (x-c) = 1 and a^T x = b,
    we have (x - c_new)^T P_inv_new (x - c_new) = 1.
    """
    P = np.linalg.inv(P_inv)
    aPa = float(a @ P @ a)
    if 1.0 + s * aPa <= 1e-12:
        raise ValueError(
            f"Pencil parameter out of valid range: 1 + s*aPa = {1.0 + s*aPa}"
        )

    # M = P_inv + s * a a^T
    M = P_inv + s * np.outer(a, a)

    # v = P_inv @ c + s * b * a, then c_new = M^{-1} v
    v = P_inv @ c + s * b * a
    c_new = np.linalg.solve(M, v)

    # R = (c_new^T M c_new) - (c^T P_inv c) - s b^2 + 1
    # This is the "scale" so that the unit-level set of g_s in standard form
    # becomes the unit ellipsoid.
    R = float(c_new @ M @ c_new - c @ P_inv @ c - s * b * b + 1.0)
    if R <= 1e-12:
        raise ValueError(
            f"Degenerate ellipsoid: R = {R} <= 0. Pencil parameter too extreme."
        )

    # P_inv_new = M / R (so that level set of (x-c_new)^T P_inv_new (x-c_new) = 1
    # corresponds to g_s(x) = 1)
    P_inv_new = M / R

    return c_new, P_inv_new


def op_two_plane_pencil(c, P_inv, i, t):
    """Apply pencil parameter t for the pair of planes x_i = +1 and x_i = -1.

    Construction: add t * (x_i^2 - 1) to the quadratic form.

      g_t(x) := (x - c)^T P_inv (x - c) + t * (x_i^2 - 1)
      E_t   := { x : g_t(x) <= 1 }

    For any x with x_i = +1 or x_i = -1, (x_i^2 - 1) = 0, so g_t(x) = f_0(x).
    Thus E_t and E_0 share the same intersection with the union of planes
    {x_i = +1} and {x_i = -1}. In particular, every ±1 vertex on the surface
    of E_0 stays on the surface of E_t. So if x* is on E_0's surface AND
    x*_i in {-1, +1}, then x* is on E_t's surface for every valid t.

    Note: this works simultaneously for x* and -x* (both have x*_i^2 = 1),
    so the same operation preserves both invariants — no "paired" form
    needed for box constraints.

    Algebra:
      g_t(x) = x^T (P_inv + t e_i e_i^T) x - 2 c^T P_inv x
                                         + (c^T P_inv c - t)
             = (x - c')^T M (x - c') + (constant)
      with M = P_inv + t e_i e_i^T and c' = M^{-1} P_inv c.

      Standard form (x - c')^T P_inv_new (x - c') = 1 has
      P_inv_new = M / R where
        R = 1 - c^T P_inv c + t + c'^T M c'.

    Inputs:
        c      : (N,) center.
        P_inv  : (N, N) shape inverse.
        i      : integer index, 0 <= i < N.
        t      : scalar pencil parameter. t = 0 is identity.
                 Valid range: t > -P_inv[i, i].

    Returns:
        c_new, P_inv_new
    """
    N = len(c)
    if not (0 <= i < N):
        raise ValueError(f"index out of range: {i}")
    pii = float(P_inv[i, i])
    if t <= -pii + 1e-12:
        raise ValueError(
            f"Pencil parameter out of valid range: t={t} must be > -P_inv[{i},{i}]={-pii}"
        )

    # M = P_inv + t * e_i e_i^T
    M = P_inv.copy()
    M[i, i] = M[i, i] + t

    # New center: M c' = P_inv c (linear term unchanged in g_t)
    rhs = P_inv @ c
    c_new = np.linalg.solve(M, rhs)

    # R = 1 - c^T P_inv c + t + c'^T M c'
    R = float(1.0 - c @ P_inv @ c + t + c_new @ M @ c_new)
    if R <= 1e-12:
        raise ValueError(
            f"Degenerate ellipsoid: R = {R} <= 0. Pencil parameter too extreme."
        )

    P_inv_new = M / R
    return c_new, P_inv_new


def on_surface(x, c, P_inv, tol=1e-10):
    """Check whether x is on the ellipsoid surface (within tolerance)."""
    val = float((x - c) @ P_inv @ (x - c))
    return abs(val - 1.0) < tol, val

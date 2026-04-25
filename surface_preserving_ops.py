"""
Surface-preserving ellipsoid operations -- axis-aligned form.

The dual-ellipsoid descent maintains two axis-aligned ellipsoids E_1
and E_2 sharing a single center c:

    E_i = { x : sum_j A_i[j] * (x_j - c[j])^2 = R_i_sq }

E_1 carries x* on its surface; E_2 carries -x*. The operations below
preserve the relevant invariants (I1: x* on E_1, I2: -x* on E_2) and
preserve shared c across both ellipsoids.

Axis-aligned representation: each ellipsoid is described by
    c       : (N,)   shared center
    A_i     : (N,)   per-axis stiffness, A_i[j] > 0
    R_i_sq  : scalar > 0
The shape inverse is diag(A_i / R_i_sq); the i-th axis half-length is
sqrt(R_i_sq / A_i[i]).

Two operations:

  Op1 (hyperplane H_k = {x : x_k = b}):
      Adds s * (x_k - b)^2 to the form. Vanishes on H_k, so any point
      satisfying x_k = b that was on E_0's surface stays on the new
      surface. For x* (a +/-1 vertex), this means b must be in {-1, +1}
      and must match x*'s sign at coordinate k. Op1 commits to a sign
      hypothesis for x*_k -- if the hypothesis is wrong, I1 is broken.
      Paired version: E_1 uses x_k = b, E_2 uses x_k = -b (since
      -x*_k = -x*_k).

  Op2 (two parallel planes x_k = +1, x_k = -1):
      Adds M * (x_k^2 - 1) to the form. Vanishes on x_k = +/-1, so any
      vertex with x*_k^2 = 1 stays on the new surface. Preserves both
      x* and -x* simultaneously without committing to a sign.
      Important: Op2 cannot move c[k] when c[k] = 0 (the formula
      newC[k] = A[k]*c[k]/(A[k]+M) returns 0 for any M when c[k] = 0).
      Use Op1 to break the c[k] = 0 symmetry first.

Why axis-aligned and not general:
  In axis-aligned form, the i-th column of the shape matrix P is
  parallel to e_i (since P is diagonal). So the natural pencil motion
  -- which moves c along the i-th column of P -- is automatically
  along e_i. That means a single shared parameter (delta) on both
  ellipsoids in the paired version produces a single shared new
  center, so shared c is automatic. In general (non-axis-aligned)
  ellipsoids, P_:i has off-diagonal components and the same parameter
  on each side moves c in different directions, breaking shared c.

  The cost of axis-alignment: the pencil hyperplane in Op1 must be
  axis-aligned (a = e_k). Sparse but multi-coordinate constraint
  hyperplanes from the original Ax=b system (a row of A with several
  +/-1 entries) cannot be encoded as a single Op1 step. Those
  constraints have to enter the descent through the fitness function
  or through a different operation, not through the per-step
  ellipsoid update.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DualEllipsoidState:
    """Dual-ellipsoid state: two axis-aligned ellipsoids sharing one center.

        E_1 = { x : sum_j A1[j] (x_j - c[j])^2 = R1_sq }
        E_2 = { x : sum_j A2[j] (x_j - c[j])^2 = R2_sq }

    E_1's surface contains x*; E_2's surface contains -x*. The shared
    center c is the descent's primary degree of freedom -- the goal of
    the algorithm is to drive c to the segment between x* and -x*, from
    which sign extraction recovers x*.
    """

    c: np.ndarray
    A1: np.ndarray
    R1_sq: float
    A2: np.ndarray
    R2_sq: float

    @classmethod
    def initial_sphere(cls, N):
        """Initial state: both ellipsoids = sphere of radius sqrt(N) at origin.

        Every +/-1 vertex lies on this sphere's surface, so I1 (x* on E_1)
        and I2 (-x* on E_2) hold automatically regardless of which vertex
        is x*.
        """
        return cls(
            c=np.zeros(N),
            A1=np.ones(N),
            R1_sq=float(N),
            A2=np.ones(N),
            R2_sq=float(N),
        )

    @property
    def N(self):
        return len(self.c)

    def x_on_E1(self, x, tol=1e-10):
        """Check whether x is on E_1's surface."""
        return on_surface(x, self.c, self.A1, self.R1_sq, tol)

    def x_on_E2(self, x, tol=1e-10):
        """Check whether x is on E_2's surface."""
        return on_surface(x, self.c, self.A2, self.R2_sq, tol)

    def valid_range_hyperplane(self, k, b):
        """(delta_lo, delta_hi) for paired Op1 on coordinate k with offset b."""
        return valid_delta_range_op_hyperplane(self.c, k, b)

    def valid_range_two_plane(self, k):
        """(delta_lo, delta_hi) for paired Op2 on coordinate k."""
        return valid_delta_range_op_two_plane(self.c, k)


def on_surface(x, c, A, R_sq, tol=1e-10):
    """Check whether x lies on the surface of the axis-aligned ellipsoid.

    Surface: sum_j A[j] (x_j - c[j])^2 = R_sq.
    """
    val = float(np.sum(A * (x - c) ** 2))
    return abs(val - R_sq) < tol, val


# ---------------------------------------------------------------------------
# Op1: axis-aligned hyperplane H_k = {x : x_k = b}
# ---------------------------------------------------------------------------
#
# Pencil: q(x) = f_0(x) + s * (x_k - b)^2.
# The added term vanishes on H_k, so any x in E_0 ∩ H_k stays at level R_sq.
#
# Re-completing the square in x_k:
#   newA[k]   = A[k] + s
#   newC[k]   = (A[k] * c[k] + s * b) / (A[k] + s)
#   newR_sq   = R_sq - s * A[k] * (c[k] - b)^2 / (A[k] + s)
#
# The other coordinates' A and c entries are unchanged, so the
# representation stays axis-aligned.
#
# Inverting newC[k] = c[k] + delta to solve for s:
#   s = A[k] * delta / (b - c[k] - delta)
#
# Validity:
#   - denominator b - c[k] - delta != 0 (else delta would push the new
#     center exactly onto the hyperplane, and s diverges),
#   - newA[k] > 0,
#   - newR_sq > 0.


def op_hyperplane(c, A, R_sq, k, b, delta):
    """Apply Op1 (axis-aligned hyperplane x_k = b) to translate c[k] by delta.

    Inputs:
        c      : (N,) center.
        A      : (N,) per-axis stiffness.
        R_sq   : scalar.
        k      : coordinate index.
        b      : hyperplane offset (must equal x*_k for I1 to be preserved;
                 typically +1 or -1 since x* is a +/-1 vertex).
        delta  : signed motion of c[k]. delta = 0 is identity.

    Returns:
        c_new, A_new, R_sq_new.
    """
    bc = b - c[k]
    denom = bc - delta
    if abs(denom) < 1e-14:
        raise ValueError(
            f"Degenerate: b - c[k] - delta = {denom} ~ 0. "
            f"delta would push c[k] onto the hyperplane."
        )

    s = A[k] * delta / denom

    new_A_k = A[k] + s
    if new_A_k <= 1e-14:
        raise ValueError(
            f"Degenerate: A[k] + s = {new_A_k} <= 0. "
            f"delta out of valid range."
        )

    new_R_sq = R_sq - s * A[k] * bc * bc / new_A_k
    if new_R_sq <= 1e-14:
        raise ValueError(
            f"Degenerate: R_sq_new = {new_R_sq} <= 0."
        )

    c_new = c.copy()
    c_new[k] = c[k] + delta
    A_new = A.copy()
    A_new[k] = new_A_k
    return c_new, A_new, new_R_sq


def paired_op_hyperplane(state, k, b, delta):
    """Paired Op1: hyperplane x_k = b on E_1, x_k = -b on E_2; shared c update.

    Both sides translate c[k] by delta; each side computes its own s
    internally to achieve that motion under its own (A_i, R_i_sq).

    Side 1 preserves x* (when x*_k = b). Side 2 preserves -x* (when
    -x*_k = -b, i.e., x*_k = b, same condition). So this paired op
    commits to the hypothesis x*_k = b. If x*_k turns out to be -b,
    both invariants break.

    Inputs:
        state : DualEllipsoidState.
        k     : coordinate index.
        b     : hyperplane offset for E_1 (E_2 uses -b). Typically +/-1.
        delta : signed motion of c[k]; must lie in
                state.valid_range_hyperplane(k, b).

    Returns:
        DualEllipsoidState (new instance).
    """
    c1, A1_new, R1_sq_new = op_hyperplane(
        state.c, state.A1, state.R1_sq, k, b, delta
    )
    c2, A2_new, R2_sq_new = op_hyperplane(
        state.c, state.A2, state.R2_sq, k, -b, delta
    )
    assert np.allclose(c1, c2, atol=1e-12), (
        "paired_op_hyperplane: shared c violated"
    )
    return DualEllipsoidState(c1, A1_new, R1_sq_new, A2_new, R2_sq_new)


# ---------------------------------------------------------------------------
# Op2: two parallel planes x_k = +1, x_k = -1 (the box constraint x_k^2 = 1)
# ---------------------------------------------------------------------------
#
# Pencil: q(x) = f_0(x) + M * (x_k^2 - 1).
# Vanishes on x_k = +1 AND x_k = -1, so any vertex with x_k^2 = 1
# (in particular both x* and -x*) stays at level R_sq.
#
# Re-completing the square:
#   newA[k]   = A[k] + M
#   newC[k]   = A[k] * c[k] / (A[k] + M)
#   newR_sq   = R_sq + M - A[k] * c[k]^2 + A[k]^2 * c[k]^2 / (A[k] + M)
#
# Inverting newC[k] = c[k] + delta to solve for M:
#   M = -A[k] * delta / (c[k] + delta)
#
# Validity:
#   - c[k] + delta != 0 (else M diverges),
#   - if c[k] = 0 then delta must be 0 (formula gives newC[k] = 0 always),
#   - newA[k] > 0,
#   - newR_sq > 0.


def op_two_plane(c, A, R_sq, k, delta):
    """Apply Op2 (two-plane x_k^2 = 1 pencil) to translate c[k] by delta.

    Inputs:
        c      : (N,) center.
        A      : (N,) per-axis stiffness.
        R_sq   : scalar.
        k      : coordinate index.
        delta  : signed motion of c[k]. Must be 0 if c[k] = 0.

    Returns:
        c_new, A_new, R_sq_new.
    """
    if abs(delta) < 1e-14:
        return c.copy(), A.copy(), R_sq

    if abs(c[k]) < 1e-14:
        raise ValueError(
            f"Op2: c[k] = 0 and delta != 0; Op2 cannot move c[k] from "
            f"the origin. Use Op1 to break the symmetry first."
        )

    new_c_k = c[k] + delta
    if abs(new_c_k) < 1e-14:
        raise ValueError(
            f"Op2: new c[k] = {new_c_k} ~ 0 forces M -> infinity."
        )

    M = -A[k] * delta / new_c_k

    new_A_k = A[k] + M
    if new_A_k <= 1e-14:
        raise ValueError(
            f"Degenerate: A[k] + M = {new_A_k} <= 0. delta out of range."
        )

    new_R_sq = (
        R_sq
        + M
        - A[k] * c[k] * c[k]
        + A[k] * A[k] * c[k] * c[k] / new_A_k
    )
    if new_R_sq <= 1e-14:
        raise ValueError(
            f"Degenerate: R_sq_new = {new_R_sq} <= 0."
        )

    c_new = c.copy()
    c_new[k] = new_c_k
    A_new = A.copy()
    A_new[k] = new_A_k
    return c_new, A_new, new_R_sq


def paired_op_two_plane(state, k, delta):
    """Paired Op2: apply two-plane pencil to BOTH ellipsoids with shared delta.

    Both sides preserve their respective invariants (the pencil
    vanishes on x_k = +/-1, so it preserves any +/-1 vertex on the
    surface, including both x* and -x*). Each side computes its own
    M_i internally to achieve the shared delta motion of c[k].

    Inputs:
        state : DualEllipsoidState.
        k     : coordinate index.
        delta : signed motion of c[k]; must lie in
                state.valid_range_two_plane(k). delta = 0 if c[k] = 0.

    Returns:
        DualEllipsoidState (new instance).
    """
    c1, A1_new, R1_sq_new = op_two_plane(state.c, state.A1, state.R1_sq, k, delta)
    c2, A2_new, R2_sq_new = op_two_plane(state.c, state.A2, state.R2_sq, k, delta)
    assert np.allclose(c1, c2, atol=1e-12), (
        "paired_op_two_plane: shared c violated"
    )
    return DualEllipsoidState(c1, A1_new, R1_sq_new, A2_new, R2_sq_new)


# ---------------------------------------------------------------------------
# Valid-range helpers
# ---------------------------------------------------------------------------
#
# These return the (delta_lo, delta_hi) range for which the paired ops
# are valid given the current state. The descent picks step sizes
# within this range.
#
# Op1 paired with hyperplanes x_k = +b on E_1 and x_k = -b on E_2:
#   The geometric walls are at c_new[k] = b (E_1 hits its hyperplane)
#   and c_new[k] = -b (E_2 hits its hyperplane). So
#       c_new[k] in (-1, 1)   for b in {-1, +1}
#       delta in (-1 - c[k], 1 - c[k]).
#   The R_sq > 0 constraint is dominated by the geometric wall when x*
#   is on E_1's surface and -x* on E_2's (proof: R_sq >= A * (1±c)^2,
#   which makes the R_sq wall further out than the geometric one).
#
# Op2 paired:
#   Op2 forces c_new[k] to keep the same sign as c[k]. The wall is at
#   c_new[k] = 0, with no upper magnitude bound (soft degeneracy as
#   |delta| -> infinity, where A_new[k] -> 0). Range:
#       c[k] > 0:  delta in (-c[k], +inf)
#       c[k] < 0:  delta in (-inf, -c[k])
#       c[k] = 0:  delta = 0 (no motion)


def valid_delta_range_op_hyperplane(c, k, b):
    """Valid (delta_lo, delta_hi) for `paired_op_hyperplane(..., k, b, delta)`.

    Range is bounded: c_new[k] must stay in (-1, 1). Independent of A_i
    and R_i_sq when x* on E_1 and -x* on E_2 (the geometric walls are
    tighter than the R_sq > 0 constraint in that regime).

    Inputs:
        c   : (N,) shared center.
        k   : coordinate index.
        b   : hyperplane offset for E_1 (E_2 uses -b). Must be +/-1 for
              physical meaning, though the formula works for any b.

    Returns:
        (delta_lo, delta_hi), an open interval. delta = 0 is always inside.
    """
    wall_1 = b - c[k]
    wall_2 = -b - c[k]
    return (wall_2, wall_1) if wall_2 < wall_1 else (wall_1, wall_2)


def valid_delta_range_op_two_plane(c, k):
    """Valid (delta_lo, delta_hi) for `paired_op_two_plane(..., k, delta)`.

    Range is one-sided unbounded: c_new[k] must keep the same sign as
    c[k]. If c[k] = 0, no motion is possible (returns (0.0, 0.0)).

    Inputs:
        c   : (N,) shared center.
        k   : coordinate index.

    Returns:
        (delta_lo, delta_hi). One endpoint is +/-inf for c[k] != 0;
        both are 0 for c[k] = 0.
    """
    if abs(c[k]) < 1e-14:
        return (0.0, 0.0)
    if c[k] > 0:
        return (-c[k], float("inf"))
    return (float("-inf"), -c[k])

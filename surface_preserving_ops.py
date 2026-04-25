"""
Surface-preserving ellipsoid operations -- axis-aligned form, with
shared center AND shared per-axis stiffness across the dual ellipsoids.

The dual-ellipsoid descent maintains two axis-aligned ellipsoids E_1
and E_2 with shared center c AND shared per-axis stiffness vector A:

    E_i = { x : sum_j A[j] (x_j - c[j])^2 = R_i_sq }

E_1's surface contains x*; E_2's surface contains -x*. The two
ellipsoids differ only in the scalars R_1_sq and R_2_sq.

Two operations:

  Op1 (general hyperplane H = {x : a^T x = b}):
      Linear pencil M * (a^T x - b). Vanishes at x* (since a^T x* = b)
      and at -x* (when paired with offset -b on E_2). Adds NO quadratic
      part, so axis-alignment is preserved even when a has multiple
      non-zero entries -- this is what makes general sparse hyperplane
      normals work.

      Updates:
          newA   = A                                      (unchanged)
          newC   = C - M * a / (2 * A)                    (componentwise)
          newR_1_sq = R_1_sq + M*b + Sum A*(newC^2 - C^2)
          newR_2_sq = R_2_sq - M*b + Sum A*(newC^2 - C^2)

      The two ellipsoids' R values diverge by 2*M*b per Op1 paired step.

      Validity: for x* on E_1 and -x* on E_2, Cauchy-Schwarz gives
      (b - a^T C)^2 <= R_i_sq * Sum a^2/A, which is exactly the
      discriminant of newR_i_sq(M) = R_i_sq + M*(b' - a^T C) + M^2 *
      Sum a^2 / (4A) being non-positive. So newR_i_sq > 0 for all M.
      M is unbounded.

  Op2 (two parallel planes x_k = +1, x_k = -1):
      Quadratic pencil M * (x_k^2 - 1). Vanishes on x_k = +/-1, so
      preserves both x* and -x*. The added term affects only the (k,k)
      diagonal entry of the shape matrix, so axis-alignment is
      preserved. Both ellipsoids update A[k] identically (same M when
      A is shared and delta is shared), so shared A is preserved.

      Updates (paired, shared delta):
          M = -A[k] * delta / (c[k] + delta)
          newA[k]   = A[k] + M = A[k] * c[k] / (c[k] + delta)
          newC[k]   = c[k] + delta
          newR_i_sq = R_i_sq + M - A[k]*c[k]^2 + A[k]^2*c[k]^2/(A[k]+M)
                      (same on both sides; R_1_sq - R_2_sq unchanged)

      Validity: c[k] + delta must keep the same sign as c[k];
      Op2 cannot move c[k] from c[k] = 0.

Together, Op1 handles all m sparse multi-coord constraint hyperplanes
from the original Ax=b system, AND Op2 handles the box constraints
x_k^2 = 1.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DualEllipsoidState:
    """Dual-ellipsoid state with shared center c AND shared axes A.

        E_i = { x : sum_j A[j] * (x_j - c[j])^2 = R_i_sq }

    E_1's surface contains x*; E_2's surface contains -x*.
    """

    c: np.ndarray
    A: np.ndarray
    R1_sq: float
    R2_sq: float

    @classmethod
    def initial_sphere(cls, N):
        """Initial state: both ellipsoids = sphere of radius sqrt(N) at origin.

        Every +/-1 vertex lies on this sphere, so I1 and I2 hold for any
        x* in {-1, +1}^N.
        """
        return cls(
            c=np.zeros(N),
            A=np.ones(N),
            R1_sq=float(N),
            R2_sq=float(N),
        )

    @property
    def N(self):
        return len(self.c)

    def x_on_E1(self, x, tol=1e-10):
        return on_surface(x, self.c, self.A, self.R1_sq, tol)

    def x_on_E2(self, x, tol=1e-10):
        return on_surface(x, self.c, self.A, self.R2_sq, tol)

    def valid_range_two_plane(self, k):
        return valid_delta_range_op_two_plane(self.c, k)


def on_surface(x, c, A, R_sq, tol=1e-10):
    """Check whether x lies on the axis-aligned ellipsoid surface."""
    val = float(np.sum(A * (x - c) ** 2))
    return abs(val - R_sq) < tol, val


# ---------------------------------------------------------------------------
# Op1 -- linear pencil, general sparse hyperplane H = {x : a^T x = b}
# ---------------------------------------------------------------------------


def op_hyperplane(c, A, R_sq, a, b, M):
    """Single-ellipsoid Op1: linear pencil M*(a^T x - b).

    For any x with a^T x = b on the original surface, x stays on the new
    surface. In particular x* (which satisfies a^T x* = b) is preserved.

    Inputs:
        c     : (N,) center.
        A     : (N,) per-axis stiffness, all positive.
        R_sq  : scalar (RHS of ellipsoid equation).
        a     : (N,) hyperplane normal (any pattern, sparse OK).
        b     : scalar hyperplane offset.
        M     : scalar pencil parameter. M = 0 is identity. Unbounded.

    Returns:
        c_new, A_new (= A, unchanged), R_sq_new.
    """
    a = np.asarray(a, dtype=float)
    c_new = c - M * a / (2.0 * A)
    delta_R = float(np.sum(A * (c_new ** 2 - c ** 2)))
    R_sq_new = R_sq + M * b + delta_R
    if R_sq_new <= 1e-14:
        raise ValueError(
            f"Degenerate: R_sq_new = {R_sq_new} <= 0. (Should not happen for "
            f"x* on the original surface; check inputs.)"
        )
    return c_new, A.copy(), R_sq_new


def paired_op_hyperplane(state, a, b, M):
    """Paired Op1: linear pencil with offset b on E_1 and -b on E_2.

    Preserves I1 (when a^T x* = b) and I2 (since a^T(-x*) = -b). The
    shared c is automatic because newC = c - M*a/(2*A) uses only the
    shared A. R_1_sq and R_2_sq diverge by 2*M*b per call.

    Inputs:
        state : DualEllipsoidState.
        a     : (N,) hyperplane normal.
        b     : scalar offset for E_1 (E_2 uses -b).
        M     : scalar pencil parameter (unbounded; identity at 0).

    Returns:
        DualEllipsoidState.
    """
    a = np.asarray(a, dtype=float)
    c_new = state.c - M * a / (2.0 * state.A)
    delta_R = float(np.sum(state.A * (c_new ** 2 - state.c ** 2)))
    R1_sq_new = state.R1_sq + M * b + delta_R
    R2_sq_new = state.R2_sq - M * b + delta_R
    if R1_sq_new <= 1e-14 or R2_sq_new <= 1e-14:
        raise ValueError(
            f"Degenerate: R1_sq_new={R1_sq_new}, R2_sq_new={R2_sq_new}. "
            f"(Should not happen for invariants holding; check inputs.)"
        )
    return DualEllipsoidState(c_new, state.A.copy(), R1_sq_new, R2_sq_new)


# ---------------------------------------------------------------------------
# Op2 -- two parallel planes x_k = +/-1
# ---------------------------------------------------------------------------


def op_two_plane(c, A, R_sq, k, delta):
    """Single-ellipsoid Op2: quadratic pencil M*(x_k^2 - 1).

    Preserves any vertex with x_k^2 = 1 on the surface, including both
    x* and -x*. Cannot move c[k] from c[k] = 0.

    Inputs:
        c, A, R_sq : ellipsoid state.
        k          : coordinate index.
        delta      : signed motion of c[k]; must be 0 if c[k] = 0.

    Returns:
        c_new, A_new, R_sq_new.
    """
    if abs(delta) < 1e-14:
        return c.copy(), A.copy(), R_sq
    if abs(c[k]) < 1e-14:
        raise ValueError(
            f"Op2: c[k] = 0 and delta != 0; cannot move from origin. "
            f"Use Op1 to break symmetry first."
        )
    new_c_k = c[k] + delta
    if abs(new_c_k) < 1e-14:
        raise ValueError(f"Op2: new c[k] = {new_c_k} ~ 0 forces M -> infinity.")
    M = -A[k] * delta / new_c_k
    new_A_k = A[k] + M
    if new_A_k <= 1e-14:
        raise ValueError(f"Degenerate: A[k] + M = {new_A_k} <= 0.")
    R_sq_new = (
        R_sq + M - A[k] * c[k] ** 2 + A[k] ** 2 * c[k] ** 2 / new_A_k
    )
    if R_sq_new <= 1e-14:
        raise ValueError(f"Degenerate: R_sq_new = {R_sq_new} <= 0.")
    c_new = c.copy(); c_new[k] = new_c_k
    A_new = A.copy(); A_new[k] = new_A_k
    return c_new, A_new, R_sq_new


def paired_op_two_plane(state, k, delta):
    """Paired Op2: same delta on both ellipsoids; shared (c, A) preserved.

    Both sides preserve their respective invariants (the pencil
    vanishes on x_k = +/-1). With shared A, both sides' updates to A[k]
    are identical, so shared A is preserved. R_1_sq - R_2_sq is
    unchanged by this op.

    Inputs:
        state : DualEllipsoidState.
        k     : coordinate index.
        delta : signed motion of c[k]. Must be 0 if c[k] = 0; valid
                range is state.valid_range_two_plane(k).

    Returns:
        DualEllipsoidState.
    """
    if abs(delta) < 1e-14:
        return DualEllipsoidState(
            state.c.copy(), state.A.copy(), state.R1_sq, state.R2_sq
        )
    if abs(state.c[k]) < 1e-14:
        raise ValueError(
            f"Op2: c[k] = 0 and delta != 0; cannot move from origin. "
            f"Use Op1 to break symmetry first."
        )
    new_c_k = state.c[k] + delta
    if abs(new_c_k) < 1e-14:
        raise ValueError(f"Op2: new c[k] = {new_c_k} ~ 0 forces M -> infinity.")
    M = -state.A[k] * delta / new_c_k
    new_A_k = state.A[k] + M
    if new_A_k <= 1e-14:
        raise ValueError(f"Degenerate: A[k] + M = {new_A_k} <= 0.")

    delta_R = (
        M
        - state.A[k] * state.c[k] ** 2
        + state.A[k] ** 2 * state.c[k] ** 2 / new_A_k
    )
    R1_sq_new = state.R1_sq + delta_R
    R2_sq_new = state.R2_sq + delta_R
    if R1_sq_new <= 1e-14 or R2_sq_new <= 1e-14:
        raise ValueError(
            f"Degenerate: R1_sq_new={R1_sq_new}, R2_sq_new={R2_sq_new}."
        )

    c_new = state.c.copy(); c_new[k] = new_c_k
    A_new = state.A.copy(); A_new[k] = new_A_k
    return DualEllipsoidState(c_new, A_new, R1_sq_new, R2_sq_new)


# ---------------------------------------------------------------------------
# Valid-range helpers
# ---------------------------------------------------------------------------


def valid_delta_range_op_two_plane(c, k):
    """Valid (delta_lo, delta_hi) for paired_op_two_plane on coordinate k.

    Op2 cannot move c[k] across zero (the formula's wall is at
    c_new[k] = 0). For c[k] = 0, no motion possible.
    """
    if abs(c[k]) < 1e-14:
        return (0.0, 0.0)
    if c[k] > 0:
        return (-c[k], float("inf"))
    return (float("-inf"), -c[k])

"""
Dual-ellipsoid surface-preserving descent.

Implements the design in `design-journal.md` §9. Two ellipsoids E1, E2
evolve under the surface-preserving operations from `surface_preserving_ops.py`:

  - E1 = (c1, P_inv1) maintains x* on its surface (invariant I1).
  - E2 = (c2, P_inv2) maintains -x* on its surface (invariant I2).

Both invariants are preserved by every operation, regardless of parameter.
The descent searches over operation parameters to minimize a fitness:

    weights['size']  * (size(E1) + size(E2))
  + weights['sph']   * (sphericity_defect(E1) + sphericity_defect(E2))
  + weights['align'] * ||c1 - c2||^2

"size" = mean(eigvals(P)) is mean squared semi-axis. "sphericity defect"
= var(eigvals(P)) drives toward roundness. The 'align' term comes from
the original §9 "shared center" framing; empirically (§9.9) it conflicts
with shrinking E1 -> x* and E2 -> -x* (which want centers far apart),
so the recommended profile uses align=0.

Knobs available per descent round:
  - For each constraint (a_k, b_k): one Op1 parameter `s1_k` that updates E1
    using (a_k, b_k), and one `s2_k` that updates E2 using (a_k, -b_k).
  - For each coordinate i: one Op2 parameter `t1_i` for E1 and `t2_i` for E2.

The fitness is computed in closed form from (c, P_inv) for each ellipsoid;
no reference to x* anywhere in the inner loop. x* is only used by callers
for diagnostics.

Empirical status (small N, |b|=1 instances, planted unique solution):
  weights={size:5, sph:1, align:0}, kicked init, 30 rounds:
  - N=6,  m∈{4,5}: 20/20 instances solved (sign(c) = ±x*).
  - N=8,  m∈{5,6}: 10/10 instances solved.
  - N=10, m∈{6,7}: 10/10 instances solved.
  See test_dual_ellipsoid_descent.py for the test bed and design-journal
  §9.8 for the full table.
"""

import numpy as np

from surface_preserving_ops import op_hyperplane_pencil, op_two_plane_pencil


# ---------------------------------------------------------------------------
# Ellipsoid scalar measurements
# ---------------------------------------------------------------------------

def ellipsoid_size(P_inv):
    """Mean squared semi-axis = mean eigenvalue of P = trace(P) / N.

    Smaller is "tighter" ellipsoid. For a sphere of radius r, this equals r^2.

    Computed via eigvalsh of P_inv (more stable than inverting first when
    P_inv has small eigenvalues, i.e., the ellipsoid has long axes; and
    when P_inv is huge -- near concentration -- the small eigenvalues are
    the meaningful ones for size).
    """
    eigs_pinv = np.linalg.eigvalsh(P_inv)
    eigs_pinv = np.clip(eigs_pinv, 1e-30, None)
    return float(np.mean(1.0 / eigs_pinv))


def ellipsoid_sphericity_defect(P_inv):
    """Variance of P's eigenvalues (semi-axis squared lengths).

    Zero for a sphere; positive otherwise. Driving this to zero forces the
    ellipsoid toward sphericity.

    Computed from eigvalsh(P_inv) directly to avoid the np.linalg.inv step
    that fails when the ellipsoid has degenerated (near concentration).
    """
    eigs_pinv = np.linalg.eigvalsh(P_inv)
    eigs_pinv = np.clip(eigs_pinv, 1e-30, None)
    eigs_p = 1.0 / eigs_pinv
    return float(np.var(eigs_p))


def ellipsoid_max_radius_sq(P_inv):
    """Largest squared semi-axis = 1 / min eigenvalue of P_inv.

    This is the worst-case "how far from center to surface."
    """
    eigs_pinv = np.linalg.eigvalsh(P_inv)
    return float(1.0 / max(eigs_pinv[0], 1e-30))


def ellipsoid_concentrated(P_inv, threshold=1e-6):
    """Has the ellipsoid shrunk to (almost) a point?

    True if mean semi-axis-squared is below threshold. After this, the
    descent should stop -- the center is committed and further ops are
    numerically unstable.
    """
    return ellipsoid_size(P_inv) < threshold


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def fitness(c1, P_inv1, c2, P_inv2, weights):
    """Weighted-sum fitness over both ellipsoids.

    weights = dict with keys:
      'size'   : weight on (size(E1) + size(E2))
      'sph'    : weight on (sphericity_defect(E1) + sphericity_defect(E2))
      'align'  : weight on ||c1 - c2||^2

    Returns +inf if any eigendecomposition fails (degenerate ellipsoid).
    """
    try:
        f_size = ellipsoid_size(P_inv1) + ellipsoid_size(P_inv2)
        f_sph = ellipsoid_sphericity_defect(P_inv1) + ellipsoid_sphericity_defect(P_inv2)
    except np.linalg.LinAlgError:
        return np.inf
    f_align = float((c1 - c2) @ (c1 - c2))
    return (
        weights["size"] * f_size
        + weights["sph"] * f_sph
        + weights["align"] * f_align
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def init_sphere(N):
    """Sphere of radius sqrt(N) centered at origin.

    Both x* and -x* (any +/-1 vertex) are on the surface, since
    ||x*||^2 = N matches the sphere's squared radius.
    """
    c = np.zeros(N)
    P_inv = np.eye(N) / float(N)
    return c, P_inv


def init_kicked(N, A, b, rng, kick_size=0.2):
    """Two ellipsoids, each kicked once by a random constraint.

    E1 is kicked using a random (a_k, b_k) -> stays preserving x*.
    E2 is kicked using a random (a_k, -b_k) -> stays preserving -x*.
    The kicks are with opposite signs so the centers separate.
    Invariants I1 and I2 are exactly preserved by each kick.
    """
    c1, P_inv1 = init_sphere(N)
    c2, P_inv2 = init_sphere(N)

    m = len(b)
    k = int(rng.integers(0, m))
    a_k = A[k]
    b_k = float(b[k])

    # Pick s within validity envelope. P=N*I initially, so a^T P a = N * ||a||^2 = 3N
    # (a is 3-sparse with +/-1 entries). Bound |s| < 0.5 / (a^T P a) for safety.
    aPa = float(N * (a_k @ a_k))
    s_safe = kick_size / aPa
    c1, P_inv1 = op_hyperplane_pencil(c1, P_inv1, a_k, b_k, s=+s_safe)
    c2, P_inv2 = op_hyperplane_pencil(c2, P_inv2, a_k, -b_k, s=+s_safe)
    return c1, P_inv1, c2, P_inv2


# ---------------------------------------------------------------------------
# Knob update -- single-knob line search
# ---------------------------------------------------------------------------

def _validity_bound_op1(P_inv, a):
    """Returns (lo, hi) such that any s in (lo, hi) is safely valid for Op1.

    Op1 requires 1 + s*(a^T P a) > 0 strictly, i.e., s > -1/(a^T P a).
    We back off by factor 0.9 for numerical safety.
    """
    P = np.linalg.inv(P_inv)
    aPa = float(a @ P @ a)
    if aPa <= 1e-12:
        return -np.inf, np.inf
    lo = -0.9 / aPa
    hi = +5.0 / aPa  # arbitrary upper cap; fitness gradient handles direction
    return lo, hi


def _validity_bound_op2(P_inv, i):
    """Returns (lo, hi) for Op2: t > -P_inv[i,i] strictly."""
    pii = float(P_inv[i, i])
    if pii <= 1e-12:
        return -np.inf, np.inf
    lo = -0.9 * pii
    hi = +5.0 * pii
    return lo, hi


def _golden_section_min(f, a, b, tol=1e-5, max_iter=50):
    """Golden-section search for the minimum of unimodal f on [a, b]."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return 0.0, np.inf
    phi = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - phi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + phi * (b - a)
            f2 = f(x2)
    return (a + b) / 2.0, min(f1, f2)


def best_param_op1(c_target, P_inv_target, c_other, c_other_self, a, b_const,
                   weights, eval_state):
    """Line-search the best s for Op1 applied to ONE ellipsoid.

    eval_state: 1 or 2 indicating which ellipsoid we're updating; the other
    ellipsoid's state is fixed during this line search.

    Args:
        c_target, P_inv_target: state of the ellipsoid being updated.
        c_other, c_other_self: kept for fitness alignment computation.
            c_other is the OTHER ellipsoid's center (held fixed).
        a, b_const: the constraint to apply.
        weights: dict.
    """
    lo, hi = _validity_bound_op1(P_inv_target, a)

    def f_of_s(s):
        if s <= lo + 1e-9 or s >= hi - 1e-9:
            return np.inf
        try:
            c_new, P_inv_new = op_hyperplane_pencil(c_target, P_inv_target, a, b_const, s=s)
        except Exception:
            return np.inf
        if eval_state == 1:
            return fitness(c_new, P_inv_new, c_other, c_other_self, weights)
        else:
            return fitness(c_other, c_other_self, c_new, P_inv_new, weights)

    # Search a bit narrower than the validity bound for stability.
    a_search = lo * 0.95
    b_search = hi * 0.95
    s_best, f_best = _golden_section_min(f_of_s, a_search, b_search)
    f_zero = f_of_s(0.0)
    if f_zero <= f_best + 1e-12:
        return 0.0, f_zero
    return s_best, f_best


def best_param_op2(c_target, P_inv_target, c_other, P_inv_other, i,
                   weights, eval_state):
    """Line-search the best t for Op2 applied to ONE ellipsoid."""
    lo, hi = _validity_bound_op2(P_inv_target, i)

    def f_of_t(t):
        if t <= lo + 1e-9 or t >= hi - 1e-9:
            return np.inf
        try:
            c_new, P_inv_new = op_two_plane_pencil(c_target, P_inv_target, i, t=t)
        except Exception:
            return np.inf
        if eval_state == 1:
            return fitness(c_new, P_inv_new, c_other, P_inv_other, weights)
        else:
            return fitness(c_other, P_inv_other, c_new, P_inv_new, weights)

    a_search = lo * 0.95
    b_search = hi * 0.95
    t_best, f_best = _golden_section_min(f_of_t, a_search, b_search)
    f_zero = f_of_t(0.0)
    if f_zero <= f_best + 1e-12:
        return 0.0, f_zero
    return t_best, f_best


# ---------------------------------------------------------------------------
# Descent loop
# ---------------------------------------------------------------------------

def diagnostics(c1, P_inv1, c2, P_inv2, x_star=None):
    try:
        size1 = ellipsoid_size(P_inv1)
        size2 = ellipsoid_size(P_inv2)
        sph1 = ellipsoid_sphericity_defect(P_inv1)
        sph2 = ellipsoid_sphericity_defect(P_inv2)
    except np.linalg.LinAlgError:
        size1 = size2 = sph1 = sph2 = float("nan")
    out = {
        "size_E1": size1,
        "size_E2": size2,
        "sph_E1": sph1,
        "sph_E2": sph2,
        "align": float((c1 - c2) @ (c1 - c2)),
        "norm_c1": float(np.linalg.norm(c1)),
        "norm_c2": float(np.linalg.norm(c2)),
    }
    if x_star is not None:
        # I1, I2 residuals (should stay tiny if invariants hold)
        i1 = float((x_star - c1) @ P_inv1 @ (x_star - c1)) - 1.0
        i2 = float((-x_star - c2) @ P_inv2 @ (-x_star - c2)) - 1.0
        out["I1_resid"] = i1
        out["I2_resid"] = i2
        # Distance from c1 to x*, c2 to -x*
        out["dist_c1_xstar"] = float(np.linalg.norm(c1 - x_star))
        out["dist_c2_minusxstar"] = float(np.linalg.norm(c2 + x_star))
        # Sign agreement
        s1 = np.sign(c1)
        s2 = np.sign(c2)
        # Tie-break zeros to +1 for the comparison (not perfect; just for diagnostic)
        s1_resolved = np.where(s1 == 0, 1.0, s1)
        s2_resolved = np.where(s2 == 0, 1.0, s2)
        out["hamming_c1_to_xstar"] = int(np.sum(s1_resolved != x_star))
        out["hamming_c2_to_minusxstar"] = int(np.sum(s2_resolved != -x_star))
    return out


def descend(A, b, max_rounds=50, weights=None, init=None, rng=None,
            x_star=None, verbose=False):
    """Run coordinate-descent over Op1 and Op2 knobs.

    Each "round" cycles once through:
      - For each constraint k: best Op1 on E1 with (a_k, b_k); best Op1 on E2 with (a_k, -b_k).
      - For each coordinate i: best Op2 on E1 at i; best Op2 on E2 at i.

    Returns (c1, P_inv1, c2, P_inv2, history) where history is a list of dicts.
    """
    m, N = A.shape
    if weights is None:
        weights = {"size": 1.0, "sph": 1.0, "align": 0.5}
    if rng is None:
        rng = np.random.default_rng(0)

    if init == "kicked":
        c1, P_inv1, c2, P_inv2 = init_kicked(N, A, b, rng)
    else:
        c1, P_inv1 = init_sphere(N)
        c2, P_inv2 = init_sphere(N)

    history = [diagnostics(c1, P_inv1, c2, P_inv2, x_star=x_star)]
    history[0]["round"] = 0
    history[0]["fitness"] = fitness(c1, P_inv1, c2, P_inv2, weights)

    for r in range(1, max_rounds + 1):
        # Early stop: each ellipsoid concentrated to a point => committed
        try:
            done1 = ellipsoid_concentrated(P_inv1)
            done2 = ellipsoid_concentrated(P_inv2)
        except np.linalg.LinAlgError:
            done1 = True
            done2 = True
        if done1 and done2:
            if verbose:
                print(f"early stop at round {r}: both ellipsoids concentrated")
            break

        # Hyperplane sweep
        for k in range(m):
            a_k = A[k]
            b_k = float(b[k])
            # E1 update with (a_k, b_k)
            s1_best, _ = best_param_op1(c1, P_inv1, c2, P_inv2, a_k, b_k, weights, eval_state=1)
            if abs(s1_best) > 1e-9:
                try:
                    c1, P_inv1 = op_hyperplane_pencil(c1, P_inv1, a_k, b_k, s=s1_best)
                except Exception:
                    pass
            # E2 update with (a_k, -b_k)
            s2_best, _ = best_param_op1(c2, P_inv2, c1, P_inv1, a_k, -b_k, weights, eval_state=2)
            if abs(s2_best) > 1e-9:
                try:
                    c2, P_inv2 = op_hyperplane_pencil(c2, P_inv2, a_k, -b_k, s=s2_best)
                except Exception:
                    pass

        # Coordinate sweep
        for i in range(N):
            t1_best, _ = best_param_op2(c1, P_inv1, c2, P_inv2, i, weights, eval_state=1)
            if abs(t1_best) > 1e-9:
                try:
                    c1, P_inv1 = op_two_plane_pencil(c1, P_inv1, i, t=t1_best)
                except Exception:
                    pass
            t2_best, _ = best_param_op2(c2, P_inv2, c1, P_inv1, i, weights, eval_state=2)
            if abs(t2_best) > 1e-9:
                try:
                    c2, P_inv2 = op_two_plane_pencil(c2, P_inv2, i, t=t2_best)
                except Exception:
                    pass

        d = diagnostics(c1, P_inv1, c2, P_inv2, x_star=x_star)
        d["round"] = r
        d["fitness"] = fitness(c1, P_inv1, c2, P_inv2, weights)
        history.append(d)
        if verbose:
            tags = []
            for key in ("size_E1", "size_E2", "sph_E1", "sph_E2", "align",
                        "norm_c1", "norm_c2", "dist_c1_xstar", "dist_c2_minusxstar",
                        "I1_resid", "I2_resid", "hamming_c1_to_xstar",
                        "hamming_c2_to_minusxstar"):
                if key in d:
                    val = d[key]
                    if isinstance(val, int):
                        tags.append(f"{key}={val}")
                    else:
                        tags.append(f"{key}={val:.3e}")
            print(f"round {r:3d} f={d['fitness']:.4e}  " + "  ".join(tags))

    return c1, P_inv1, c2, P_inv2, history

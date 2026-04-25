"""
Tests for axis-aligned surface-preserving operations.

Single-ellipsoid ops (`op_hyperplane`, `op_two_plane`) operate on the
raw (c, A, R_sq) tuple; paired ops (`paired_op_hyperplane`,
`paired_op_two_plane`) operate on a `DualEllipsoidState` and return a
new `DualEllipsoidState`.

Op1 (axis-aligned hyperplane x_k = b):
  Preserves any point on E_0 ∩ {x_k = b}. For x* a +/-1 vertex this
  requires b = x*_k. Center moves along e_k by exactly delta.

Op2 (two parallel planes x_k = +/-1):
  Preserves any point on E_0 ∩ {x_k^2 = 1}. Preserves both x* and -x*
  simultaneously. Cannot move c[k] from c[k] = 0.
"""

import itertools

import numpy as np

from surface_preserving_ops import (
    DualEllipsoidState,
    op_hyperplane,
    paired_op_hyperplane,
    op_two_plane,
    paired_op_two_plane,
    on_surface,
    valid_delta_range_op_hyperplane,
    valid_delta_range_op_two_plane,
)


def make_initial_sphere_single(N, x_star=None):
    """Single-ellipsoid initial sphere: A = ones, R_sq = N, c = 0."""
    c = np.zeros(N)
    A = np.ones(N)
    R_sq = float(N)
    if x_star is not None:
        val = float(np.sum(A * (x_star - c) ** 2))
        assert abs(val - R_sq) < 1e-12
    return c, A, R_sq


# ===========================================================================
# Op1: single-ellipsoid tests
# ===========================================================================


def test_op1_identity():
    rng = np.random.default_rng(0)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, A, R_sq = make_initial_sphere_single(N, x_star)
    k = 2
    b = float(x_star[k])

    c2, A2, R_sq2 = op_hyperplane(c, A, R_sq, k, b, delta=0.0)

    assert np.allclose(c, c2)
    assert np.allclose(A, A2)
    assert abs(R_sq - R_sq2) < 1e-12
    print("test_op1_identity: PASS")


def test_op1_preserves_x_star():
    """x* stays on surface across a sweep of delta when b = x*_k."""
    rng = np.random.default_rng(1)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, A, R_sq = make_initial_sphere_single(N, x_star)
    k = 3
    b = float(x_star[k])

    for delta in [-0.4, -0.2, -0.05, 0.05, 0.2, 0.4]:
        c2, A2, R_sq2 = op_hyperplane(c, A, R_sq, k, b, delta)
        ok, val = on_surface(x_star, c2, A2, R_sq2, tol=1e-9)
        assert ok, f"x* not preserved at delta={delta}: val={val}, R_sq={R_sq2}"
    print("test_op1_preserves_x_star: PASS")


def test_op1_motion_along_e_k():
    """c moves only in coordinate k, by exactly delta."""
    rng = np.random.default_rng(2)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, A, R_sq = make_initial_sphere_single(N, x_star)
    k = 1
    b = float(x_star[k])
    delta = 0.2

    c2, _, _ = op_hyperplane(c, A, R_sq, k, b, delta)
    expected = c.copy()
    expected[k] = c[k] + delta
    assert np.allclose(c2, expected, atol=1e-13)
    print("test_op1_motion_along_e_k: PASS")


def test_op1_preserves_axis_alignment():
    rng = np.random.default_rng(3)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, A, R_sq = make_initial_sphere_single(N, x_star)
    k = 0
    b = float(x_star[k])

    _, A_new, _ = op_hyperplane(c, A, R_sq, k, b, delta=0.1)
    diff = A_new - A
    nonzero_indices = np.where(np.abs(diff) > 1e-12)[0]
    assert list(nonzero_indices) == [k]
    assert all(a > 0 for a in A_new)
    print("test_op1_preserves_axis_alignment: PASS")


def test_op1_composition():
    """100 random Op1 ops keep x* on surface."""
    rng = np.random.default_rng(4)
    N = 8
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, A, R_sq = make_initial_sphere_single(N, x_star)

    n_applied = 0
    for _ in range(100):
        k = int(rng.integers(N))
        b = float(x_star[k])
        delta = rng.uniform(-0.05, 0.05)
        try:
            c, A, R_sq = op_hyperplane(c, A, R_sq, k, b, delta)
            n_applied += 1
        except ValueError:
            continue

    ok, val = on_surface(x_star, c, A, R_sq, tol=1e-6)
    assert ok, f"x* drift: val={val}, R_sq={R_sq}"
    print(f"test_op1_composition: PASS ({n_applied} ops, val={val:.2e})")


def test_op1_independent_of_x_star():
    N = 5
    c = np.zeros(N)
    A = np.ones(N)
    R_sq = float(N)
    c1, A1, R1 = op_hyperplane(c, A, R_sq, 2, 1.0, 0.1)
    c2, A2, R2 = op_hyperplane(c, A, R_sq, 2, 1.0, 0.1)
    assert np.allclose(c1, c2)
    assert np.allclose(A1, A2)
    assert abs(R1 - R2) < 1e-15
    print("test_op1_independent_of_x_star: PASS")


def test_op1_degenerate_raises():
    N = 4
    c = np.zeros(N)
    A = np.ones(N)
    R_sq = float(N)
    try:
        op_hyperplane(c, A, R_sq, k=0, b=1.0, delta=1.0)
    except ValueError:
        print("test_op1_degenerate_raises: PASS")
        return
    raise AssertionError("expected ValueError")


# ===========================================================================
# Op1 paired (state-based)
# ===========================================================================


def test_op1_paired_preserves_both():
    rng = np.random.default_rng(10)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    k = 4
    b = float(x_star[k])

    state = paired_op_hyperplane(state, k, b, delta=0.1)

    ok1, val1 = state.x_on_E1(x_star, tol=1e-9)
    ok2, val2 = state.x_on_E2(-x_star, tol=1e-9)
    assert ok1, f"x* drift on E_1: val={val1}"
    assert ok2, f"-x* drift on E_2: val={val2}"
    print("test_op1_paired_preserves_both: PASS")


def test_op1_paired_diverges_A_keeps_c():
    """Multi-step paired Op1: A1, A2 diverge while shared c is preserved."""
    rng = np.random.default_rng(11)
    N = 7
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)

    diverged = False
    n_applied = 0
    for _ in range(40):
        k = int(rng.integers(N))
        b = float(x_star[k])
        delta = rng.uniform(-0.04, 0.04)
        try:
            state = paired_op_hyperplane(state, k, b, delta)
            n_applied += 1
        except ValueError:
            continue
        if not np.allclose(state.A1, state.A2, atol=1e-6):
            diverged = True

    assert diverged, "A1, A2 should diverge under paired Op1 with b != 0"
    ok1, val1 = state.x_on_E1(x_star, tol=1e-5)
    ok2, val2 = state.x_on_E2(-x_star, tol=1e-5)
    assert ok1 and ok2, f"E1 val={val1}, E2 val={val2}"
    print(
        f"test_op1_paired_diverges_A_keeps_c: PASS "
        f"({n_applied} ops, x* val={val1:.2e}, -x* val={val2:.2e})"
    )


def test_op1_paired_kicks_off_origin():
    rng = np.random.default_rng(12)
    N = 8
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    assert np.allclose(state.c, 0.0)

    n_applied = 0
    for _ in range(8):
        k = int(rng.integers(N))
        b = float(x_star[k])
        delta = rng.uniform(-0.15, 0.15)
        try:
            state = paired_op_hyperplane(state, k, b, delta)
            n_applied += 1
        except ValueError:
            continue

    assert np.linalg.norm(state.c) > 1e-3
    ok1, _ = state.x_on_E1(x_star, tol=1e-6)
    ok2, _ = state.x_on_E2(-x_star, tol=1e-6)
    assert ok1 and ok2
    print(
        f"test_op1_paired_kicks_off_origin: PASS "
        f"(|c|={np.linalg.norm(state.c):.4f}, {n_applied} ops)"
    )


# ===========================================================================
# Op2: single-ellipsoid tests
# ===========================================================================


def test_op2_identity_at_zero_delta():
    rng = np.random.default_rng(20)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, A, R_sq = make_initial_sphere_single(N, x_star)
    c2, A2, R_sq2 = op_two_plane(c, A, R_sq, k=2, delta=0.0)
    assert np.allclose(c, c2)
    assert np.allclose(A, A2)
    assert abs(R_sq - R_sq2) < 1e-12
    print("test_op2_identity_at_zero_delta: PASS")


def test_op2_at_origin_no_motion_raises():
    N = 4
    c = np.zeros(N)
    A = np.ones(N)
    R_sq = float(N)
    try:
        op_two_plane(c, A, R_sq, k=1, delta=0.1)
    except ValueError:
        print("test_op2_at_origin_no_motion_raises: PASS")
        return
    raise AssertionError("expected ValueError")


def test_op2_preserves_x_star_after_op1_kick():
    """Op2 preserves whatever vertices lie on E_0's surface; after Op1
    kicks (committing to a sign), x* is on E_0 but -x* is not."""
    rng = np.random.default_rng(21)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, A, R_sq = make_initial_sphere_single(N, x_star)
    k = 2
    c, A, R_sq = op_hyperplane(c, A, R_sq, k, b=float(x_star[k]), delta=0.2)

    c2, A2, R_sq2 = op_two_plane(c, A, R_sq, k, delta=0.05)
    ok, val = on_surface(x_star, c2, A2, R_sq2, tol=1e-9)
    assert ok, f"x* drift: val={val}"
    print(f"test_op2_preserves_x_star_after_op1_kick: PASS (val={val:.2e})")


def test_op2_motion_along_e_k():
    rng = np.random.default_rng(22)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, A, R_sq = make_initial_sphere_single(N, x_star)
    k = 3
    c, A, R_sq = op_hyperplane(c, A, R_sq, k, b=float(x_star[k]), delta=0.3)

    delta = 0.05
    c2, _, _ = op_two_plane(c, A, R_sq, k, delta)
    expected = c.copy()
    expected[k] += delta
    assert np.allclose(c2, expected, atol=1e-13)
    print("test_op2_motion_along_e_k: PASS")


def test_op2_preserves_all_x_k_squared_eq_1_vertices():
    rng = np.random.default_rng(23)
    N = 4
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, A, R_sq = make_initial_sphere_single(N, x_star)
    c, A, R_sq = op_hyperplane(c, A, R_sq, k=0, b=float(x_star[0]), delta=0.1)
    on_surf = []
    for bits in itertools.product((-1.0, 1.0), repeat=N):
        v = np.array(bits)
        ok, _ = on_surface(v, c, A, R_sq, tol=1e-8)
        if ok:
            on_surf.append(v)
    assert len(on_surf) >= 2

    c2, A2, R_sq2 = op_two_plane(c, A, R_sq, k=0, delta=0.03)
    for v in on_surf:
        ok, val = on_surface(v, c2, A2, R_sq2, tol=1e-8)
        assert ok, f"vertex {v} drift: val={val}"
    print(
        f"test_op2_preserves_all_x_k_squared_eq_1_vertices: PASS "
        f"({len(on_surf)} vertices)"
    )


def test_op2_independent_of_x_star():
    N = 5
    c = 0.5 * np.ones(N)
    A = np.ones(N)
    R_sq = float(np.sum(A * (np.ones(N) - c) ** 2))
    c1, A1, R1 = op_two_plane(c, A, R_sq, 2, 0.1)
    c2, A2, R2 = op_two_plane(c, A, R_sq, 2, 0.1)
    assert np.allclose(c1, c2)
    assert np.allclose(A1, A2)
    assert abs(R1 - R2) < 1e-15
    print("test_op2_independent_of_x_star: PASS")


# ===========================================================================
# Op2 paired (state-based)
# ===========================================================================


def test_op2_paired_preserves_both():
    rng = np.random.default_rng(30)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    k = 0
    b = float(x_star[k])
    state = paired_op_hyperplane(state, k, b, delta=0.2)
    state = paired_op_two_plane(state, k, delta=0.05)

    ok1, val1 = state.x_on_E1(x_star, tol=1e-8)
    ok2, val2 = state.x_on_E2(-x_star, tol=1e-8)
    assert ok1 and ok2, f"E1 val={val1}, E2 val={val2}"
    print("test_op2_paired_preserves_both: PASS")


def test_paired_mixed_op1_op2_long_horizon():
    rng = np.random.default_rng(40)
    N = 8
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)

    for k in range(N):
        b = float(x_star[k])
        try:
            state = paired_op_hyperplane(state, k, b, delta=0.05)
        except ValueError:
            continue

    n_op1 = n_op2 = 0
    for _ in range(60):
        k = int(rng.integers(N))
        if rng.random() < 0.5:
            b = float(x_star[k])
            delta = rng.uniform(-0.02, 0.02)
            try:
                state = paired_op_hyperplane(state, k, b, delta)
                n_op1 += 1
            except ValueError:
                pass
        else:
            delta = rng.uniform(-0.02, 0.02)
            try:
                state = paired_op_two_plane(state, k, delta)
                n_op2 += 1
            except ValueError:
                pass

    ok1, val1 = state.x_on_E1(x_star, tol=1e-5)
    ok2, val2 = state.x_on_E2(-x_star, tol=1e-5)
    assert ok1 and ok2, f"E1 val={val1}, E2 val={val2}"
    print(
        f"test_paired_mixed_op1_op2_long_horizon: PASS "
        f"(Op1: {n_op1}, Op2: {n_op2}, x*={val1:.2e}, -x*={val2:.2e})"
    )


def test_paired_keeps_c_shared_through_divergence():
    rng = np.random.default_rng(41)
    N = 7
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)

    diverged = False
    n_steps = 0
    for k in range(N):
        b = float(x_star[k])
        try:
            state = paired_op_hyperplane(state, k, b, delta=0.05)
            n_steps += 1
        except ValueError:
            pass
    for _ in range(50):
        k = int(rng.integers(N))
        if rng.random() < 0.5:
            b = float(x_star[k])
            delta = rng.uniform(-0.03, 0.03)
            try:
                state = paired_op_hyperplane(state, k, b, delta)
                n_steps += 1
            except ValueError:
                pass
        else:
            delta = rng.uniform(-0.03, 0.03)
            try:
                state = paired_op_two_plane(state, k, delta)
                n_steps += 1
            except ValueError:
                pass
        if (not np.allclose(state.A1, state.A2, atol=1e-5)
                or abs(state.R1_sq - state.R2_sq) > 1e-5):
            diverged = True

    assert diverged
    print(
        f"test_paired_keeps_c_shared_through_divergence: PASS "
        f"({n_steps} steps; A1, A2 diverged; c stayed shared)"
    )


# ===========================================================================
# Valid-range helpers
# ===========================================================================


def test_valid_range_op_hyperplane_at_origin():
    state = DualEllipsoidState.initial_sphere(N=4)
    for b in (1.0, -1.0):
        lo, hi = state.valid_range_hyperplane(k=2, b=b)
        assert abs(lo - (-1.0)) < 1e-12
        assert abs(hi - 1.0) < 1e-12
    # Also test the free-function form
    c = np.zeros(4)
    lo, hi = valid_delta_range_op_hyperplane(c, k=2, b=1.0)
    assert lo == -1.0 and hi == 1.0
    print("test_valid_range_op_hyperplane_at_origin: PASS")


def test_valid_range_op_hyperplane_off_origin():
    c = np.array([0.3, 0.0, 0.0, 0.0])
    lo, hi = valid_delta_range_op_hyperplane(c, k=0, b=1.0)
    assert abs(lo - (-1.3)) < 1e-12
    assert abs(hi - 0.7) < 1e-12
    lo2, hi2 = valid_delta_range_op_hyperplane(c, k=0, b=-1.0)
    assert abs(lo - lo2) < 1e-12 and abs(hi - hi2) < 1e-12
    print("test_valid_range_op_hyperplane_off_origin: PASS")


def test_valid_range_op_hyperplane_actual_validity():
    rng = np.random.default_rng(50)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    state = paired_op_hyperplane(state, k=2, b=float(x_star[2]), delta=0.3)

    k = 2
    b = float(x_star[2])
    lo, hi = state.valid_range_hyperplane(k, b)
    paired_op_hyperplane(state, k, b, 0.5 * (lo + hi))
    paired_op_hyperplane(state, k, b, hi - 1e-3)
    try:
        paired_op_hyperplane(state, k, b, hi + 1e-3)
    except ValueError:
        print("test_valid_range_op_hyperplane_actual_validity: PASS")
        return
    raise AssertionError("expected ValueError just outside hi")


def test_valid_range_op_two_plane_at_origin():
    state = DualEllipsoidState.initial_sphere(N=4)
    lo, hi = state.valid_range_two_plane(k=1)
    assert lo == 0.0 and hi == 0.0
    print("test_valid_range_op_two_plane_at_origin: PASS")


def test_valid_range_op_two_plane_positive_c():
    c = np.array([0.5, 0.0, 0.0, 0.0])
    lo, hi = valid_delta_range_op_two_plane(c, k=0)
    assert abs(lo - (-0.5)) < 1e-12
    assert hi == float("inf")
    print("test_valid_range_op_two_plane_positive_c: PASS")


def test_valid_range_op_two_plane_negative_c():
    c = np.array([-0.4, 0.0, 0.0, 0.0])
    lo, hi = valid_delta_range_op_two_plane(c, k=0)
    assert lo == float("-inf")
    assert abs(hi - 0.4) < 1e-12
    print("test_valid_range_op_two_plane_negative_c: PASS")


def test_valid_range_op_two_plane_actual_validity():
    rng = np.random.default_rng(51)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    k = 1
    state = paired_op_hyperplane(
        state, k, b=float(x_star[k]), delta=0.3 * float(x_star[k])
    )
    lo, hi = state.valid_range_two_plane(k)
    if state.c[k] > 0:
        delta_inside = -state.c[k] * 0.5
    else:
        delta_inside = -state.c[k] * 0.5
    paired_op_two_plane(state, k, delta_inside)  # should succeed
    try:
        paired_op_two_plane(state, k, -state.c[k])  # at the wall
    except ValueError:
        print("test_valid_range_op_two_plane_actual_validity: PASS")
        return
    raise AssertionError("expected ValueError at lo wall")


# ===========================================================================
# DualEllipsoidState class
# ===========================================================================


def test_state_initial_sphere():
    state = DualEllipsoidState.initial_sphere(N=6)
    assert state.N == 6
    assert np.allclose(state.c, 0.0)
    assert np.allclose(state.A1, 1.0)
    assert np.allclose(state.A2, 1.0)
    assert state.R1_sq == 6.0 and state.R2_sq == 6.0
    # Every +/-1 vertex on both surfaces:
    for bits in itertools.product((-1.0, 1.0), repeat=6):
        v = np.array(bits)
        ok1, _ = state.x_on_E1(v)
        ok2, _ = state.x_on_E2(v)
        assert ok1 and ok2
    print("test_state_initial_sphere: PASS")


def test_state_paired_ops_return_new_state():
    """Paired ops must return a new DualEllipsoidState (immutable update)."""
    state = DualEllipsoidState.initial_sphere(N=4)
    state2 = paired_op_hyperplane(state, k=0, b=1.0, delta=0.1)
    assert isinstance(state2, DualEllipsoidState)
    assert state2 is not state
    # Original unchanged:
    assert np.allclose(state.c, 0.0)
    print("test_state_paired_ops_return_new_state: PASS")


# ===========================================================================
# Run all
# ===========================================================================


if __name__ == "__main__":
    # Op1 single
    test_op1_identity()
    test_op1_preserves_x_star()
    test_op1_motion_along_e_k()
    test_op1_preserves_axis_alignment()
    test_op1_composition()
    test_op1_independent_of_x_star()
    test_op1_degenerate_raises()
    # Op1 paired
    test_op1_paired_preserves_both()
    test_op1_paired_diverges_A_keeps_c()
    test_op1_paired_kicks_off_origin()
    # Op2 single
    test_op2_identity_at_zero_delta()
    test_op2_at_origin_no_motion_raises()
    test_op2_preserves_x_star_after_op1_kick()
    test_op2_motion_along_e_k()
    test_op2_preserves_all_x_k_squared_eq_1_vertices()
    test_op2_independent_of_x_star()
    # Op2 paired + mixed
    test_op2_paired_preserves_both()
    test_paired_mixed_op1_op2_long_horizon()
    test_paired_keeps_c_shared_through_divergence()
    # Range helpers
    test_valid_range_op_hyperplane_at_origin()
    test_valid_range_op_hyperplane_off_origin()
    test_valid_range_op_hyperplane_actual_validity()
    test_valid_range_op_two_plane_at_origin()
    test_valid_range_op_two_plane_positive_c()
    test_valid_range_op_two_plane_negative_c()
    test_valid_range_op_two_plane_actual_validity()
    # State class
    test_state_initial_sphere()
    test_state_paired_ops_return_new_state()
    print("\nALL TESTS PASS")

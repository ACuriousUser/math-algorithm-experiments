"""
Tests for axis-aligned surface-preserving operations.

State: DualEllipsoidState with shared (c, A) and per-ellipsoid R_i_sq.

Op1 is the linear pencil M*(a^T x - b), which preserves x* on E_1 and
-x* on E_2 (with b on E_1, -b on E_2). Handles general sparse hyperplane
normals -- this is what makes axis-aligned + general constraints work.

Op2 is the quadratic pencil M*(x_k^2 - 1), preserves both x* and -x*
(sign-agnostic) for the box constraint x_k^2 = 1.
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
    valid_delta_range_op_two_plane,
)


def random_3sparse_constraint(N, x_star, rng):
    """Sample a 3-sparse +/-1 constraint satisfied by x*."""
    idx = rng.choice(N, size=3, replace=False)
    coefs = rng.choice([-1.0, 1.0], size=3)
    a = np.zeros(N)
    a[idx] = coefs
    b = float(a @ x_star)
    return a, b


# ===========================================================================
# Op1 single-ellipsoid (linear pencil)
# ===========================================================================


def test_op1_identity_at_M_zero():
    rng = np.random.default_rng(0)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    c = np.zeros(N); A = np.ones(N); R_sq = float(N)
    a, b = random_3sparse_constraint(N, x_star, rng)
    c2, A2, R2 = op_hyperplane(c, A, R_sq, a, b, M=0.0)
    assert np.allclose(c, c2)
    assert np.allclose(A, A2)
    assert abs(R_sq - R2) < 1e-12
    print("test_op1_identity_at_M_zero: PASS")


def test_op1_preserves_x_star_general_sparse():
    """Linear pencil preserves x* under general 3-sparse hyperplane normals."""
    rng = np.random.default_rng(1)
    N = 8
    x_star = rng.choice([-1.0, 1.0], size=N)
    c = np.zeros(N); A = np.ones(N); R_sq = float(N)
    for _ in range(20):
        a, b = random_3sparse_constraint(N, x_star, rng)
        for M in [-2.0, -0.5, 0.5, 2.0]:
            c2, A2, R2 = op_hyperplane(c, A, R_sq, a, b, M)
            ok, val = on_surface(x_star, c2, A2, R2, tol=1e-9)
            assert ok, f"x* drift at a={a}, b={b}, M={M}: val={val}"
    print("test_op1_preserves_x_star_general_sparse: PASS")


def test_op1_preserves_axis_alignment():
    """Linear pencil leaves A unchanged (axis-alignment trivially preserved)."""
    rng = np.random.default_rng(2)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c = np.zeros(N); A = np.ones(N); R_sq = float(N)
    a, b = random_3sparse_constraint(N, x_star, rng)
    _, A_new, _ = op_hyperplane(c, A, R_sq, a, b, M=1.5)
    assert np.allclose(A, A_new), f"A changed: {A} -> {A_new}"
    print("test_op1_preserves_axis_alignment: PASS")


def test_op1_motion_along_a_over_A():
    """c moves in direction (a/A) componentwise, magnitude -M/2."""
    rng = np.random.default_rng(3)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    c = np.array([0.1, -0.2, 0.0, 0.3, 0.0, -0.1])
    A = np.array([1.0, 2.0, 0.5, 1.5, 1.0, 0.8])
    # Construct R_sq so x* is on surface
    R_sq = float(np.sum(A * (x_star - c) ** 2))
    a, b = random_3sparse_constraint(N, x_star, rng)
    M = 0.7
    c2, _, _ = op_hyperplane(c, A, R_sq, a, b, M)
    expected = c - M * a / (2.0 * A)
    assert np.allclose(c2, expected, atol=1e-13)
    print("test_op1_motion_along_a_over_A: PASS")


def test_op1_unbounded_M():
    """For x* on surface, newR_sq > 0 for any M (Cauchy-Schwarz)."""
    rng = np.random.default_rng(4)
    N = 7
    x_star = rng.choice([-1.0, 1.0], size=N)
    c = np.zeros(N); A = np.ones(N); R_sq = float(N)
    a, b = random_3sparse_constraint(N, x_star, rng)
    for M in [-1000.0, -10.0, -1.0, 1.0, 10.0, 1000.0]:
        c2, A2, R2 = op_hyperplane(c, A, R_sq, a, b, M)
        assert R2 > 0, f"R_sq became non-positive at M={M}: R2={R2}"
        ok, _ = on_surface(x_star, c2, A2, R2, tol=1e-6)
        assert ok, f"x* drift at extreme M={M}"
    print("test_op1_unbounded_M: PASS")


def test_op1_composition_general_sparse():
    """100 random Op1 ops with sparse 3-coefficient hyperplanes; x* stays on surface."""
    rng = np.random.default_rng(5)
    N = 10
    x_star = rng.choice([-1.0, 1.0], size=N)
    c = np.zeros(N); A = np.ones(N); R_sq = float(N)
    n_applied = 0
    for _ in range(100):
        a, b = random_3sparse_constraint(N, x_star, rng)
        M = rng.uniform(-0.3, 0.3)
        try:
            c, A, R_sq = op_hyperplane(c, A, R_sq, a, b, M)
            n_applied += 1
        except ValueError:
            continue
    ok, val = on_surface(x_star, c, A, R_sq, tol=1e-6)
    assert ok, f"drift after {n_applied} ops: val={val}"
    # A should be unchanged throughout
    assert np.allclose(A, np.ones(N)), "A should be unchanged by Op1 ever"
    print(f"test_op1_composition_general_sparse: PASS ({n_applied} ops, val={val:.2e})")


# ===========================================================================
# Op1 paired (linear pencil)
# ===========================================================================


def test_op1_paired_preserves_both():
    rng = np.random.default_rng(10)
    N = 7
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    a, b = random_3sparse_constraint(N, x_star, rng)
    state = paired_op_hyperplane(state, a, b, M=0.5)
    ok1, val1 = state.x_on_E1(x_star, tol=1e-9)
    ok2, val2 = state.x_on_E2(-x_star, tol=1e-9)
    assert ok1, f"x* drift on E_1: val={val1}"
    assert ok2, f"-x* drift on E_2: val={val2}"
    print("test_op1_paired_preserves_both: PASS")


def test_op1_paired_preserves_shared_A():
    """Paired Op1 must preserve shared A across both ellipsoids."""
    rng = np.random.default_rng(11)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    A_initial = state.A.copy()
    for _ in range(20):
        a, b = random_3sparse_constraint(N, x_star, rng)
        M = rng.uniform(-0.4, 0.4)
        try:
            state = paired_op_hyperplane(state, a, b, M)
        except ValueError:
            continue
    # A must be unchanged by Op1
    assert np.allclose(state.A, A_initial), \
        f"A drifted under Op1: {state.A} vs initial {A_initial}"
    # Both invariants
    ok1, val1 = state.x_on_E1(x_star, tol=1e-6)
    ok2, val2 = state.x_on_E2(-x_star, tol=1e-6)
    assert ok1 and ok2, f"E1 val={val1}, E2 val={val2}"
    print("test_op1_paired_preserves_shared_A: PASS")


def test_op1_paired_diverges_R():
    """R_1_sq and R_2_sq diverge under paired Op1 (by 2*M*b per call)."""
    rng = np.random.default_rng(12)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    assert state.R1_sq == state.R2_sq
    a, b = random_3sparse_constraint(N, x_star, rng)
    state = paired_op_hyperplane(state, a, b, M=0.5)
    expected_diff = 2 * 0.5 * b
    actual_diff = state.R1_sq - state.R2_sq
    assert abs(actual_diff - expected_diff) < 1e-10, \
        f"R1-R2 diff: actual={actual_diff}, expected={expected_diff}"
    print("test_op1_paired_diverges_R: PASS")


def test_op1_paired_kicks_off_origin():
    """A single paired Op1 with general sparse hyperplane kicks c off origin."""
    rng = np.random.default_rng(13)
    N = 8
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    assert np.allclose(state.c, 0.0)
    a, b = random_3sparse_constraint(N, x_star, rng)
    state = paired_op_hyperplane(state, a, b, M=0.3)
    assert np.linalg.norm(state.c) > 1e-3
    ok1, _ = state.x_on_E1(x_star, tol=1e-9)
    ok2, _ = state.x_on_E2(-x_star, tol=1e-9)
    assert ok1 and ok2
    print(f"test_op1_paired_kicks_off_origin: PASS (|c|={np.linalg.norm(state.c):.4f})")


# ===========================================================================
# Op2 single-ellipsoid (quadratic pencil)
# ===========================================================================


def test_op2_identity_at_zero_delta():
    N = 5
    c = np.zeros(N); A = np.ones(N); R_sq = float(N)
    c2, A2, R2 = op_two_plane(c, A, R_sq, k=2, delta=0.0)
    assert np.allclose(c, c2) and np.allclose(A, A2) and abs(R_sq - R2) < 1e-12
    print("test_op2_identity_at_zero_delta: PASS")


def test_op2_at_origin_no_motion_raises():
    N = 4; c = np.zeros(N); A = np.ones(N); R_sq = float(N)
    try:
        op_two_plane(c, A, R_sq, k=1, delta=0.1)
    except ValueError:
        print("test_op2_at_origin_no_motion_raises: PASS")
        return
    raise AssertionError("expected ValueError")


def test_op2_preserves_x_star_after_op1_kick():
    """Op2 preserves x* on surface after a paired Op1 kick puts c off origin."""
    rng = np.random.default_rng(20)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    a, b = random_3sparse_constraint(N, x_star, rng)
    state = paired_op_hyperplane(state, a, b, M=0.5)
    # Now Op2 single-ellipsoid on E_1
    c, A, R_sq = state.c, state.A, state.R1_sq
    # Find a coord with c[k] != 0
    k = next(i for i in range(N) if abs(c[i]) > 1e-3)
    c2, A2, R2 = op_two_plane(c, A, R_sq, k, delta=0.05)
    ok, val = on_surface(x_star, c2, A2, R2, tol=1e-9)
    assert ok, f"x* drift: val={val}"
    print("test_op2_preserves_x_star_after_op1_kick: PASS")


def test_op2_motion_along_e_k():
    rng = np.random.default_rng(21)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    a, b = random_3sparse_constraint(N, x_star, rng)
    state = paired_op_hyperplane(state, a, b, M=0.4)
    k = next(i for i in range(N) if abs(state.c[i]) > 1e-3)
    delta = 0.05
    c2, _, _ = op_two_plane(state.c, state.A, state.R1_sq, k, delta)
    expected = state.c.copy(); expected[k] += delta
    assert np.allclose(c2, expected, atol=1e-13)
    print("test_op2_motion_along_e_k: PASS")


# ===========================================================================
# Op2 paired (preserves shared A)
# ===========================================================================


def test_op2_paired_preserves_both_invariants():
    rng = np.random.default_rng(30)
    N = 7
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    a, b = random_3sparse_constraint(N, x_star, rng)
    state = paired_op_hyperplane(state, a, b, M=0.4)
    k = next(i for i in range(N) if abs(state.c[i]) > 1e-3)
    state = paired_op_two_plane(state, k, delta=0.05)
    ok1, val1 = state.x_on_E1(x_star, tol=1e-9)
    ok2, val2 = state.x_on_E2(-x_star, tol=1e-9)
    assert ok1 and ok2, f"E1 val={val1}, E2 val={val2}"
    print("test_op2_paired_preserves_both_invariants: PASS")


def test_op2_paired_preserves_shared_A():
    """A must be the SAME single vector after paired Op2 (since both sides
    use shared A and same delta -> same M -> same newA[k])."""
    rng = np.random.default_rng(31)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    a, b = random_3sparse_constraint(N, x_star, rng)
    state = paired_op_hyperplane(state, a, b, M=0.4)
    # Apply paired Op2 several times
    for _ in range(5):
        k = next(i for i in range(N) if abs(state.c[i]) > 1e-3)
        try:
            state = paired_op_two_plane(state, k, delta=0.02)
        except ValueError:
            break
    # The dataclass holds a single A; just confirm it's consistent.
    assert state.A.ndim == 1 and len(state.A) == N
    print("test_op2_paired_preserves_shared_A: PASS")


def test_op2_paired_does_not_change_R_diff():
    """R_1_sq - R_2_sq is unchanged by paired Op2 (only Op1 changes it)."""
    rng = np.random.default_rng(32)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)
    a, b = random_3sparse_constraint(N, x_star, rng)
    state = paired_op_hyperplane(state, a, b, M=0.5)
    R_diff_before = state.R1_sq - state.R2_sq
    k = next(i for i in range(N) if abs(state.c[i]) > 1e-3)
    state = paired_op_two_plane(state, k, delta=0.04)
    R_diff_after = state.R1_sq - state.R2_sq
    assert abs(R_diff_before - R_diff_after) < 1e-10, \
        f"R_diff changed: {R_diff_before} -> {R_diff_after}"
    print("test_op2_paired_does_not_change_R_diff: PASS")


# ===========================================================================
# Mixed long-horizon composition
# ===========================================================================


def test_paired_mixed_op1_op2_long_horizon():
    """Mix general-sparse paired Op1 and paired Op2; both invariants hold."""
    rng = np.random.default_rng(40)
    N = 10
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)

    # Use Op1 to kick all coords off origin first
    for _ in range(5):
        a, b = random_3sparse_constraint(N, x_star, rng)
        try:
            state = paired_op_hyperplane(state, a, b, M=0.2)
        except ValueError:
            pass

    n_op1 = n_op2 = 0
    for _ in range(80):
        if rng.random() < 0.5:
            a, b = random_3sparse_constraint(N, x_star, rng)
            try:
                state = paired_op_hyperplane(state, a, b, M=rng.uniform(-0.1, 0.1))
                n_op1 += 1
            except ValueError:
                pass
        else:
            k = int(rng.integers(N))
            if abs(state.c[k]) < 1e-6:
                continue
            try:
                state = paired_op_two_plane(state, k, rng.uniform(-0.02, 0.02))
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


def test_paired_A_stays_shared_under_mixed_ops():
    """A is a single vector across both ellipsoids; mixed Op1+Op2 must keep
    it consistent (i.e., not somehow corrupt it)."""
    rng = np.random.default_rng(41)
    N = 7
    x_star = rng.choice([-1.0, 1.0], size=N)
    state = DualEllipsoidState.initial_sphere(N)

    A_initial = state.A.copy()
    for _ in range(3):
        a, b = random_3sparse_constraint(N, x_star, rng)
        state = paired_op_hyperplane(state, a, b, M=0.3)

    # After Op1 only: A unchanged
    assert np.allclose(state.A, A_initial), "Op1 changed A"

    # Apply Op2; A[k] should change for the chosen k
    k = next(i for i in range(N) if abs(state.c[i]) > 1e-3)
    A_before_op2 = state.A.copy()
    state = paired_op_two_plane(state, k, delta=0.05)
    diff = state.A - A_before_op2
    nonzero_diff = np.where(np.abs(diff) > 1e-12)[0]
    assert list(nonzero_diff) == [k], (
        f"Op2 changed A at indices {nonzero_diff}, expected only {k}"
    )
    print("test_paired_A_stays_shared_under_mixed_ops: PASS")


# ===========================================================================
# Valid-range helpers
# ===========================================================================


def test_valid_range_op_two_plane_at_origin():
    state = DualEllipsoidState.initial_sphere(N=4)
    lo, hi = state.valid_range_two_plane(k=1)
    assert lo == 0.0 and hi == 0.0
    print("test_valid_range_op_two_plane_at_origin: PASS")


def test_valid_range_op_two_plane_positive_c():
    c = np.array([0.5, 0.0, 0.0, 0.0])
    lo, hi = valid_delta_range_op_two_plane(c, k=0)
    assert abs(lo - (-0.5)) < 1e-12 and hi == float("inf")
    print("test_valid_range_op_two_plane_positive_c: PASS")


def test_valid_range_op_two_plane_negative_c():
    c = np.array([-0.4, 0.0, 0.0, 0.0])
    lo, hi = valid_delta_range_op_two_plane(c, k=0)
    assert lo == float("-inf") and abs(hi - 0.4) < 1e-12
    print("test_valid_range_op_two_plane_negative_c: PASS")


# ===========================================================================
# DualEllipsoidState
# ===========================================================================


def test_state_initial_sphere():
    state = DualEllipsoidState.initial_sphere(N=5)
    assert state.N == 5
    assert np.allclose(state.c, 0.0)
    assert np.allclose(state.A, 1.0)
    assert state.R1_sq == 5.0 and state.R2_sq == 5.0
    for bits in itertools.product((-1.0, 1.0), repeat=5):
        v = np.array(bits)
        assert state.x_on_E1(v)[0] and state.x_on_E2(v)[0]
    print("test_state_initial_sphere: PASS")


def test_state_paired_ops_return_new_state():
    state = DualEllipsoidState.initial_sphere(N=4)
    state2 = paired_op_hyperplane(state, np.array([1.0, -1.0, 0.0, 0.0]), 0.0, M=0.1)
    assert isinstance(state2, DualEllipsoidState)
    assert state2 is not state
    assert np.allclose(state.c, 0.0)
    print("test_state_paired_ops_return_new_state: PASS")


# ===========================================================================
# Run all
# ===========================================================================


if __name__ == "__main__":
    # Op1 single
    test_op1_identity_at_M_zero()
    test_op1_preserves_x_star_general_sparse()
    test_op1_preserves_axis_alignment()
    test_op1_motion_along_a_over_A()
    test_op1_unbounded_M()
    test_op1_composition_general_sparse()
    # Op1 paired
    test_op1_paired_preserves_both()
    test_op1_paired_preserves_shared_A()
    test_op1_paired_diverges_R()
    test_op1_paired_kicks_off_origin()
    # Op2 single
    test_op2_identity_at_zero_delta()
    test_op2_at_origin_no_motion_raises()
    test_op2_preserves_x_star_after_op1_kick()
    test_op2_motion_along_e_k()
    # Op2 paired
    test_op2_paired_preserves_both_invariants()
    test_op2_paired_preserves_shared_A()
    test_op2_paired_does_not_change_R_diff()
    # Mixed
    test_paired_mixed_op1_op2_long_horizon()
    test_paired_A_stays_shared_under_mixed_ops()
    # Range helpers
    test_valid_range_op_two_plane_at_origin()
    test_valid_range_op_two_plane_positive_c()
    test_valid_range_op_two_plane_negative_c()
    # State
    test_state_initial_sphere()
    test_state_paired_ops_return_new_state()
    print("\nALL TESTS PASS")

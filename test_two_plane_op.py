"""
Tests for the surface-preserving two-plane (box) operation.

Verifies:
  For any x with (x-c)^T P_inv (x-c) = 1 and x_i in {-1, +1},
  we have (x - c_new)^T P_inv_new (x - c_new) = 1.

Key property: the operation works simultaneously for x* AND -x* (both
satisfy x_i^2 = 1), so the same op preserves both invariants without a
"paired" form.
"""

import itertools

import numpy as np

from surface_preserving_ops import (
    op_hyperplane_pencil,
    op_two_plane_pencil,
    on_surface,
)


def make_initial_sphere(N, x_star):
    r2 = float(x_star @ x_star)
    c = np.zeros(N)
    P_inv = np.eye(N) / r2
    return c, P_inv


def random_3sparse_constraint(N, x_star, rng):
    idx = rng.choice(N, size=3, replace=False)
    coefs = rng.choice([-1, 1], size=3).astype(float)
    a = np.zeros(N)
    a[idx] = coefs
    b = float(a @ x_star)
    return a, b


# ---------------------------------------------------------------------------
# Test 1: identity at t=0
# ---------------------------------------------------------------------------

def test_identity():
    rng = np.random.default_rng(10)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)
    c2, P_inv2 = op_two_plane_pencil(c, P_inv, i=2, t=0.0)
    assert np.allclose(c, c2, atol=1e-12)
    assert np.allclose(P_inv, P_inv2, atol=1e-12)
    print("test_identity_op2: PASS")


# ---------------------------------------------------------------------------
# Test 2: x* preserved across various t for one variable
# ---------------------------------------------------------------------------

def test_x_star_preserved_various_t():
    rng = np.random.default_rng(11)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)

    # P_inv[i,i] = 1/N, so valid t > -1/N
    pii = float(P_inv[2, 2])
    for t in [-0.5 * pii, -0.1 * pii, 0.0, 0.5 * pii, 1.0 * pii, 5.0 * pii]:
        c2, P_inv2 = op_two_plane_pencil(c, P_inv, i=2, t=t)
        ok, val = on_surface(x_star, c2, P_inv2, tol=1e-8)
        assert ok, f"x* not on surface after t={t} on var 2: val={val}"
    print("test_x_star_preserved_various_t: PASS")


# ---------------------------------------------------------------------------
# Test 3: x* AND -x* preserved simultaneously by SAME op
# ---------------------------------------------------------------------------

def test_x_star_and_minus_x_star_both_preserved():
    rng = np.random.default_rng(12)
    N = 7
    x_star = rng.choice([-1.0, 1.0], size=N)
    minus_x_star = -x_star

    # Sphere has both x* and -x* on surface
    c, P_inv = make_initial_sphere(N, x_star)
    ok_pos, _ = on_surface(x_star, c, P_inv, tol=1e-12)
    ok_neg, _ = on_surface(minus_x_star, c, P_inv, tol=1e-12)
    assert ok_pos and ok_neg

    # Apply the SAME box op multiple times, both should remain on surface
    for _ in range(15):
        i = rng.integers(N)
        pii = float(P_inv[i, i])
        t = rng.uniform(-0.5 * pii, 2.0 * pii)
        c, P_inv = op_two_plane_pencil(c, P_inv, i=int(i), t=t)
        ok_pos, val_p = on_surface(x_star, c, P_inv, tol=1e-7)
        ok_neg, val_n = on_surface(minus_x_star, c, P_inv, tol=1e-7)
        assert ok_pos, f"x* lost: {val_p}"
        assert ok_neg, f"-x* lost: {val_n}"
    print("test_x_star_and_minus_x_star_both_preserved: PASS")


# ---------------------------------------------------------------------------
# Test 4: ALL ±1 vertices on initial surface preserved
# ---------------------------------------------------------------------------

def test_all_pm1_vertices_preserved():
    """Initial sphere of radius sqrt(N) has every ±1 vertex on its surface.
    Box op should keep them all."""
    rng = np.random.default_rng(13)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)

    i = 1
    pii = float(P_inv[i, i])
    t = 1.5 * pii
    c2, P_inv2 = op_two_plane_pencil(c, P_inv, i=i, t=t)

    n_preserved = 0
    n_total = 0
    for bits in itertools.product((-1.0, 1.0), repeat=N):
        v = np.array(bits)
        # Initially on surface (|v|^2 = N) ✓
        ok_init, _ = on_surface(v, c, P_inv, tol=1e-10)
        assert ok_init
        n_total += 1
        ok, _ = on_surface(v, c2, P_inv2, tol=1e-8)
        if ok:
            n_preserved += 1
    assert n_preserved == n_total, f"only {n_preserved}/{n_total} preserved"
    print(f"test_all_pm1_vertices_preserved: PASS ({n_preserved} vertices)")


# ---------------------------------------------------------------------------
# Test 5: Composition of many box ops
# ---------------------------------------------------------------------------

def test_composition_of_box_ops():
    rng = np.random.default_rng(14)
    N = 7
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)

    n_ops = 100
    for _ in range(n_ops):
        i = int(rng.integers(N))
        pii = float(P_inv[i, i])
        # Keep step modest to avoid degeneracy
        t = rng.uniform(-0.3 * pii, 0.5 * pii)
        c, P_inv = op_two_plane_pencil(c, P_inv, i=i, t=t)

    ok, val = on_surface(x_star, c, P_inv, tol=1e-5)
    assert ok, f"x* drifted after {n_ops} box ops: val={val}"
    print(f"test_composition_of_box_ops: PASS (val={val:.2e})")


# ---------------------------------------------------------------------------
# Test 6: Mixed Op1 and Op2 composition
# ---------------------------------------------------------------------------

def test_mixed_op1_op2_composition():
    """Real use case: alternate hyperplane and box ops, both preserving x*."""
    rng = np.random.default_rng(15)
    N = 8
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)

    n_total = 80
    for k in range(n_total):
        if k % 2 == 0:
            a, b = random_3sparse_constraint(N, x_star, rng)
            P = np.linalg.inv(P_inv)
            aPa = float(a @ P @ a)
            s = rng.uniform(-0.4 / aPa, 0.4 / aPa)
            c, P_inv = op_hyperplane_pencil(c, P_inv, a, b, s=s)
        else:
            i = int(rng.integers(N))
            pii = float(P_inv[i, i])
            t = rng.uniform(-0.3 * pii, 0.4 * pii)
            c, P_inv = op_two_plane_pencil(c, P_inv, i=i, t=t)

    ok, val = on_surface(x_star, c, P_inv, tol=1e-5)
    assert ok, f"x* drifted after {n_total} mixed ops: val={val}"
    print(f"test_mixed_op1_op2_composition: PASS (val={val:.2e})")


# ---------------------------------------------------------------------------
# Test 7: Center actually moves in coordinate direction
# ---------------------------------------------------------------------------

def test_center_moves_in_coord_direction():
    rng = np.random.default_rng(16)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)

    # First make c non-trivial via Op1
    a, b = random_3sparse_constraint(N, x_star, rng)
    c, P_inv = op_hyperplane_pencil(c, P_inv, a, b, s=0.4)

    # Now apply Op2 on coord 2
    pii = float(P_inv[2, 2])
    t = 1.2 * pii
    c2, P_inv2 = op_two_plane_pencil(c, P_inv, i=2, t=t)

    delta = c2 - c
    assert np.linalg.norm(delta) > 1e-3, f"center should move; delta={delta}"

    # Most of the change should be in coordinate 2 (since the rank-1 update
    # is along e_2). Check the dominant component is in dim 2.
    # (Not a strict requirement but a nice property.)
    assert abs(delta[2]) > 0.5 * np.linalg.norm(delta) - 1e-9, \
        f"primary motion should be in coord 2: delta={delta}"

    ok, val = on_surface(x_star, c2, P_inv2, tol=1e-8)
    assert ok, f"x* not preserved: val={val}"
    print(f"test_center_moves_in_coord_direction: PASS (delta={delta})")


# ---------------------------------------------------------------------------
# Test 8: Independent of x*'s identity
# ---------------------------------------------------------------------------

def test_op2_independent_of_x_star():
    """Op2 should depend only on (c, P_inv, i, t), not on x*."""
    N = 6
    c0 = np.zeros(N)
    P_inv0 = np.eye(N) / 6.0
    i, t = 3, 0.05

    c_a, P_inv_a = op_two_plane_pencil(c0, P_inv0, i=i, t=t)
    c_b, P_inv_b = op_two_plane_pencil(c0, P_inv0, i=i, t=t)
    assert np.allclose(c_a, c_b)
    assert np.allclose(P_inv_a, P_inv_b)
    print("test_op2_independent_of_x_star: PASS")


# ---------------------------------------------------------------------------
# Test 9: Symmetry — sign of x*_i doesn't matter
# ---------------------------------------------------------------------------

def test_op2_symmetric_in_x_star_sign():
    """Apply same Op2 starting from same sphere, with x* having +1 vs -1 in
    coord i. Both should remain on the surface."""
    rng = np.random.default_rng(17)
    N = 6
    # Build x* with +1 in coord 2
    x_star_p = rng.choice([-1.0, 1.0], size=N)
    x_star_p[2] = 1.0
    # Mirror in coord 2
    x_star_n = x_star_p.copy()
    x_star_n[2] = -1.0

    c0 = np.zeros(N)
    P_inv0 = np.eye(N) / N

    pii = float(P_inv0[2, 2])
    t = 0.6 * pii
    c, P_inv = op_two_plane_pencil(c0, P_inv0, i=2, t=t)

    ok_p, val_p = on_surface(x_star_p, c, P_inv, tol=1e-8)
    ok_n, val_n = on_surface(x_star_n, c, P_inv, tol=1e-8)
    assert ok_p and ok_n, f"both ±1 cases should be preserved: {val_p}, {val_n}"
    print("test_op2_symmetric_in_x_star_sign: PASS")


# ---------------------------------------------------------------------------
# Test 10: Validity-bound check
# ---------------------------------------------------------------------------

def test_validity_bound_raises():
    N = 4
    c = np.zeros(N)
    P_inv = np.eye(N) / N
    pii = float(P_inv[1, 1])
    bad_t = -1.5 * pii  # below the validity bound
    raised = False
    try:
        op_two_plane_pencil(c, P_inv, i=1, t=bad_t)
    except ValueError:
        raised = True
    assert raised, "expected ValueError for t below validity range"
    print("test_validity_bound_raises: PASS")


if __name__ == "__main__":
    test_identity()
    test_x_star_preserved_various_t()
    test_x_star_and_minus_x_star_both_preserved()
    test_all_pm1_vertices_preserved()
    test_composition_of_box_ops()
    test_mixed_op1_op2_composition()
    test_center_moves_in_coord_direction()
    test_op2_independent_of_x_star()
    test_op2_symmetric_in_x_star_sign()
    test_validity_bound_raises()
    print("\nALL TESTS PASS")

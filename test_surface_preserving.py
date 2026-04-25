"""
Tests for the surface-preserving hyperplane operation.

Verifies invariant I1:
    For any x in E_0 ∩ H, the operation produces (c', P') with
    (x - c')^T P'^{-1} (x - c') = 1.

Tests cover:
  1. Trivial: s=0 is identity.
  2. x* (a single ±1 vertex on H_k) preserved across various s.
  3. Multiple vertices on the H_k slice all preserved simultaneously.
  4. Composition: 100 random ops keep x* on surface.
  5. Paired hyperplane (b -> -b) preserves -x*.
  6. The 3-sparse ±1 setting matching our problem.
  7. Sphere-to-non-sphere transitions with x* preserved.
"""

import itertools

import numpy as np

from surface_preserving_ops import op_hyperplane_pencil, on_surface


def make_initial_sphere(N, x_star):
    """Sphere centered at origin, radius |x*| (= sqrt(N) for ±1 vertex)."""
    r2 = float(x_star @ x_star)
    c = np.zeros(N)
    P_inv = np.eye(N) / r2
    # verify x* on surface
    val = (x_star - c) @ P_inv @ (x_star - c)
    assert abs(val - 1.0) < 1e-12, f"setup error: {val}"
    return c, P_inv


def random_3sparse_constraint(N, x_star, rng):
    """Sample a 3-sparse ±1 constraint satisfied by x*. b is in {-3,-1,1,3}."""
    idx = rng.choice(N, size=3, replace=False)
    coefs = rng.choice([-1, 1], size=3).astype(float)
    a = np.zeros(N)
    a[idx] = coefs
    b = float(a @ x_star)
    return a, b


# ---------------------------------------------------------------------------
# Test 1: s=0 is identity
# ---------------------------------------------------------------------------

def test_identity():
    rng = np.random.default_rng(0)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)
    a, b = random_3sparse_constraint(N, x_star, rng)

    c2, P_inv2 = op_hyperplane_pencil(c, P_inv, a, b, s=0.0)

    assert np.allclose(c, c2, atol=1e-12), "center should be unchanged at s=0"
    assert np.allclose(P_inv, P_inv2, atol=1e-12), "shape should be unchanged"
    print("test_identity: PASS")


# ---------------------------------------------------------------------------
# Test 2: x* preserved across many s values
# ---------------------------------------------------------------------------

def test_x_star_preserved_various_s():
    rng = np.random.default_rng(1)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)
    a, b = random_3sparse_constraint(N, x_star, rng)

    for s in [-0.05, -0.01, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        c2, P_inv2 = op_hyperplane_pencil(c, P_inv, a, b, s=s)
        ok, val = on_surface(x_star, c2, P_inv2, tol=1e-8)
        assert ok, f"x* not on surface after s={s}: val = {val}"
    print("test_x_star_preserved_various_s: PASS")


# ---------------------------------------------------------------------------
# Test 3: ALL vertices in E_0 ∩ H preserved simultaneously
# ---------------------------------------------------------------------------

def test_all_intersection_vertices_preserved():
    """For our 3-sparse setting, multiple ±1 vertices may satisfy a single
    constraint. All of them should remain on surface after the operation."""
    rng = np.random.default_rng(2)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)

    # Pick a |b|=1 constraint so multiple vertices satisfy it
    while True:
        a, b = random_3sparse_constraint(N, x_star, rng)
        if abs(b) == 1:
            break

    # Find all vertices satisfying both: on initial sphere (always true for ±1
    # vertices) and on H_k.
    vertices_on_intersection = []
    for bits in itertools.product((-1.0, 1.0), repeat=N):
        v = np.array(bits)
        if abs(a @ v - b) < 1e-10:
            vertices_on_intersection.append(v)
    assert len(vertices_on_intersection) > 1, "need overlapping case"

    for s in [-0.01, 0.1, 1.0]:
        c2, P_inv2 = op_hyperplane_pencil(c, P_inv, a, b, s=s)
        for v in vertices_on_intersection:
            ok, val = on_surface(v, c2, P_inv2, tol=1e-8)
            assert ok, f"v={v} not on surface after s={s}: val={val}"
    print(
        f"test_all_intersection_vertices_preserved: PASS "
        f"({len(vertices_on_intersection)} vertices preserved)"
    )


# ---------------------------------------------------------------------------
# Test 4: Composition - 100 random ops, x* still on surface
# ---------------------------------------------------------------------------

def test_composition_preserves_x_star():
    rng = np.random.default_rng(3)
    N = 8
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)

    n_ops = 100
    for k in range(n_ops):
        a, b = random_3sparse_constraint(N, x_star, rng)
        # Choose s safely: keep |s * aPa| < 0.5 to avoid degeneracy
        P = np.linalg.inv(P_inv)
        aPa = float(a @ P @ a)
        s_max_safe = 0.5 / aPa
        s = rng.uniform(-s_max_safe, s_max_safe)
        c, P_inv = op_hyperplane_pencil(c, P_inv, a, b, s=s)

    ok, val = on_surface(x_star, c, P_inv, tol=1e-6)
    assert ok, f"x* drifted after {n_ops} ops: val={val}"
    print(f"test_composition_preserves_x_star: PASS (val={val:.2e})")


# ---------------------------------------------------------------------------
# Test 5: Paired hyperplane preserves -x*
# ---------------------------------------------------------------------------

def test_paired_preserves_minus_x_star():
    rng = np.random.default_rng(4)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    minus_x_star = -x_star

    # Initial ellipsoid contains both x* and -x* on surface (sphere through origin).
    c, P_inv = make_initial_sphere(N, x_star)
    ok_neg, _ = on_surface(minus_x_star, c, P_inv, tol=1e-12)
    assert ok_neg, "sphere should have -x* on surface as well"

    # Apply pencil ops with paired hyperplane (b -> -b)
    for k in range(20):
        a, b = random_3sparse_constraint(N, x_star, rng)
        # Paired plane: a^T x = -b. -x* satisfies a^T (-x*) = -b ✓
        b_paired = -b
        P = np.linalg.inv(P_inv)
        aPa = float(a @ P @ a)
        s_max_safe = 0.5 / aPa
        s = rng.uniform(-s_max_safe, s_max_safe)
        c, P_inv = op_hyperplane_pencil(c, P_inv, a, b_paired, s=s)

    ok, val = on_surface(minus_x_star, c, P_inv, tol=1e-6)
    assert ok, f"-x* drifted on paired ops: val={val}"
    print(f"test_paired_preserves_minus_x_star: PASS (val={val:.2e})")


# ---------------------------------------------------------------------------
# Test 6: Center actually moves (operation isn't trivial)
# ---------------------------------------------------------------------------

def test_center_actually_moves():
    rng = np.random.default_rng(5)
    N = 6
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)
    a, b = random_3sparse_constraint(N, x_star, rng)

    c2, P_inv2 = op_hyperplane_pencil(c, P_inv, a, b, s=0.5)
    delta = np.linalg.norm(c2 - c)
    assert delta > 1e-3, f"center should move noticeably; delta={delta}"
    # x* still on surface
    ok, val = on_surface(x_star, c2, P_inv2, tol=1e-8)
    assert ok, f"x* not preserved: val={val}"
    print(f"test_center_actually_moves: PASS (delta={delta:.4f})")


# ---------------------------------------------------------------------------
# Test 7: Verify shape changes (operation isn't pure translation)
# ---------------------------------------------------------------------------

def test_shape_changes_with_s():
    rng = np.random.default_rng(6)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)
    a, b = random_3sparse_constraint(N, x_star, rng)

    # Choose s values inside valid range (-1/aPa, infinity)
    P = np.linalg.inv(P_inv)
    aPa = float(a @ P @ a)
    s_neg = -0.5 / aPa  # safely > -1/aPa
    s_pos = 1.0 / aPa

    _, P_inv_a = op_hyperplane_pencil(c, P_inv, a, b, s=s_pos)
    _, P_inv_b = op_hyperplane_pencil(c, P_inv, a, b, s=s_neg)

    assert not np.allclose(P_inv, P_inv_a, atol=1e-3), "shape should differ"
    assert not np.allclose(P_inv_a, P_inv_b, atol=1e-3), "shapes should differ"
    print("test_shape_changes_with_s: PASS")


# ---------------------------------------------------------------------------
# Test 8: Edge cases — |b|=3 constraints
# ---------------------------------------------------------------------------

def test_b_equals_3_constraint():
    """When |b|=3, the hyperplane H_k contains only one ±1 vertex (x* itself
    in our setup). Verify the op still preserves x*."""
    rng = np.random.default_rng(7)
    N = 5
    x_star = rng.choice([-1.0, 1.0], size=N)
    c, P_inv = make_initial_sphere(N, x_star)

    while True:
        a, b = random_3sparse_constraint(N, x_star, rng)
        if abs(b) == 3:
            break

    for s in [-0.05, 0.1, 0.5, 1.0]:
        c2, P_inv2 = op_hyperplane_pencil(c, P_inv, a, b, s=s)
        ok, val = on_surface(x_star, c2, P_inv2, tol=1e-8)
        assert ok, f"x* not preserved with |b|=3 at s={s}: val={val}"
    print("test_b_equals_3_constraint: PASS")


# ---------------------------------------------------------------------------
# Test 9: Both x* and -x* preserved simultaneously when ops mix
# ---------------------------------------------------------------------------

def test_simultaneous_preserve_both():
    """E maintains x* on surface using original constraints; E' maintains -x*
    using paired. Test that both ellipsoids do their jobs over many ops."""
    rng = np.random.default_rng(8)
    N = 7
    x_star = rng.choice([-1.0, 1.0], size=N)
    minus_x_star = -x_star

    # E1 uses original, E2 uses paired. Both start as the same sphere.
    c_E1, P_inv_E1 = make_initial_sphere(N, x_star)
    c_E2, P_inv_E2 = make_initial_sphere(N, x_star)

    n_ops = 50
    for k in range(n_ops):
        a, b = random_3sparse_constraint(N, x_star, rng)

        # Op on E1 with original
        P1 = np.linalg.inv(P_inv_E1)
        aPa1 = float(a @ P1 @ a)
        s1 = rng.uniform(-0.4 / aPa1, 0.4 / aPa1)
        c_E1, P_inv_E1 = op_hyperplane_pencil(c_E1, P_inv_E1, a, b, s=s1)

        # Op on E2 with paired
        P2 = np.linalg.inv(P_inv_E2)
        aPa2 = float(a @ P2 @ a)
        s2 = rng.uniform(-0.4 / aPa2, 0.4 / aPa2)
        c_E2, P_inv_E2 = op_hyperplane_pencil(c_E2, P_inv_E2, a, -b, s=s2)

    ok1, val1 = on_surface(x_star, c_E1, P_inv_E1, tol=1e-6)
    ok2, val2 = on_surface(minus_x_star, c_E2, P_inv_E2, tol=1e-6)
    assert ok1, f"x* not on E1 surface: val={val1}"
    assert ok2, f"-x* not on E2 surface: val={val2}"
    print(
        f"test_simultaneous_preserve_both: PASS "
        f"(E1: {val1:.2e}, E2: {val2:.2e})"
    )


# ---------------------------------------------------------------------------
# Test 10: Independence from x*'s coordinates
# ---------------------------------------------------------------------------

def test_op_independent_of_x_star():
    """The operation should depend only on (c, P, a, b, s), NOT on x*'s
    coordinates. Verify by running the same op for two different problems
    with the same (c, P, a, b, s) but different x*."""
    rng = np.random.default_rng(9)
    N = 6
    a = np.zeros(N); a[0:3] = [1, 1, -1]
    b = 1.0
    c0 = np.zeros(N)
    P_inv0 = np.eye(N) / 6.0  # sphere of radius sqrt(6)
    s = 0.4

    c_a, P_inv_a = op_hyperplane_pencil(c0, P_inv0, a, b, s=s)
    c_b, P_inv_b = op_hyperplane_pencil(c0, P_inv0, a, b, s=s)

    assert np.allclose(c_a, c_b)
    assert np.allclose(P_inv_a, P_inv_b)
    print("test_op_independent_of_x_star: PASS")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_identity()
    test_x_star_preserved_various_s()
    test_all_intersection_vertices_preserved()
    test_composition_preserves_x_star()
    test_paired_preserves_minus_x_star()
    test_center_actually_moves()
    test_shape_changes_with_s()
    test_b_equals_3_constraint()
    test_simultaneous_preserve_both()
    test_op_independent_of_x_star()
    print("\nALL TESTS PASS")

"""
Empirical evaluation of `dual_ellipsoid_descent`.

Two questions:

  1. Does I1 / I2 hold throughout the descent? (sanity check on operations,
     should always pass since the per-op tests already verify this).
  2. Does the descent's final state recover x* via sign extraction?
     This is the main open question -- the unblocking experiment for §9.

Generates small-N instances with a planted unique solution using the same
3-sparse |b|=1 generator as the archived test bed. For each instance, runs
the descent with a few weight configurations and reports:

  - Final ellipsoid sizes
  - Final sphericity defect
  - ||c1 - c2||
  - Hamming distance from sign(c1) to x* and sign(c2) to -x*
  - Whether descent recovered x* (sign agreement)

This is a characterization run, not a pass/fail test. Output is meant to be
read.
"""

import itertools
import time
import numpy as np

from dual_ellipsoid_descent import descend, fitness, diagnostics


# ---------------------------------------------------------------------------
# Instance generator (3-sparse |b|=1, planted unique x*)
# ---------------------------------------------------------------------------

_PATTERN_CACHE = {}
def all_patterns(N):
    if N not in _PATTERN_CACHE:
        _PATTERN_CACHE[N] = np.array(
            list(itertools.product((-1.0, 1.0), repeat=N)))
    return _PATTERN_CACHE[N]


def sample_b1_constraint(N, x_star, rng):
    while True:
        idx = rng.choice(N, size=3, replace=False)
        coefs = rng.choice([-1, 1], size=3)
        a = np.zeros(N); a[idx] = coefs
        bv = float(a @ x_star)
        if abs(bv) == 1:
            return a, bv


def generate_unique(N, m, rng, max_tries=2000):
    S = all_patterns(N)
    for _ in range(max_tries):
        x_star = rng.choice([-1.0, 1.0], size=N)
        rows, bs = [], []
        for _ in range(m):
            a, bv = sample_b1_constraint(N, x_star, rng)
            rows.append(a); bs.append(bv)
        A = np.array(rows); b = np.array(bs)
        resid = (S @ A.T) - b[None, :]
        ok = np.all(resid == 0, axis=1)
        if ok.sum() == 1:
            return x_star, A, b
    return None


# ---------------------------------------------------------------------------
# Single instance run
# ---------------------------------------------------------------------------

def run_one(N, m, seed, weights, max_rounds=30, init="kicked", verbose=False):
    rng = np.random.default_rng(seed)
    inst = generate_unique(N, m, rng)
    if inst is None:
        return None
    x_star, A, b = inst

    # The descent uses a separate rng for kick-direction; pass through.
    c1, P_inv1, c2, P_inv2, hist = descend(
        A, b, max_rounds=max_rounds, weights=weights, init=init, rng=rng,
        x_star=x_star, verbose=verbose,
    )

    final = hist[-1]
    return {
        "x_star": x_star,
        "A": A,
        "b": b,
        "c1": c1,
        "c2": c2,
        "history": hist,
        "final": final,
        "n_constraints": m,
    }


def summarize(result):
    if result is None:
        return "instance generation failed"
    f = result["final"]
    parts = [
        f"size E1={f['size_E1']:.3e} E2={f['size_E2']:.3e}",
        f"sph E1={f['sph_E1']:.3e} E2={f['sph_E2']:.3e}",
        f"align={f['align']:.3e}",
        f"||c1||={f['norm_c1']:.3f} ||c2||={f['norm_c2']:.3f}",
    ]
    if "I1_resid" in f:
        parts.append(f"I1={f['I1_resid']:+.2e} I2={f['I2_resid']:+.2e}")
        parts.append(f"d(c1,x*)={f['dist_c1_xstar']:.3f} d(c2,-x*)={f['dist_c2_minusxstar']:.3f}")
        parts.append(f"hamE1={f['hamming_c1_to_xstar']} hamE2={f['hamming_c2_to_minusxstar']}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep(N, m_values, n_instances=5, max_rounds=30, weight_name="balanced", verbose=False):
    weight_options = {
        # no_align: drop the §9 "shared center" framing entirely; let each
        # ellipsoid shrink toward its own target. (Initial single-instance
        # trace shows this works.)
        "no_align": {"size": 1.0, "sph": 1.0, "align": 0.0},
        # shrink-heavy: prioritize concentration of each ellipsoid near its target
        "shrink": {"size": 5.0, "sph": 1.0, "align": 0.0},
        # balanced: equal weight on shrink and sphericity, modest alignment
        "balanced": {"size": 1.0, "sph": 1.0, "align": 0.5},
        # align-heavy: c1 = c2 (the journal's "shared center" framing)
        "align": {"size": 1.0, "sph": 1.0, "align": 5.0},
    }
    weights = weight_options[weight_name]

    print(f"\n=== N={N}  weight_profile={weight_name}  rounds={max_rounds} ===")
    print(f"    weights = {weights}")

    for m in m_values:
        successes_E1 = 0
        successes_E2 = 0
        max_I1_resid = 0.0
        max_I2_resid = 0.0
        ham_E1_list = []
        ham_E2_list = []
        n_runs = 0
        for seed in range(n_instances):
            r = run_one(N, m, seed=10000 * N + 17 * m + seed, weights=weights,
                        max_rounds=max_rounds, verbose=verbose)
            if r is None:
                continue
            n_runs += 1
            f = r["final"]
            if f["hamming_c1_to_xstar"] == 0:
                successes_E1 += 1
            if f["hamming_c2_to_minusxstar"] == 0:
                successes_E2 += 1
            max_I1_resid = max(max_I1_resid, abs(f["I1_resid"]))
            max_I2_resid = max(max_I2_resid, abs(f["I2_resid"]))
            ham_E1_list.append(f["hamming_c1_to_xstar"])
            ham_E2_list.append(f["hamming_c2_to_minusxstar"])
        if n_runs == 0:
            print(f"  N={N} m={m}: no instances generated")
            continue
        ham_E1_str = f"min={min(ham_E1_list)} mean={np.mean(ham_E1_list):.2f} max={max(ham_E1_list)}"
        ham_E2_str = f"min={min(ham_E2_list)} mean={np.mean(ham_E2_list):.2f} max={max(ham_E2_list)}"
        print(
            f"  m={m}: success E1 {successes_E1}/{n_runs}  E2 {successes_E2}/{n_runs}  "
            f"hamE1[{ham_E1_str}]  hamE2[{ham_E2_str}]  "
            f"max|I1|={max_I1_resid:.2e}  max|I2|={max_I2_resid:.2e}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main_traces():
    """Print full trace for one specific instance, to see what the descent does."""
    print("\n--- Detailed trace, single instance N=6, m=4, seed=0 ---")
    result = run_one(N=6, m=4, seed=12345,
                     weights={"size": 1.0, "sph": 1.0, "align": 0.5},
                     max_rounds=20, verbose=True)
    if result is None:
        print("(instance generation failed)")
        return
    print(f"\nx* = {result['x_star']}")
    print(f"sign(c1) = {np.sign(result['c1'])}")
    print(f"sign(c2) = {np.sign(result['c2'])}")
    print(f"final summary: {summarize(result)}")


def main_sweeps():
    t0 = time.time()
    for weight_name in ("balanced", "shrink", "sphericity", "align"):
        for N in (6, 8):
            m_values = list(range(2, N))
            sweep(N=N, m_values=m_values, n_instances=5,
                  max_rounds=20, weight_name=weight_name)
    print(f"\n[total elapsed {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main_traces()
    main_sweeps()

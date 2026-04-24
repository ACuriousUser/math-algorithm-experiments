"""Faster experiment.  Pre-compute all sign patterns as a matrix; use
vectorised linear-algebra for enumeration and for the discrete residual.
"""

import itertools
import time
import numpy as np

from experiment_hillclimb import (
    continuous_fitness, count_local_optima, hillclimb,
)


_PATTERN_CACHE = {}
def all_patterns(N):
    if N not in _PATTERN_CACHE:
        _PATTERN_CACHE[N] = np.array(
            list(itertools.product((-1.0, 1.0), repeat=N)))
    return _PATTERN_CACHE[N]


def sample_b1_constraint(N, x_star, rng):
    """Random 3-var |b|=1 constraint satisfied by x_star.  Returns (a, b)."""
    while True:
        idx = rng.choice(N, size=3, replace=False)
        coefs = rng.choice([-1, 1], size=3)
        a = np.zeros(N); a[idx] = coefs
        bv = float(a @ x_star)
        if abs(bv) == 1:
            return a, bv


def generate(N, m, rng, max_retries=2000):
    """Keep sampling until we find an instance with a unique hypercube solution."""
    S = all_patterns(N)           # (2^N, N)
    for _ in range(max_retries):
        x_star = rng.choice([-1.0, 1.0], size=N)
        rows, bs = [], []
        for _ in range(m):
            a, bv = sample_b1_constraint(N, x_star, rng)
            rows.append(a); bs.append(bv)
        A = np.array(rows); b = np.array(bs)
        # vectorised: each pattern is a row in S, check A @ pattern == b
        resid = (S @ A.T) - b[None, :]     # (2^N, m)
        ok = np.all(resid == 0, axis=1)
        if ok.sum() == 1:
            return x_star, A, b
    return None


def discrete_residual_vec(A, b, S):
    """||A s - b||^2 for every sign pattern s (row of S).  Returns 1-D array."""
    resid = (S @ A.T) - b[None, :]
    return np.sum(resid * resid, axis=1)


def continuous_fitness_vec(A, b, S):
    """Continuous fitness for every sign pattern (row of S).  Returns 1-D array.
    Uses the custom Newton solver one pattern at a time."""
    out = np.empty(len(S))
    for i, s in enumerate(S):
        out[i] = continuous_fitness(s, A, b)
    return out


def run(N, m, seed, n_instances=8, n_trials=10):
    rng = np.random.default_rng(seed)
    S = all_patterns(N)
    stats = dict(N=N, m=m, n_instances_target=n_instances,
                 found=0, tried=0,
                 disc_locmin=[], cont_locmax=[],
                 cont_xstar_feasible=0, cont_global_is_xstar=0,
                 disc_global_is_xstar=0,
                 disc_hc_total=0, disc_hc_success=0,
                 cont_hc_total=0, cont_hc_success=0,
                 fitness_gap_xstar_vs_best=[])
    pats = [S[i] for i in range(len(S))]

    while stats["found"] < n_instances:
        stats["tried"] += 1
        if stats["tried"] > 20 * n_instances:
            break
        inst = generate(N, m, rng, max_retries=2000)
        if inst is None:
            break      # we've essentially exhausted the search space
        x_star, A, b = inst
        stats["found"] += 1

        d_vals = discrete_residual_vec(A, b, S)
        c_vals = continuous_fitness_vec(A, b, S)
        ix = int(np.where(np.all(S == x_star, axis=1))[0][0])

        # Discrete global
        dmin = d_vals.min()
        d_arg = np.where(d_vals <= dmin + 1e-9)[0]
        if len(d_arg) == 1 and d_arg[0] == ix:
            stats["disc_global_is_xstar"] += 1
        # Continuous feasibility / global
        if np.isfinite(c_vals[ix]):
            stats["cont_xstar_feasible"] += 1
        finite = np.isfinite(c_vals)
        if finite.any():
            cmax = c_vals[finite].max()
            c_arg = np.where(np.isfinite(c_vals) & (c_vals >= cmax - 1e-6))[0]
            if len(c_arg) == 1 and c_arg[0] == ix:
                stats["cont_global_is_xstar"] += 1
            if np.isfinite(c_vals[ix]):
                stats["fitness_gap_xstar_vs_best"].append(float(cmax - c_vals[ix]))
        stats["disc_locmin"].append(
            len(count_local_optima(pats, d_vals, find_max=False)))
        stats["cont_locmax"].append(
            len(count_local_optima(pats, c_vals, find_max=True)))

        # Hillclimb
        def disc_fn(s, A=A, b=b): r = A @ s - b; return float(r @ r)
        def cont_fn(s, A=A, b=b): return continuous_fitness(s, A, b)

        for _ in range(n_trials):
            s0 = rng.choice([-1.0, 1.0], size=N)
            sf, _, _ = hillclimb(s0, A, b, disc_fn, maximize=False)
            stats["disc_hc_total"] += 1
            if np.allclose(sf, x_star): stats["disc_hc_success"] += 1
            sf2, _, _ = hillclimb(s0, A, b, cont_fn, maximize=True)
            stats["cont_hc_total"] += 1
            if np.allclose(sf2, x_star): stats["cont_hc_success"] += 1
    return stats


def fmt(s):
    n = s["found"]
    if n == 0:
        return (f"N={s['N']} m={s['m']}: no unique instances found "
                f"(gave up after {s['tried']} outer tries)")
    lmd = s["disc_locmin"]; lmc = s["cont_locmax"]
    g = s["fitness_gap_xstar_vs_best"]
    gap_str = (f"fitness-gap(best - x*) mean={np.mean(g):+.4f} min={min(g):+.4f} max={max(g):+.4f}"
               if g else "fitness-gap: (x* never continuous-feasible)")
    return (
      f"=== N={s['N']}  m={s['m']}  |b|=1 only   "
      f"(found {n} unique instances from {s['tried']} tries) ===\n"
      f"    discrete   : global=x* {s['disc_global_is_xstar']}/{n}, "
      f"locmin min/mean/max = {min(lmd)}/{np.mean(lmd):.2f}/{max(lmd)},  "
      f"hc {s['disc_hc_success']}/{s['disc_hc_total']}\n"
      f"    continuous : x*-feasible {s['cont_xstar_feasible']}/{n}, "
      f"global=x* {s['cont_global_is_xstar']}/{n}, "
      f"locmax min/mean/max = {min(lmc)}/{np.mean(lmc):.2f}/{max(lmc)},  "
      f"hc {s['cont_hc_success']}/{s['cont_hc_total']}\n"
      f"    {gap_str}"
    )


if __name__ == "__main__":
    t0 = time.time()
    for N in (6, 8, 10):
        for m in range(3, N):
            s = run(N, m, seed=30000 + 100*N + m, n_instances=8, n_trials=8)
            print(fmt(s), flush=True)
        print(flush=True)
    print(f"[total elapsed {time.time()-t0:.1f}s]", flush=True)

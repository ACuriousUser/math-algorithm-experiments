"""Focused experiment.  Only |b|=1 constraints, varying m from moderate to
large, report whether
  - x* is unique over {-1,+1}^N (if not, skip)
  - continuous fitness is feasible at s=x*
  - continuous global max equals x* (when it is feasible)
  - number of Hamming-local maxima / minima
  - hillclimb success rates
"""

import itertools
import time
import numpy as np

from experiment_hillclimb import (
    all_solutions, discrete_residual, continuous_fitness,
    enumerate_values, count_local_optima, hillclimb,
)


def sample_b1_constraint(N, x_star, rng):
    """3-variable |b|=1 constraint.  We keep sampling until |a . x_star| = 1."""
    for _ in range(50):
        idx = rng.choice(N, size=3, replace=False)
        coefs = rng.choice([-1, 1], size=3)
        a = np.zeros(N); a[idx] = coefs
        bv = float(a @ x_star)
        if abs(bv) == 1:
            return a, bv
    return None


def generate(N, m, rng, max_retries=200):
    """Build m |b|=1 constraints, accept only if x* uniquely satisfies them
    over the hypercube.  Returns (x_star, A, b) or None."""
    for _ in range(max_retries):
        x_star = rng.choice([-1.0, 1.0], size=N)
        rows, bs = [], []
        bad = False
        for _ in range(m):
            r = sample_b1_constraint(N, x_star, rng)
            if r is None:
                bad = True; break
            a, bv = r; rows.append(a); bs.append(bv)
        if bad:
            continue
        A = np.array(rows); b = np.array(bs)
        sols = all_solutions(A, b)
        if len(sols) == 1 and np.allclose(sols[0], x_star):
            return x_star, A, b
    return None


def run(N, m, seed, n_instances=10, n_trials=10):
    rng = np.random.default_rng(seed)
    stats = dict(N=N, m=m, n_instances_target=n_instances,
                 found=0, tried=0,
                 disc_locmin=[], cont_locmax=[],
                 cont_xstar_feasible=0, cont_global_is_xstar=0,
                 disc_global_is_xstar=0,
                 disc_hc_total=0, disc_hc_success=0,
                 cont_hc_total=0, cont_hc_success=0,
                 fitness_gap_xstar_vs_best=[])
    while stats["found"] < n_instances:
        stats["tried"] += 1
        if stats["tried"] > 500 * n_instances:
            break
        inst = generate(N, m, rng, max_retries=200)
        if inst is None:
            continue
        x_star, A, b = inst
        stats["found"] += 1

        pats, d_vals = enumerate_values(A, b, discrete_residual)
        _, c_vals = enumerate_values(A, b, continuous_fitness)
        ix = next(i for i, p in enumerate(pats) if np.allclose(p, x_star))

        # Discrete
        d_min = d_vals.min()
        if np.sum(d_vals == d_min) == 1 and d_vals[ix] == d_min:
            stats["disc_global_is_xstar"] += 1
        stats["disc_locmin"].append(
            len(count_local_optima(pats, d_vals, find_max=False)))

        # Continuous
        if np.isfinite(c_vals[ix]):
            stats["cont_xstar_feasible"] += 1
        finite = np.isfinite(c_vals)
        if finite.any():
            cmax = c_vals[finite].max()
            argmax = [i for i, v in enumerate(c_vals)
                      if np.isfinite(v) and v >= cmax - 1e-6]
            if len(argmax) == 1 and argmax[0] == ix:
                stats["cont_global_is_xstar"] += 1
            # gap from x* fitness to the global max
            if np.isfinite(c_vals[ix]):
                stats["fitness_gap_xstar_vs_best"].append(
                    float(cmax - c_vals[ix]))
        stats["cont_locmax"].append(
            len(count_local_optima(pats, c_vals, find_max=True)))

        for _ in range(n_trials):
            s0 = rng.choice([-1.0, 1.0], size=N)
            sf, _, _ = hillclimb(s0, A, b, discrete_residual, maximize=False)
            stats["disc_hc_total"] += 1
            if np.allclose(sf, x_star): stats["disc_hc_success"] += 1
            sf2, _, _ = hillclimb(s0, A, b, continuous_fitness, maximize=True)
            stats["cont_hc_total"] += 1
            if np.allclose(sf2, x_star): stats["cont_hc_success"] += 1
    return stats


def fmt(s):
    n = s["found"]
    if n == 0:
        return f"N={s['N']} m={s['m']}: no unique instances (of {s['tried']} tried)"
    lmd = s["disc_locmin"]; lmc = s["cont_locmax"]
    g = s["fitness_gap_xstar_vs_best"]
    gap_str = (f"fitness-gap(x*-to-global) mean={np.mean(g):+.3f} max={max(g):+.3f}"
               if g else "fitness-gap: (no feasible x*)")
    return (
      f"=== N={s['N']}  m={s['m']}  |b|=1 only   (found {n}/{s['n_instances_target']} "
      f"in {s['tried']} tries) ===\n"
      f"    discrete   global=x*: {s['disc_global_is_xstar']}/{n}   "
      f"locmin: min={min(lmd)} mean={np.mean(lmd):.2f} max={max(lmd)}   "
      f"hc {s['disc_hc_success']}/{s['disc_hc_total']}\n"
      f"    continuous x*-feasible: {s['cont_xstar_feasible']}/{n}   "
      f"global=x*: {s['cont_global_is_xstar']}/{n}   "
      f"locmax: min={min(lmc)} mean={np.mean(lmc):.2f} max={max(lmc)}   "
      f"hc {s['cont_hc_success']}/{s['cont_hc_total']}\n"
      f"    {gap_str}"
    )


if __name__ == "__main__":
    t0 = time.time()
    # Sweep m from smallest-that-gives-uniqueness up to N-1
    for N in (6, 8, 10):
        for m in range(max(2, N // 3), N):
            s = run(N, m, seed=20000 + 100*N + m, n_instances=8, n_trials=10)
            print(fmt(s), flush=True)
        print(flush=True)
    print(f"[elapsed {time.time()-t0:.1f}s]")

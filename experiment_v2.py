"""Revised experiment: fix the number of constraints m (not "keep adding until
unique"), accept only instances where x* is the unique hypercube solution,
and report what fraction of sampled instances actually meet that uniqueness
condition.  This is the regime the algorithm is designed for.
"""

import itertools
import time
import numpy as np

from experiment_hillclimb import (
    sample_constraint, all_solutions,
    discrete_residual, continuous_fitness,
    enumerate_values, count_local_optima, hillclimb,
)


def generate_fixed_m(N, m, rng, force_b1, tries=2000):
    """Try to sample an instance with exactly m constraints and a unique
    hypercube solution.  Returns (x_star, A, b) or None after `tries` fails."""
    for _ in range(tries):
        x_star = rng.choice([-1.0, 1.0], size=N)
        rows, bvec = [], []
        attempts = 0
        while len(rows) < m and attempts < 10 * m:
            a, bv = sample_constraint(N, x_star, rng)
            if force_b1 and abs(bv) > 1:
                attempts += 1
                continue
            rows.append(a); bvec.append(bv)
            attempts += 1
        if len(rows) < m:
            continue
        A = np.array(rows); b = np.array(bvec)
        sols = all_solutions(A, b)
        if len(sols) == 1 and np.allclose(sols[0], x_star):
            return x_star, A, b
    return None


def report(N, m, force_b1, seed, n_instances=15, n_trials=20):
    local = np.random.default_rng(seed)
    found = 0; samples_tried = 0
    stats = {
        "m": m, "N": N, "force_b1": force_b1,
        "disc_n_local_min": [],
        "cont_n_local_max": [],
        "cont_feasible_xstar": 0,
        "disc_global_unique_xstar": 0,
        "cont_global_unique_xstar": 0,
        "disc_hc_success": 0, "disc_hc_total": 0,
        "cont_hc_success": 0, "cont_hc_total": 0,
        "cont_xstar_is_feasible_hamming_neighbors": [],
    }
    while found < n_instances:
        samples_tried += 1
        inst = generate_fixed_m(N, m, local, force_b1=force_b1, tries=400)
        if inst is None:
            if samples_tried > 100 * n_instances:
                break
            continue
        x_star, A, b = inst
        found += 1

        pats, d_vals = enumerate_values(A, b, discrete_residual)
        _, c_vals = enumerate_values(A, b, continuous_fitness)

        # Indices
        ix = next(i for i, p in enumerate(pats) if np.allclose(p, x_star))

        # Discrete unique global?
        d_min = d_vals.min()
        d_argmin = [i for i, v in enumerate(d_vals) if v <= d_min + 1e-9]
        if len(d_argmin) == 1 and d_argmin[0] == ix:
            stats["disc_global_unique_xstar"] += 1

        # Continuous: feasible at x*?
        if np.isfinite(c_vals[ix]):
            stats["cont_feasible_xstar"] += 1
        # How many Hamming-1 neighbors of x* are also continuous-feasible?
        feasible_nbrs = 0
        for j in range(N):
            t = x_star.copy(); t[j] = -t[j]
            it = next(i for i, p in enumerate(pats) if np.allclose(p, t))
            if np.isfinite(c_vals[it]):
                feasible_nbrs += 1
        stats["cont_xstar_is_feasible_hamming_neighbors"].append(feasible_nbrs)

        # Continuous global unique?
        finite = np.isfinite(c_vals)
        if finite.any():
            c_max = c_vals[finite].max()
            c_argmax = [i for i, v in enumerate(c_vals)
                        if np.isfinite(v) and v >= c_max - 1e-6]
            if len(c_argmax) == 1 and c_argmax[0] == ix:
                stats["cont_global_unique_xstar"] += 1

        stats["disc_n_local_min"].append(
            len(count_local_optima(pats, d_vals, find_max=False)))
        stats["cont_n_local_max"].append(
            len(count_local_optima(pats, c_vals, find_max=True)))

        for _ in range(n_trials):
            s0 = local.choice([-1.0, 1.0], size=N)
            sf, _, _ = hillclimb(s0, A, b, discrete_residual, maximize=False)
            stats["disc_hc_total"] += 1
            if np.allclose(sf, x_star):
                stats["disc_hc_success"] += 1
            sf2, _, _ = hillclimb(s0, A, b, continuous_fitness, maximize=True)
            stats["cont_hc_total"] += 1
            if np.allclose(sf2, x_star):
                stats["cont_hc_success"] += 1

    stats["n_instances_found"] = found
    stats["samples_tried"] = samples_tried
    return stats


def fmt(s):
    if s["n_instances_found"] == 0:
        return f"N={s['N']} m={s['m']} force_b1={s['force_b1']}: no unique instances found"
    n = s["n_instances_found"]
    lm_d = s["disc_n_local_min"]; lm_c = s["cont_n_local_max"]
    nbrs = s["cont_xstar_is_feasible_hamming_neighbors"]
    lines = [
        f"=== N={s['N']}  m={s['m']}  force_b1={s['force_b1']}  "
        f"(found {n} unique-solution instances out of {s['samples_tried']} tries) ===",
        f"    discrete   global=x*: {s['disc_global_unique_xstar']}/{n}   "
        f"local-min: min={min(lm_d)} mean={np.mean(lm_d):.2f} max={max(lm_d)}   "
        f"hc-success {s['disc_hc_success']}/{s['disc_hc_total']}",
        f"    continuous global=x*: {s['cont_global_unique_xstar']}/{n}   "
        f"x*-feasible: {s['cont_feasible_xstar']}/{n}   "
        f"avg feasible Hamming-1 neighbors of x*: {np.mean(nbrs):.2f}/{s['N']}",
        f"    continuous local-max: min={min(lm_c)} mean={np.mean(lm_c):.2f} max={max(lm_c)}   "
        f"hc-success {s['cont_hc_success']}/{s['cont_hc_total']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    t0 = time.time()
    for N in (6, 8, 10):
        for frac in ("N/3", "N/2", "2N/3", "N-2"):
            if frac == "N/3": m = max(2, N // 3)
            elif frac == "N/2": m = N // 2
            elif frac == "2N/3": m = (2 * N) // 3
            else: m = N - 2
            for force_b1 in (False, True):
                s = report(N, m, force_b1, seed=9000 + 10 * N + m +
                           (1 if force_b1 else 0) * 333,
                           n_instances=15, n_trials=10)
                print(fmt(s))
        print()
    print(f"[total elapsed {time.time()-t0:.1f}s]")

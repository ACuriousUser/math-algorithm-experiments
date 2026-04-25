"""Extract a concrete counterexample to the falsified guess-and-flip
ellipsoid-fitness design: an instance where x* is the unique hypercube
solution, and yet the continuous ellipsoid fitness is higher for some
OTHER sign pattern. Produces the example printed in findings.md §3b.

This script tests the OLD (falsified) design. The CURRENT working
design lives in surface_preserving_ops.py / design-journal.md §9."""

import numpy as np
import itertools
from experiment_v4 import generate, all_patterns
from experiment_hillclimb import continuous_fitness, discrete_residual


def find_counterexample(N, m, seed, max_instances=60):
    rng = np.random.default_rng(seed)
    S = all_patterns(N)
    tries = 0
    while tries < max_instances:
        inst = generate(N, m, rng, max_retries=500)
        if inst is None:
            tries += 1
            continue
        x_star, A, b = inst
        c_vals = np.array([continuous_fitness(s, A, b) for s in S])
        ix = int(np.where(np.all(S == x_star, axis=1))[0][0])
        if not np.isfinite(c_vals[ix]):
            tries += 1
            continue
        # find pattern with highest finite fitness
        finite = np.isfinite(c_vals)
        cmax = c_vals[finite].max()
        arg = int(np.where(finite & (c_vals >= cmax - 1e-6))[0][0])
        if arg != ix and cmax > c_vals[ix] + 1e-3:
            return x_star, A, b, S[arg], c_vals, ix, arg
        tries += 1
    return None


if __name__ == "__main__":
    for N, m in [(6, 4), (8, 5), (10, 6)]:
        print(f"\n=== searching for counterexample at N={N}, m={m} ===")
        res = find_counterexample(N, m, seed=77777 + 100*N + m,
                                  max_instances=40)
        if res is None:
            print("  no counterexample found in budget")
            continue
        x_star, A, b, s_best, c_vals, ix, arg = res
        print(f"  x*            = {x_star.astype(int)}")
        print(f"  A =\n{A.astype(int)}")
        print(f"  b = {b.astype(int)}")
        print(f"  A x* - b      = {(A @ x_star - b).astype(int)} (all zero ✓)")
        print(f"  continuous fitness at x*   = {c_vals[ix]:.4f}")
        print(f"  best non-x* pattern s'     = {s_best.astype(int)}")
        print(f"  continuous fitness at s'   = {c_vals[arg]:.4f}  "
              f"(gap +{c_vals[arg] - c_vals[ix]:.4f})")
        print(f"  ||A s' - b||^2             = {discrete_residual(s_best, A, b):.1f}  "
              f"(so s' violates the linear equations)")
        h = int(np.sum(s_best != x_star))
        print(f"  Hamming distance(x*, s')   = {h} (out of {N})")

        # Print the fitness ranking around x*
        finite_mask = np.isfinite(c_vals)
        print(f"  feasible sign patterns (out of 2^{N} = {2**N}): "
              f"{int(finite_mask.sum())}")
        order = np.argsort(-np.where(finite_mask, c_vals, -np.inf))
        print("  top 5 by fitness:")
        S_all = all_patterns(N)
        for k in order[:5]:
            tag = " <-- x*" if k == ix else ""
            print(f"    fitness {c_vals[k]:+.4f}   pattern {S_all[k].astype(int)}{tag}")

"""Continuous-fitness-only phase, split off so we don't redo the discrete run.
Also tracks head-to-head comparisons so we can report on agreement."""

import time
import numpy as np

from experiment_hillclimb import (
    generate_instance, enumerate_values, count_local_optima,
    hillclimb, discrete_residual, continuous_fitness,
)


def run_phase(N, n_instances, force_b1, seed, n_trials=10, amplify=False):
    local = np.random.default_rng(seed)
    out = {
        "N": N, "force_b1": force_b1, "n_instances": n_instances,
        "unique_global_is_xstar": 0,
        "n_local_maxima": [],
        "feasible_xstar": 0,
        "hc_success": 0, "hc_total": 0,
        "disc_locmin_vs_cont_locmax": [],  # (d_min_count, c_max_count)
    }
    done = 0
    while done < n_instances:
        inst = generate_instance(N, local, force_b1=force_b1)
        if inst is None:
            continue
        x_star, A, b = inst

        pats, d_vals = enumerate_values(A, b, discrete_residual)
        _, c_vals = enumerate_values(A, b, continuous_fitness)

        ix = next(i for i, p in enumerate(pats) if np.allclose(p, x_star))
        if np.isfinite(c_vals[ix]):
            out["feasible_xstar"] += 1

        # Continuous unique global (only considering finite values)
        finite = np.isfinite(c_vals)
        if finite.any():
            gmax = c_vals[finite].max()
            argmax = [i for i, v in enumerate(c_vals)
                      if np.isfinite(v) and v >= gmax - 1e-6]
            if len(argmax) == 1 and argmax[0] == ix:
                out["unique_global_is_xstar"] += 1

        c_locmaxs = count_local_optima(pats, c_vals, find_max=True)
        d_locmins = count_local_optima(pats, d_vals, find_max=False)
        out["n_local_maxima"].append(len(c_locmaxs))
        out["disc_locmin_vs_cont_locmax"].append(
            (len(d_locmins), len(c_locmaxs)))

        for _ in range(n_trials):
            s0 = local.choice([-1.0, 1.0], size=N)
            sf, _, _ = hillclimb(s0, A, b, continuous_fitness, maximize=True)
            out["hc_total"] += 1
            if np.allclose(sf, x_star):
                out["hc_success"] += 1

        done += 1
    return out


def fmt(r):
    n = r["n_instances"]
    lm = r["n_local_maxima"]
    line1 = (f"=== N={r['N']}  force_b1={r['force_b1']} ===  "
             f"x*-feasible {r['feasible_xstar']}/{n}   "
             f"x*-unique-global {r['unique_global_is_xstar']}/{n}")
    line2 = (f"    #local-maxima per instance: "
             f"min={min(lm)} mean={np.mean(lm):.2f} max={max(lm)}   "
             f"hc-success: {r['hc_success']}/{r['hc_total']}")
    pair = r["disc_locmin_vs_cont_locmax"]
    dtotal = sum(d for d, _ in pair)
    ctotal = sum(c for _, c in pair)
    line3 = (f"    totals over {n} instances: "
             f"discrete-local-min count={dtotal},  continuous-local-max count={ctotal}")
    return "\n".join([line1, line2, line3])


if __name__ == "__main__":
    t0 = time.time()
    for force_b1 in (False, True):
        tag = "all |b|" if not force_b1 else "|b|<=1 only (overlap regime)"
        print(f"--- regime: {tag} ---")
        for N in (6, 8, 10):
            r = run_phase(N, n_instances=10, force_b1=force_b1,
                          seed=5000 + (1 if force_b1 else 0) * 100 + N,
                          n_trials=10)
            print(fmt(r))
        print()
    print(f"[total elapsed {time.time() - t0:.1f}s]")

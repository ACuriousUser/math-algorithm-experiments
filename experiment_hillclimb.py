"""
Empirical test of guess-and-flip algorithms for the hypercube-vertex-from-
sparse-linear-constraints problem.

Setup:
  - N variables x_i in {-1, +1}
  - m sparse 3-variable {-1,+1}-coefficient equality constraints A x = b
  - x* is the unique solution
  - Goal: recover x* by hill-climbing over sign patterns s in {-1,+1}^N

Two fitness functions to compare:
  1. Discrete residual  R(s) = ||A s - b||^2   (MINIMIZE)
  2. Continuous ellipsoid fitness
        F(s) = max_c  sum_i [log(s_i c_i) + log(1 - s_i c_i)]
               s.t.   A c = b
     which is the analytic-center objective of the region
        { c : s_i c_i in (0, 1), A c = b }
     We MAXIMIZE.  Parameterize u_i = s_i c_i in (0,1);  then the constraint
     becomes  (A * diag(s)) u = b,  and the objective is
        Phi(u) = sum_i [log u_i + log (1 - u_i)].

What to measure (enumerate all 2^N patterns on small N):
  (a) Is s = x* always the unique global optimum of each fitness?
  (b) How many Hamming-1 local optima exist?  Non-x* local optima trap
      the hill-climber.
  (c) How often does hill-climb from a random start converge to x*?
"""

import itertools
import time

import numpy as np

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------

def sample_constraint(N, x_star, rng):
    idx = rng.choice(N, size=3, replace=False)
    coefs = rng.choice([-1, 1], size=3)
    a = np.zeros(N)
    a[idx] = coefs
    b = float(a @ x_star)
    return a, b


def all_solutions(A, b):
    N = A.shape[1]
    sols = []
    for bits in itertools.product((-1.0, 1.0), repeat=N):
        s = np.array(bits)
        if np.allclose(A @ s, b):
            sols.append(s)
    return sols


def generate_instance(N, rng, max_constraints=None, force_b1=False):
    if max_constraints is None:
        max_constraints = 12 * N

    x_star = rng.choice([-1.0, 1.0], size=N)
    A_rows, b_vals = [], []

    for _ in range(max_constraints):
        a, bv = sample_constraint(N, x_star, rng)
        if force_b1 and abs(bv) > 1:
            continue
        A_rows.append(a)
        b_vals.append(bv)
        A = np.array(A_rows)
        bvec = np.array(b_vals)
        sols = all_solutions(A, bvec)
        if len(sols) == 1:
            assert np.allclose(sols[0], x_star)
            return x_star, A, bvec
    return None


# ---------------------------------------------------------------------------
# Discrete residual  R(s) = ||A s - b||^2
# ---------------------------------------------------------------------------

def discrete_residual(s, A, b):
    r = A @ s - b
    return float(r @ r)


# ---------------------------------------------------------------------------
# Continuous fitness with a custom Newton solver on the dual
# ---------------------------------------------------------------------------
#
# Primal:  max sum [log u_i + log(1-u_i)]   s.t.  M u = b,   u in (0,1)
#          where M = A * diag(s)  (size m x N)
#
# Lagrangian stationarity:  1/u_i - 1/(1-u_i) = M_i^T lambda  =: g_i
#    i.e.  (1 - 2u_i) / (u_i(1-u_i)) = g_i
#    => g_i u_i^2 - (g_i + 2) u_i + 1 = 0
#    => u_i(g_i) = [(g_i + 2) - sqrt(g_i^2 + 4)] / (2 g_i)  for g_i != 0
#       u_i(0)   = 1/2
#
#    du_i/dg_i = u_i (1-u_i) / (2 g_i u_i - g_i - 2)       (always negative)
#
# Dual problem is solve  F(lambda) = M u(M^T lambda) - b = 0,  Newton with
# Jacobian  J = M D M^T   where D = diag(du_i/dg_i) < 0.

def u_of_g(g):
    out = np.empty_like(g)
    small = np.abs(g) < 1e-10
    # Taylor: u = 1/2 - g/8 + O(g^3)  (from series of the exact formula)
    out[small] = 0.5 - g[small] / 8.0
    gs = g[~small]
    out[~small] = ((gs + 2.0) - np.sqrt(gs * gs + 4.0)) / (2.0 * gs)
    return out


def du_dg(g, u):
    # u(1-u) / (2 g u - g - 2)
    denom = 2.0 * g * u - g - 2.0
    return u * (1.0 - u) / denom


def continuous_fitness(s, A, b, tol=1e-10, max_iter=60, return_u=False):
    """Solve analytic-center problem on  { c : s_i c_i in (0,1), Ac = b }.
    Returns the maximum objective value; -inf if infeasible.
    """
    N = len(s)
    M = A * s[None, :]          # (m, N)
    m = M.shape[0]

    # Dual Newton from lambda = 0 (u = 1/2 everywhere)
    lam = np.zeros(m)
    for it in range(max_iter):
        g = M.T @ lam                # (N,)
        u = u_of_g(g)
        if np.any(u <= tol) or np.any(u >= 1 - tol):
            # Shrink step toward previous u; but at it=0 this means feasibility failure
            return (-np.inf, None) if return_u else -np.inf

        F = M @ u - b                # (m,)
        err = np.linalg.norm(F)
        if err < 1e-10:
            break

        D = du_dg(g, u)              # (N,), negative
        # Jacobian J = M D M^T,  (m, m)
        J = (M * D[None, :]) @ M.T
        # J is negative definite; Newton step
        try:
            step = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            return (-np.inf, None) if return_u else -np.inf

        # Damped Newton: shrink step until u stays in interior
        alpha = 1.0
        lam_old, u_old, err_old = lam, u, err
        for _ in range(30):
            lam = lam_old - alpha * step
            g = M.T @ lam
            u = u_of_g(g)
            if np.all(u > tol) and np.all(u < 1 - tol):
                F_new = M @ u - b
                if np.linalg.norm(F_new) < (1 - 1e-4 * alpha) * err_old:
                    break
            alpha *= 0.5
        else:
            # Couldn't make progress
            return (-np.inf, None) if return_u else -np.inf

    if np.any(u <= tol) or np.any(u >= 1 - tol):
        return (-np.inf, None) if return_u else -np.inf
    if np.linalg.norm(M @ u - b) > 1e-6:
        return (-np.inf, None) if return_u else -np.inf

    val = float(np.sum(np.log(u)) + np.sum(np.log(1.0 - u)))
    return (val, u) if return_u else val


# ---------------------------------------------------------------------------
# Landscape analysis
# ---------------------------------------------------------------------------

def enumerate_values(A, b, fitness_fn):
    N = A.shape[1]
    patterns = [np.array(p) for p in itertools.product((-1.0, 1.0), repeat=N)]
    values = np.array([fitness_fn(p, A, b) for p in patterns])
    return patterns, values


def count_local_optima(patterns, values, find_max):
    """Return list of indices s.t. every Hamming-1 neighbor is strictly worse.
    If find_max, 'worse' means smaller; else larger."""
    N = len(patterns[0])
    idx_of = {tuple(p): i for i, p in enumerate(patterns)}
    opts = []
    for i, p in enumerate(patterns):
        vp = values[i]
        if not np.isfinite(vp):
            continue
        is_opt = True
        for j in range(N):
            t = p.copy(); t[j] = -t[j]
            vn = values[idx_of[tuple(t)]]
            if find_max:
                if not np.isfinite(vn) or vn >= vp:
                    is_opt = False; break
            else:
                if vn <= vp:
                    is_opt = False; break
        if is_opt:
            opts.append(i)
    return opts


# ---------------------------------------------------------------------------
# Hill-climb from random starts (greedy best-improvement)
# ---------------------------------------------------------------------------

def hillclimb(s0, A, b, fitness_fn, maximize, max_iter=2000):
    s = s0.copy()
    val = fitness_fn(s, A, b)
    steps = 0
    while steps < max_iter:
        best_j, best_val = -1, val
        for j in range(len(s)):
            t = s.copy(); t[j] = -t[j]
            vt = fitness_fn(t, A, b)
            better = (vt > best_val + 1e-12) if maximize else (vt < best_val - 1e-12)
            if better:
                best_j, best_val = j, vt
        if best_j < 0:
            break
        s[best_j] = -s[best_j]
        val = best_val
        steps += 1
    return s, val, steps


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_for_N(N, n_instances, force_b1, seed, do_continuous, n_trials=20):
    local = np.random.default_rng(seed)
    out = {
        "N": N, "force_b1": force_b1, "n_instances": n_instances, "skipped": 0,
        "discrete_unique_global":   0,
        "discrete_n_local_min_per": [],
        "discrete_hc_success":      0,
        "discrete_hc_total":        0,
        "continuous_unique_global": 0,
        "continuous_n_local_max_per": [],
        "continuous_hc_success":    0,
        "continuous_hc_total":      0,
        "continuous_feasible_xstar": 0,
        "failed_instances":         [],   # instances with >1 local optimum
    }
    done = 0
    while done < n_instances:
        inst = generate_instance(N, local, force_b1=force_b1)
        if inst is None:
            out["skipped"] += 1
            continue
        x_star, A, b = inst

        # --- Discrete landscape ---
        pats, d_vals = enumerate_values(A, b, discrete_residual)
        d_locmins = count_local_optima(pats, d_vals, find_max=False)
        out["discrete_n_local_min_per"].append(len(d_locmins))
        # global uniqueness
        gmin = np.min(d_vals)
        argmin = [i for i, v in enumerate(d_vals) if v <= gmin + 1e-9]
        if len(argmin) == 1 and np.allclose(pats[argmin[0]], x_star):
            out["discrete_unique_global"] += 1

        for _ in range(n_trials):
            s0 = local.choice([-1.0, 1.0], size=N)
            sf, _, _ = hillclimb(s0, A, b, discrete_residual, maximize=False)
            out["discrete_hc_total"] += 1
            if np.allclose(sf, x_star):
                out["discrete_hc_success"] += 1

        instance_failed = len(d_locmins) > 1

        if do_continuous:
            pats2, c_vals = enumerate_values(A, b, continuous_fitness)
            # track how often the true x* sign pattern is feasible
            ix = next(i for i, p in enumerate(pats2) if np.allclose(p, x_star))
            if np.isfinite(c_vals[ix]):
                out["continuous_feasible_xstar"] += 1

            c_locmaxs = count_local_optima(pats2, c_vals, find_max=True)
            out["continuous_n_local_max_per"].append(len(c_locmaxs))
            finite = c_vals[np.isfinite(c_vals)]
            if len(finite) > 0:
                gmax = np.max(finite)
                argmax = [i for i, v in enumerate(c_vals)
                          if np.isfinite(v) and v >= gmax - 1e-6]
                if len(argmax) == 1 and np.allclose(pats2[argmax[0]], x_star):
                    out["continuous_unique_global"] += 1

            for _ in range(n_trials):
                s0 = local.choice([-1.0, 1.0], size=N)
                sf, _, _ = hillclimb(s0, A, b, continuous_fitness, maximize=True)
                out["continuous_hc_total"] += 1
                if np.allclose(sf, x_star):
                    out["continuous_hc_success"] += 1

            if len(c_locmaxs) > 1:
                instance_failed = True

        if instance_failed:
            out["failed_instances"].append({
                "x_star": x_star.tolist(),
                "A": A.tolist(),
                "b": b.tolist(),
                "discrete_local_min_count": len(d_locmins),
                "continuous_local_max_count":
                    len(out["continuous_n_local_max_per"]) > 0
                    and out["continuous_n_local_max_per"][-1] or 0,
            })

        done += 1
    return out


def fmt(r):
    lines = []
    lm = r["discrete_n_local_min_per"]
    n = r["n_instances"]
    disc = (f"  discrete:   unique_global_is_x*: {r['discrete_unique_global']}/{n}, "
            f"  #local-minima per instance: min={min(lm)} mean={np.mean(lm):.2f} max={max(lm)}, "
            f"  hc-success: {r['discrete_hc_success']}/{r['discrete_hc_total']}")
    header = (f"=== N={r['N']}  force_b1={r['force_b1']}  "
              f"(skipped {r['skipped']} unfruitful seeds) ===")
    lines.append(header)
    lines.append(disc)
    if r["continuous_n_local_max_per"]:
        lm2 = r["continuous_n_local_max_per"]
        cont = (f"  continuous: unique_global_is_x*: {r['continuous_unique_global']}/{n}, "
                f"  #local-maxima per instance: min={min(lm2)} mean={np.mean(lm2):.2f} max={max(lm2)}, "
                f"  hc-success: {r['continuous_hc_success']}/{r['continuous_hc_total']},"
                f"  x*-feasible: {r['continuous_feasible_xstar']}/{n}")
        lines.append(cont)
    return "\n".join(lines)


if __name__ == "__main__":
    t0 = time.time()
    print(">>> DISCRETE-RESIDUAL landscape (fast)\n")
    for force_b1 in (False, True):
        tag = "all |b|" if not force_b1 else "|b|<=1 only (overlap regime)"
        print(f"--- regime: {tag} ---")
        for N in (6, 8, 10, 12):
            r = run_for_N(N, n_instances=30, force_b1=force_b1,
                          seed=1000 + (1 if force_b1 else 0) * 100 + N,
                          do_continuous=False)
            print(fmt(r))
        print()
    print(f"[discrete phase elapsed {time.time()-t0:.1f}s]\n")

    t1 = time.time()
    print(">>> CONTINUOUS FITNESS landscape (slower)\n")
    for force_b1 in (False, True):
        tag = "all |b|" if not force_b1 else "|b|<=1 only (overlap regime)"
        print(f"--- regime: {tag} ---")
        for N in (6, 8, 10):
            r = run_for_N(N, n_instances=10, force_b1=force_b1,
                          seed=5000 + (1 if force_b1 else 0) * 100 + N,
                          do_continuous=True, n_trials=10)
            print(fmt(r))
        print()
    print(f"[continuous phase elapsed {time.time()-t1:.1f}s]")
    print(f"[total elapsed {time.time()-t0:.1f}s]")

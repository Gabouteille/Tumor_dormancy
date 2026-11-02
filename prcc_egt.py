#!/usr/bin/env python3
"""PRCC sensitivity analysis of the EGT cancer treatment-cycle ODE (see egt_cancer_dyn.py).

The model is reproduced here without the plotting code of egt_cancer_dyn.py. Same treatment
schedule: no treatment until Tn, then alternating high (lh, Ts) and low (lw, Tw) segments up to Tmax.
Outputs at Tmax: fraction of P cells (x1) and log10 of the total cell number.
"""
import numpy as np
import scipy.integrate
import scipy.stats

rng = np.random.default_rng(12345)

# baseline values from egt_cancer_dyn.py
BASE = dict(mig_spont=0.010, mig_ind=0.0, base_growth=1.0, comp_growth=0.5,
            epsilon=0.0, lh=1.0, lw=0.25, startfrequency_d=0.10)
# sampling ranges (uniform); mig_spont and startfrequency_d are sampled on a log scale
RANGES = dict(mig_spont=(1e-3, 1e-1), base_growth=(0.5, 1.5), comp_growth=(0.25, 0.75),
              epsilon=(0.0, 0.2), lh=(0.5, 1.5), lw=(0.0, 0.5), startfrequency_d=(0.01, 0.5))
LOG_PARAMS = {"mig_spont", "startfrequency_d"}
PARAM_NAMES = list(RANGES)

Tn, segment_interval, total_segments = 40, 15, 12
Ts = Tw = 2 * segment_interval
Tmax = total_segments * segment_interval + Tn


def rhs(y, t, l, p):
    """dx/dt of the model; the third state accumulates the integral of the average fitness."""
    x0, x1, _ = y
    e, ms, mi = p["epsilon"], p["mig_spont"], p["mig_ind"]
    f0 = e
    f1 = (p["base_growth"] - l - e) * x0 + (p["comp_growth"] - l) * x1
    fbar = f0 * x0 + f1 * x1
    if fbar > 0:
        d0 = (f0 - fbar) * x0 - mi * fbar * x0 + ms * (x1 - x0)
        d1 = (f1 - fbar) * x1 + mi * fbar * x0 - ms * (x1 - x0)
    else:
        d0 = (f0 - fbar) * x0 - mi * fbar * x1 + ms * (x1 - x0)
        d1 = (f1 - fbar) * x1 + mi * fbar * x1 - ms * (x1 - x0)
    return [d0, d1, fbar]


def simulate(p):
    """Return (P fraction, log10 total cells) at Tmax; (NaN, NaN) if the integration fails."""
    y = [p["startfrequency_d"], 1.0 - p["startfrequency_d"], 0.0]
    # schedule as (duration, treatment strength): untreated, then H, L, H, L, ... up to Tmax
    sched = [(Tn, 0.0)]
    t_now, k = Tn, 0
    while t_now < Tmax - 1e-9:
        sched.append((Ts, p["lh"]) if k % 2 == 0 else (Tw, p["lw"]))
        t_now += sched[-1][0]
        k += 1
    t0 = 0.0
    try:
        for dur, l in sched:
            t = np.linspace(t0, t0 + dur, 200)
            y = scipy.integrate.odeint(rhs, y, t, args=(l, p), rtol=1e-9, atol=1e-11)[-1]
            t0 += dur
    except Exception:
        return np.nan, np.nan
    frac_p = y[1] / (y[0] + y[1])  # normalise in case of drift off the simplex
    log_cells = y[2] / np.log(10)  # N(Tmax) = N0 * exp(integral of fbar), N0 = 1
    if not (np.isfinite(frac_p) and np.isfinite(log_cells)):
        return np.nan, np.nan
    return frac_p, log_cells


def lhs(n, ranges, names):
    """Latin hypercube sample of n points on the given ranges (log-uniform where specified)."""
    u = np.column_stack([(rng.permutation(n) + rng.random(n)) / n for _ in names])
    X = np.empty_like(u)
    for j, name in enumerate(names):
        lo, hi = ranges[name]
        if name in LOG_PARAMS:
            X[:, j] = np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo)))
        else:
            X[:, j] = lo + u[:, j] * (hi - lo)
    return X


def prcc(X, Y):
    """Partial rank correlation of each column of X with Y, controlling for the other columns."""
    R = np.column_stack([scipy.stats.rankdata(X[:, j]) for j in range(X.shape[1])])
    ry = scipy.stats.rankdata(Y)
    n, k = R.shape
    coef, pvals = np.empty(k), np.empty(k)
    for j in range(k):
        A = np.column_stack([np.ones(n), np.delete(R, j, axis=1)])
        res_x = R[:, j] - A @ np.linalg.lstsq(A, R[:, j], rcond=None)[0]
        res_y = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        r = np.corrcoef(res_x, res_y)[0, 1]
        coef[j] = r
        dof = n - k - 2
        t = r * np.sqrt(dof / max(1e-300, 1 - r * r))
        pvals[j] = 2 * scipy.stats.t.sf(abs(t), dof)
    return coef, pvals


if __name__ == "__main__":
    N = 1000
    X = lhs(N, RANGES, PARAM_NAMES)
    Yp = np.empty(N)
    Yc = np.empty(N)
    for i in range(N):
        p = dict(BASE)
        p.update(dict(zip(PARAM_NAMES, X[i])))
        Yp[i], Yc[i] = simulate(p)

    ok = np.isfinite(Yp) & np.isfinite(Yc)
    print(f"runs: {N}, valid: {ok.sum()}, failed: {N - ok.sum()}")
    print(f"Tmax = {Tmax}\n")
    for label, Y in [("P fraction at Tmax", Yp), ("log10 total cells at Tmax", Yc)]:
        r, p = prcc(X[ok], Y[ok])
        print(f"== PRCC on {label} ==")
        print(f"{'parameter':<20}{'PRCC':>10}{'p-value':>12}")
        for name, ri, pi in sorted(zip(PARAM_NAMES, r, p), key=lambda z: -abs(z[1])):
            star = "*" if pi < 0.05 else ""
            print(f"{name:<20}{ri:>10.3f}{pi:>12.2e} {star}")
        print()

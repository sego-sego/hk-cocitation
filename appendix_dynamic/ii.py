"""Indirect inference for gamma in h(k) = (k/k0)^(-gamma).

The binned Hawkes fit with article fixed effects (hawkes.py) is biased when
the exposure is the article's own past citations (incidental-parameter bias
in a dynamic FE model): weak effects are shrunk more than strong ones, which
steepens the apparent k-dependence.  We therefore treat it as an auxiliary
estimator and calibrate it by simulation:

  1. fit the binned model to the data -> per-bin branching numbers m_b and
     the auxiliary statistics  s = d m / d log k  and  lvl = mean m_b;
  2. simulate citation histories for the same articles from an additive
     Hawkes model with background = fitted background x rho, kernel
     kappa x (fitted lag profile) x (k/k0)^(-gamma), k drawn from each
     article's own citers, Jan-1 date coarsening at the observed rate;
  3. refit the same auxiliary model to the simulated data;
  4. choose (gamma, kappa) so that simulated statistics match the data.

rho is set inside each simulation so that the simulated total number of
citations equals the observed one.

Between-article composition.  Articles differ both in how strongly they
self-excite and in the reference-list lengths of their citers; with a common
kernel this produces a k-slope across articles even when h does not depend
on k.  The auxiliary model therefore gives every article its own excitation
scale s_i (hawkes.py, article_scale=True), so the bin effects are identified
within articles, and the simulator reuses the fitted s_i.  The placebo
(k shuffled among the citations of the same article) must then give ~0.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from hawkes import Hawkes
from hk_core import LAGS, Panel, bin_index, bin_windows

LW = np.array([c - a + 1 for a, c in LAGS])


ARTICLE_SCALE = True


def aux_fit(panel, ev, edges, c0=None):
    """Binned additive Hawkes fit; returns stats, m_b, k_b, fitted object."""
    y = panel.outcome(ev)
    N, _ = bin_windows(panel, ev, edges=edges)
    nb = N.shape[0]
    H = Hawkes(y, N.reshape(nb * len(LAGS), -1), panel.unit, panel.age, panel.cal,
               article_scale=ARTICLE_SCALE)
    c = H.fit(c0=c0, maxit=4000).reshape(nb, len(LAGS))
    m = c @ LW
    b = bin_index(ev["k"].values, edges)
    kb = np.array([np.median(ev["k"].values[b == q]) for q in range(nb)])
    w = np.bincount(b, minlength=nb).astype(float)
    x = np.log(kb)
    X = np.column_stack([np.ones(nb), x - np.average(x, weights=w)])
    coef = np.linalg.lstsq(X * np.sqrt(w)[:, None], m * np.sqrt(w), rcond=None)[0]
    stats = np.array([coef[1], np.average(m, weights=w)])  # slope, level
    return dict(stats=stats, m=m, kb=kb, c=c, H=H, y=y, nev=w)


def permute_k_within_target(ev, rng):
    i = ev["i"].values
    order = np.argsort(i, kind="stable")
    perm = np.lexsort((rng.random(len(i)), i))
    k = ev["k"].values
    knew = np.empty_like(k)
    knew[order] = k[perm]
    e = ev.copy()
    e["k"] = knew
    return e


def aux_diff(panel, ev, edges, rng, nperm=0):
    """Auxiliary fit plus the within-article permutation baseline.
    stats = [D, level] with D = slope - mean(permuted slopes)."""
    A = aux_fit(panel, ev, edges)
    ps = [aux_fit(panel, permute_k_within_target(ev, rng), edges)["stats"][0] for _ in range(nperm)]
    D = A["stats"][0] - (float(np.mean(ps)) if ps else 0.0)
    return dict(stats=np.array([D, A["stats"][1]]), slope=float(A["stats"][0]), perm_slopes=ps,
                m=A["m"], kb=A["kb"], c=A["c"], H=A["H"], y=A["y"], nev=A["nev"])


class Simulator:
    def __init__(self, targets, ns, panel, fit, k0, end_ym, seed=0, log=print):
        self.log = log
        self.panel = panel
        self.n = len(targets)
        self.P = targets["P"].values.astype(int)
        self.k0 = k0
        H = fit["H"]
        eta = sum(H.a[g][H.groups[g]] for g in range(3))
        self.s_art = np.ones(self.n)
        self.s_art[np.unique(panel.unit)] = H.s
        # background by (article, relative month); 0 outside the panel rows
        self.M = panel.M
        self.B = np.zeros((self.n, self.M + 1))
        self.B[panel.unit, panel.age] = np.exp(eta)
        self.end_ym = end_ym
        # lag profile of the kernel (event-weighted over bins)
        cbar = np.average(fit["c"], axis=0, weights=fit["nev"])
        self.profile = cbar / max(cbar @ LW, 1e-12)
        # each article's own citers' k (fallback: all citers)
        ns = ns.sort_values("i")
        self.k_all = ns["k"].values.astype(float)
        cnt = np.bincount(ns["i"].values, minlength=self.n)
        self.k_start = np.r_[0, np.cumsum(cnt)[:-1]]
        self.k_len = cnt
        # initial events (relative months 0 and 1) are taken from the data
        init = ns[ns["m"] <= 1]
        self.init_i, self.init_m, self.init_k = init["i"].values, np.clip(init["m"].values, 0, None), init["k"].values
        # Jan-1 coarsening rate by calendar year (excess over true January share)
        yr = ns["ym"].values // 12
        jan = ns["jan1"].values
        rate = pd.Series(jan).groupby(yr).mean()
        self.p_jan = {int(y): max(0.0, (r - 1 / 12) / (1 - 1 / 12)) for y, r in rate.items()}
        self.n_obs = len(ns[(ns["m"] >= 2)])
        self.rng = np.random.default_rng(seed)

    def draw_k(self, i):
        has = self.k_len[i] > 0
        k = np.empty(len(i))
        u = self.rng.random(len(i))
        k[has] = self.k_all[self.k_start[i[has]] + (u[has] * self.k_len[i[has]]).astype(int)]
        k[~has] = self.k_all[(u[~has] * len(self.k_all)).astype(int)]
        return k

    def run(self, gamma, kappa, rho):
        n, M = self.n, self.M
        c = kappa * self.profile
        Hm = np.zeros((n, M + 1))
        np.add.at(Hm, (self.init_i, self.init_m), (self.init_k / self.k0) ** (-gamma))
        C = np.zeros((n, M + 2))
        ev_i, ev_m, ev_k = [self.init_i], [self.init_m], [self.init_k.astype(float)]
        total = len(self.init_i)
        for t in range(2, M + 1):
            C[:, t] = C[:, t - 1] + Hm[:, t - 1]
            if t == 2:
                C[:, 1] = Hm[:, 0]
                C[:, 2] = Hm[:, 0] + Hm[:, 1]
            X = np.column_stack([C[:, max(t - a + 1, 0)] - C[:, max(t - b, 0)] for a, b in LAGS])
            lam = rho * self.B[:, t] + self.s_art * (X @ c)
            lam_sum = float(lam.sum())
            if not np.isfinite(lam_sum) or total + lam_sum > 4 * self.n_obs:
                raise OverflowError("explosive simulation")
            cnt = self.rng.poisson(lam)
            total += int(cnt.sum())
            if cnt.sum() == 0:
                continue
            i = np.repeat(np.arange(n), cnt)
            k = self.draw_k(i)
            np.add.at(Hm[:, t], i, (k / self.k0) ** (-gamma))
            ev_i.append(i)
            ev_m.append(np.full(len(i), t))
            ev_k.append(k)
        i = np.concatenate(ev_i)
        m = np.concatenate(ev_m)
        k = np.concatenate(ev_k)
        ym = self.P[i] + m
        yr = ym // 12
        pj = np.array([self.p_jan.get(int(y), 0.0) for y in range(yr.min(), yr.max() + 1)])
        jan = self.rng.random(len(i)) < pj[yr - yr.min()]
        ym_rec = np.where(jan, yr * 12, ym)
        return pd.DataFrame(dict(i=i, m=ym_rec - self.P[i], k=k, ym=ym_rec, true_m=m))

    def simulate(self, gamma, kappa, tries=4):
        """Simulate with rho tuned so the number of citations after month 1 matches the data."""
        SB = self.B.sum()
        rho = 0.5
        for attempt in range(tries):
            t0 = time.time()
            ev = self.run(gamma, kappa, rho)
            n_sim = int((ev["true_m"] >= 2).sum())
            self.log(f"      sim gamma={gamma:.2f} kappa={kappa:.3f} rho={rho:.3f}: "
                     f"{n_sim} citations (data {self.n_obs}), {time.time() - t0:.0f} s")
            if abs(n_sim / self.n_obs - 1) < 0.01:
                break
            # n_sim ~ rho*SB + (excited part); excited part is not proportional to rho
            excited = max(n_sim - rho * SB, 0.0)
            ratio_exc = excited / max(n_sim, 1)
            rho = max((self.n_obs * (1 - ratio_exc)) / SB, 1e-3)
        return ev, rho


def match_level(sim, panel, edges, gamma, target_level, kappas=(0.3, 0.6), reps=1, log=print,
                rng=None):
    """Find kappa so that the simulated auxiliary level equals the data level
    (level is close to linear in kappa); returns kappa and the fits at it."""
    pts = []
    for kap in kappas:
        try:
            ev, rho = sim.simulate(gamma, kap)
        except OverflowError:
            log(f"      sim gamma={gamma:.2f} kappa={kap:.3f}: explosive, skipped")
            continue
        t0 = time.time()
        pts.append((kap, aux_fit(panel, ev, edges)["stats"][1]))
        log(f"      aux fit on simulation: level {pts[-1][1]:.4f}, {time.time() - t0:.0f} s")
    if len(pts) < 2:
        kap = min(kappas) / 2
        pts = [(kap / 2, aux_fit(panel, sim.simulate(gamma, kap / 2)[0], edges)["stats"][1]),
               (kap, aux_fit(panel, sim.simulate(gamma, kap)[0], edges)["stats"][1])]
    (k1, l1), (k2, l2) = pts[:2]
    kap = k1 + (target_level - l1) * (k2 - k1) / (l2 - l1)
    kap = float(np.clip(kap, 0.02, 3.0))
    fits = []
    for r in range(reps):
        for shrink in (1.0, 0.9, 0.8, 0.7):
            try:
                ev, rho = sim.simulate(gamma, kap * shrink)
                break
            except OverflowError:
                log(f"      sim gamma={gamma:.2f} kappa={kap * shrink:.3f}: explosive, shrinking")
                continue
        t0 = time.time()
        A = aux_diff(panel, ev, edges, rng if rng is not None else sim.rng)
        log(f"      replicate {r + 1}/{reps}: slope {A['stats'][0]:.4f}, level {A['stats'][1]:.4f}, "
            f"{time.time() - t0:.0f} s")
        fits.append(dict(stats=A["stats"], m=A["m"], rho=rho, kappa=kap * shrink))
    log(f"  gamma={gamma:.2f}: kappa={kap:.3f}  slope={np.mean([f['stats'][0] for f in fits]):.4f} "
        f"level={np.mean([f['stats'][1] for f in fits]):.4f}")
    return kap, fits


def calibrate(sim, panel, edges, data_stats, grid=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25), reps=2, log=print):
    """Binding function D(gamma) at matched level (column 'slope' holds D)."""
    rows = []
    for g in grid:
        kap, fits = match_level(sim, panel, edges, g, data_stats[1], reps=reps, log=log)
        sl = np.array([f["stats"][0] for f in fits])
        lv = np.array([f["stats"][1] for f in fits])
        # correct the slope for any residual level mismatch (slope ~ proportional to level)
        sl_adj = sl * data_stats[1] / lv
        rows.append(dict(gamma=g, kappa=kap, slope=float(sl_adj.mean()), slope_sd=float(sl_adj.std(ddof=1)) if reps > 1 else np.nan,
                         level=float(lv.mean()), m=np.mean([f["m"] for f in fits], axis=0).tolist(),
                         rho=float(np.mean([f["rho"] for f in fits]))))
    tab = pd.DataFrame(rows)
    return tab


def invert(tab, data_slope):
    """gamma at which the binding function crosses the data slope (piecewise linear),
    with a delta-method standard error from the replicate spread."""
    g, s = tab["gamma"].values, tab["slope"].values
    order = np.argsort(g)
    g, s = g[order], s[order]
    # slope decreases with gamma; find crossing
    idx = np.where(np.diff(np.sign(s - data_slope)) != 0)[0]
    if len(idx) == 0:
        j = int(np.argmin(np.abs(s - data_slope)))
        return float(g[j]), np.nan, False
    j = idx[0]
    gh = g[j] + (data_slope - s[j]) * (g[j + 1] - g[j]) / (s[j + 1] - s[j])
    deriv = (s[j + 1] - s[j]) / (g[j + 1] - g[j])
    sd = np.nanmean(tab["slope_sd"].values[order][j:j + 2])
    # data slope has the same sampling sd as a simulated one; the binding function is averaged over reps
    reps_eff = 2
    se = float(sd * np.sqrt(1 + 1 / reps_eff) / abs(deriv)) if np.isfinite(sd) else np.nan
    return float(gh), se, True

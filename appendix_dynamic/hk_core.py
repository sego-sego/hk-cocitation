"""Event table, article-month panel and model fits for the h(k) analysis.

Notation (as in the plan): target article i, calendar month index P_i of its
publication, relative month m = ym - P_i.  An event is one citing work j
that cites i, published in relative month m_j, with reference-list length
k_j.  The outcome y_it counts non-self citing works in relative month t;
exposures count earlier events in lag windows, by k-bin or weighted by h(k).
"""
from __future__ import annotations

import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

from cpois import cpois

LAGS = ((1, 6), (7, 12), (13, 24))          # lag windows in months
LEAD = (1, 6)                               # placebo window (future events)
K_EDGES = (1, 16, 26, 36, 51, 76, 121, 10**6)  # bins [1-15], [16-25], ...


def ym_of(date):
    y, m = int(date[:4]), int(date[5:7])
    return y * 12 + m - 1


def ym_str(ym):
    return f"{ym // 12}-{ym % 12 + 1:02d}"


# ----------------------------------------------------------------------------
# raw data -> event table
# ----------------------------------------------------------------------------
def load_targets(data):
    rows = [json.loads(l) for l in gzip.open(os.path.join(data, "targets.jsonl.gz"), "rt")]
    t = pd.DataFrame(rows)
    t = t[t["date"].notna()].reset_index(drop=True)
    t["P"] = t["date"].map(ym_of)
    return t


def load_events(data, targets):
    tidx = {w: i for i, w in enumerate(targets["id"])}
    tauth = [set(a) for a in targets["authors"]]
    rec = []
    for fn in sorted(glob.glob(os.path.join(data, "citing", "*.jsonl.gz"))):
        with gzip.open(fn, "rt") as fh:
            for l in fh:
                r = json.loads(l)
                if not r.get("date") or not r.get("cites"):
                    continue
                ym = ym_of(r["date"])
                k = max(int(r.get("k") or 0), int(r.get("k_list") or 0), 1)
                auth = set(r.get("authors") or [])
                for tid in r["cites"]:
                    i = tidx.get(tid)
                    if i is None:
                        continue
                    rec.append((i, r["id"], ym, r["date"].endswith("-01-01"), k,
                                r.get("type") or "", bool(auth & tauth[i]),
                                r.get("cited_by") or 0, r.get("field") or "",
                                r.get("source") or ""))
    ev = pd.DataFrame(rec, columns=["i", "j", "ym", "jan1", "k", "type", "self",
                                    "cited_by", "field", "source"])
    ev["m"] = ev["ym"] - targets["P"].values[ev["i"].values]
    ev["year"] = ev["ym"] // 12
    # k relative to the median list length of the citing work's field and year
    med = ev.drop_duplicates("j").groupby(["field", "year"])["k"].median()
    ev["k_rel"] = ev["k"] / med.reindex(pd.MultiIndex.from_arrays([ev["field"], ev["year"]])).values
    return ev


# ----------------------------------------------------------------------------
# panel
# ----------------------------------------------------------------------------
class Panel:
    """Article-month panel for relative months t = t_min..M (calendar <= end)."""

    def __init__(self, targets, M=120, end_ym=None, t_min=2):
        self.n = len(targets)
        self.M = M
        self.P = targets["P"].values.astype(int)
        self.W = M + LEAD[1] + 2  # relative months 0..M+LEAD+1 stored
        t = np.arange(t_min, M + 1)
        unit = np.repeat(np.arange(self.n), len(t))
        age = np.tile(t, self.n)
        cal = self.P[unit] + age
        ok = np.ones(len(unit), bool) if end_ym is None else cal <= end_ym
        self.unit, self.age, self.cal = unit[ok], age[ok], cal[ok]

    def grid(self, ev, weights=None):
        """Per-target counts by relative month (events before publication -> m=0)."""
        m = np.clip(ev["m"].values, 0, None)
        keep = m < self.W
        w = np.ones(len(ev)) if weights is None else np.asarray(weights, float)
        G = np.zeros((self.n, self.W))
        np.add.at(G, (ev["i"].values[keep], m[keep]), w[keep])
        return G

    def window(self, G, a, b, lead=False):
        """Sum of G over relative months t-b..t-a (or t+a..t+b if lead)."""
        C = np.concatenate([np.zeros((self.n, 1)), np.cumsum(G, axis=1)], axis=1)
        t = self.age
        if lead:
            hi, lo = np.minimum(t + b + 1, self.W), np.minimum(t + a, self.W)
        else:
            hi, lo = np.clip(t - a + 1, 0, None), np.clip(t - b, 0, None)
        return C[self.unit, hi] - C[self.unit, lo]

    def outcome(self, ev):
        G = self.grid(ev)
        return G[self.unit, self.age]


def bin_index(k, edges=K_EDGES):
    return np.searchsorted(np.asarray(edges), k, side="right") - 1


def bin_design(panel, ev_x, kcol="k", edges=K_EDGES, weights=None, lead=False):
    b = bin_index(ev_x[kcol].values, edges)
    cols, names = [], []
    w = np.ones(len(ev_x)) if weights is None else np.asarray(weights, float)
    for bb in range(len(edges) - 1):
        sel = b == bb
        G = panel.grid(ev_x[sel], w[sel])
        for (a, c) in LAGS:
            cols.append(panel.window(G, a, c))
            names.append((bb, f"{a}-{c}"))
        if lead:
            cols.append(panel.window(G, LEAD[0], LEAD[1], lead=True))
            names.append((bb, "lead"))
    return np.column_stack(cols), names


def fit(panel, y, X):
    return cpois(y, X, panel.unit, [panel.age, panel.cal])


# ----------------------------------------------------------------------------
# nonlinear-index Poisson (Gauss-Newton on top of the FE Poisson estimator)
# ----------------------------------------------------------------------------
def prof_ll(panel, y, index, warm=None):
    """Poisson log-likelihood with the index fixed and all FE profiled out."""
    r = cpois(y, np.zeros((len(y), 1)), panel.unit, [panel.age, panel.cal], offset=index,
              theta0=warm)
    return r["loglik"], r["theta"]


def numjac(fn, theta, base, eps=1e-5):
    J = np.empty((len(base), len(theta)))
    for j in range(len(theta)):
        d = eps * max(1.0, abs(theta[j]))
        tp = theta.copy()
        tp[j] += d
        J[:, j] = (fn(tp) - base) / d
    return J


def nlfit(panel, y, fn, theta0, maxit=60, tol=1e-6):
    """Maximize the Poisson likelihood of log E[y] = FE + fn(theta).

    Gauss-Newton: at each step the index is linearized, the linear FE Poisson
    model is fitted with the remainder as offset, and the step is halved until
    the profiled likelihood increases.  The covariance is the article-clustered
    sandwich of the linearized model at the optimum.
    """
    theta = np.asarray(theta0, float)
    f = fn(theta)
    ll, warm = prof_ll(panel, y, f)
    for it in range(maxit):
        J = numjac(fn, theta, f)
        r = cpois(y, J, panel.unit, [panel.age, panel.cal], offset=f - J @ theta)
        step = r["beta"] - theta
        t = 1.0
        for _ in range(25):
            cand = theta + t * step
            fc = fn(cand)
            if np.all(np.isfinite(fc)):
                llc, warmc = prof_ll(panel, y, fc, warm)
                if llc >= ll - 1e-9 * abs(ll):
                    break
            t /= 2
        dmax = np.max(np.abs(t * step))
        theta, f, warm = cand, fc, warmc
        improve = llc - ll
        ll = llc
        if dmax < tol or improve < 1e-7:
            break
    J = numjac(fn, theta, f)
    r = cpois(y, J, panel.unit, [panel.age, panel.cal], offset=f - J @ theta)
    return dict(theta=theta, se=r["se"], vcov=r["vcov"], loglik=ll, iters=it + 1,
                dispersion=r["dispersion"], n=r["n"], lin_check=float(np.max(np.abs(r["beta"] - theta))))


def transform(X, link):
    return X if link == "lin" else np.log1p(np.maximum(X, -0.9))


# ----------------------------------------------------------------------------
# step 1: relative weight of a citation by k-bin (single index)
# ----------------------------------------------------------------------------
def bin_windows(panel, ev_x, kcol="k", edges=K_EDGES, weights=None, lead=False):
    b = bin_index(ev_x[kcol].values, edges)
    w = np.ones(len(ev_x)) if weights is None else np.asarray(weights, float)
    nb = len(edges) - 1
    N = np.zeros((nb, len(LAGS), len(panel.unit)))
    Lw = np.zeros((nb, len(panel.unit))) if lead else None
    for bb in range(nb):
        sel = b == bb
        G = panel.grid(ev_x[sel], w[sel])
        for l, (a, c) in enumerate(LAGS):
            N[bb, l] = panel.window(G, a, c)
        if lead:
            Lw[bb] = panel.window(G, LEAD[0], LEAD[1], lead=True)
    return N, Lw


def bin_index_model(N, ref, link, Lw=None):
    """theta = [beta_1..beta_L, w_b (b != ref), (lead coefficients)]."""
    nb, nl, _ = N.shape
    others = [b for b in range(nb) if b != ref]

    def fn(theta):
        beta = theta[:nl]
        w = np.ones(nb)
        w[others] = theta[nl:nl + nb - 1]
        X = np.tensordot(w, N, axes=(0, 0))  # (L, rows)
        out = beta @ transform(X, link)
        if Lw is not None:
            out = out + theta[nl + nb - 1:] @ Lw
        return out

    theta0 = np.r_[np.full(nl, 0.01), np.ones(nb - 1)]
    if Lw is not None:
        theta0 = np.r_[theta0, np.zeros(nb)]
    return fn, theta0, others


def gls_ratio(w, Vw, kb, kref):
    """GLS fit of w_b = (k_b/k_ref)^(-gamma) over the non-reference bins."""
    x = np.log(kb / kref)
    Vi = np.linalg.pinv(Vw)
    g = 0.5
    for _ in range(200):
        f = np.exp(-g * x)
        J = -x * f
        step = (J @ Vi @ (w - f)) / (J @ Vi @ J)
        g += step
        if abs(step) < 1e-12:
            break
    f = np.exp(-g * x)
    J = -x * f
    se = float(np.sqrt(1 / (J @ Vi @ J)))
    chi2 = float((w - f) @ Vi @ (w - f))
    return g, se, chi2


# ----------------------------------------------------------------------------
# step 2: parametric h(k)
# ----------------------------------------------------------------------------
def h_power(k, gamma, k0):
    return (k / k0) ** (-gamma)


def h_club(k, log_alpha, k0):
    a = np.exp(log_alpha)
    f = lambda kk: -np.expm1(-a * kk) / kk
    return f(k) / f(k0)


def param_model(panel, ev_x, hfun, k0, link, fixed=None):
    """theta = [beta_1..beta_L, p] (p omitted if fixed is given)."""
    k = ev_x["k"].values
    nl = len(LAGS)

    def windows(p):
        G = panel.grid(ev_x, hfun(k, p, k0))
        return np.array([panel.window(G, a, c) for a, c in LAGS])

    if fixed is not None:
        X = transform(windows(fixed), link)
        return (lambda th: th[:nl] @ X), np.full(nl, 0.01)

    cache = {}

    def fn(theta):
        p = float(theta[nl])
        if p not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[p] = transform(windows(p), link)
        return theta[:nl] @ cache[p]

    return fn, None

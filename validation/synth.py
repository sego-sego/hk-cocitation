"""Synthetic raw data in the OpenAlex format of fetch_openalex.py, with a
known h(k) = (k/k0)^(-gamma), to test the whole pipeline end to end.

Each target i has a level a_i, a common age profile, calendar shocks and a
latent AR(1) popularity u_it (shared by all citing works, i.e. a confounder
of the level of every bin but not of the k-slope).  Monthly citations are
Poisson with rate (an additive Hawkes process, stable for branching < 1)
    exp(a_i + g(age) + s(cal) + u_it) + sum_l c_l * X^l_it,
X^l_it = sum of h(k_j) over earlier non-self citing works j in lag window l.
The estimating model is log-linear in X, so this also tests robustness of the
k-slope to misspecification of the link.
Citing works get k from a lognormal; reviews (8%) have longer lists, and
10% of the citing works are self-citations.  10% of the dates are coarsened
to January 1 of the year, as in OpenAlex.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

import numpy as np

LAGS = ((1, 6), (7, 12), (13, 24))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--hform", default="power", choices=["power", "club"])
    ap.add_argument("--alpha", type=float, default=0.05, help="club parameter")
    ap.add_argument("--beta", default="0.05,0.025,0.0125", help="Hawkes kernel c_l per month")
    ap.add_argument("--level", type=float, default=0.03)
    ap.add_argument("--months", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jan", type=float, default=0.10)
    ap.add_argument("--ar", type=float, default=0.1, help="sd of latent popularity shocks")
    ap.add_argument("--review", type=float, default=0.08)
    ap.add_argument("--het", type=float, default=0.0,
                    help="between-article heterogeneity: citers' k median x exp(0.35 z), "
                         "kernel x exp(-het z) (spurious k-slope across articles)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    beta = np.array([float(x) for x in a.beta.split(",")])
    k0 = 35.0
    if a.hform == "power":
        h = lambda k: (k / k0) ** (-a.gamma)
    else:
        f = lambda k: -np.expm1(-a.alpha * k) / k
        h = lambda k: f(k) / f(k0)
    os.makedirs(os.path.join(a.out, "citing"), exist_ok=True)
    n, T = a.n, a.months
    P = 2010 * 12 + rng.integers(0, 12, n)
    lvl = rng.normal(np.log(a.level), 0.9, n)
    age = np.arange(T)
    g = 1.2 * np.log1p(age) - age / 30.0
    s = 0.15 * np.sin(2 * np.pi * np.arange(400) / 12)
    with gzip.open(os.path.join(a.out, "targets.jsonl.gz"), "wt") as fh:
        for i in range(n):
            fh.write(json.dumps(dict(id=f"T{i}", doi=None, date=f"{P[i] // 12}-{P[i] % 12 + 1:02d}-15",
                                     authors=[f"A{i}"], cited_by=None, field="F1")) + "\n")
    z = rng.normal(size=n)
    kmed = 35 * np.exp(0.35 * z) if a.het else np.full(n, 35.0)
    kmult = np.minimum(np.exp(-a.het * z - a.het ** 2 / 2), 1.7)  # keep branching < 1
    H = np.zeros((n, T))
    C = np.zeros((n, T + 1))
    u = np.zeros(n)
    recs = []
    for t in range(1, T):
        u = 0.95 * u + rng.normal(0, a.ar, n)
        C[:, t] = C[:, t - 1] + H[:, t - 1]
        X = np.column_stack([C[:, max(t - b + 1, 0)] - C[:, max(t - c, 0)] for b, c in LAGS])
        lam = np.exp(lvl + g[t] + s[(P + t) % 400] + u) + kmult * (X @ beta)
        cnt = rng.poisson(lam)
        idx = np.repeat(np.arange(n), cnt)
        m = len(idx)
        if m == 0:
            continue
        review = rng.random(m) < a.review
        k = np.clip(np.round(np.exp(rng.normal(np.log(kmed[idx]), 0.6)) * np.where(review, 3, 1)), 1, 600)
        self_ = rng.random(m) < 0.10
        np.add.at(H[:, t], idx[~self_], h(k[~self_]))
        ym = P[idx] + t
        jan = rng.random(m) < a.jan
        day = rng.integers(1, 29, m)
        for q in range(m):
            yr, mo = ym[q] // 12, ym[q] % 12 + 1
            date = f"{yr}-01-01" if jan[q] else f"{yr}-{mo:02d}-{day[q]:02d}"
            recs.append(dict(id=f"W{len(recs)}", doi=None, date=date,
                             type="review" if review[q] else "article", k=int(k[q]), k_list=int(k[q]),
                             cites=[f"T{idx[q]}"], authors=[f"A{idx[q]}"] if self_[q] else [f"B{len(recs)}"],
                             source="S1", source_type="journal", field="F1",
                             cited_by=int(rng.poisson(20))))
    with gzip.open(os.path.join(a.out, "citing", "batch_00000.jsonl.gz"), "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"{n} targets, {len(recs)} citing events, form={a.hform}, gamma={a.gamma}, "
          f"alpha={a.alpha}, beta={beta}")


if __name__ == "__main__":
    main()

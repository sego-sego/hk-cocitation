"""Analysis of the co-citation sample (fetch_cocite.py).

Model (Poisson with target fixed effects, conditional likelihood in cpois.py):
    E[Y_ij] = exp(alpha_i + f(k_j) + delta log n_j + year_j + field_j + type_j)
Y_ij: works citing both j and target i; n_j: works citing j.
f(k) is estimated by k-bins (relative to the bin 26-35) and as -gamma log k.
gamma = 0: public (j passes i on in full whatever its list length);
gamma = 1: fractional (j's audience is split among its k references).

Usage: python cocite_analyze.py --data data --out results_cocite
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from cpois import cpois

K_EDGES = (1, 16, 26, 36, 51, 76, 121, 10 ** 6)
LABELS = ["1-15", "16-25", "26-35", "36-50", "51-75", "76-120", ">120"]
REF = 2
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def shape_test(d, small=("year", "field", "type")):
    """Full-likelihood comparison of the power law h ~ k^-gamma and the club form
    h = (1 - exp(-alpha k))/k (alpha profiled on a grid; both have one shape parameter)."""
    y = d["y"].values
    unit = d["i"].values
    sm = [d[s_].values for s_ in small]
    lk, ln = np.log(d["k"].values), np.log(d["n_j"].values)
    rp = cpois(y, np.column_stack([lk, ln]), unit, sm)
    best = None
    for a in np.geomspace(1e-3, 1.0, 61):
        off = np.log(-np.expm1(-a * d["k"].values) / d["k"].values)
        rc = cpois(y, ln[:, None], unit, sm, offset=off)
        if best is None or rc["loglik"] > best[1]:
            best = (a, rc["loglik"])
    phi = rp["dispersion"]
    return dict(ll_power=rp["loglik"], gamma=-rp["beta"][0], ll_club=best[1], club_alpha=best[0],
                club_kc=1 / best[0], dispersion=phi, qlr_club_minus_power=2 * (best[1] - rp["loglik"]) / phi)


def fit(d, form, small=("year", "field", "type")):
    nb = len(K_EDGES) - 1
    b = np.searchsorted(K_EDGES, d["k"].values, side="right") - 1
    cols, names = [], []
    if form == "bins":
        for q in range(nb):
            if q != REF:
                cols.append((b == q).astype(float))
                names.append(LABELS[q])
    else:
        cols.append(np.log(d["k"].values))
        names.append("log k")
    cols.append(np.log(d["n_j"].values))
    names.append("log n_j")
    X = np.column_stack(cols)
    r = cpois(d["y"].values, X, d["i"].values, [d[s].values for s in small])
    return dict(names=names, beta=r["beta"].tolist(), se=r["se"].tolist(), n=r["n"],
                nunits=r["nunits"], dispersion=r["dispersion"], converged=bool(r["converged"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_cocite")
    ap.add_argument("--placebo", nargs="?", const="cocite_placebo_nb.csv", default=None,
                    help="analyze a placebo file (default cocite_placebo_nb.csv)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = pd.read_csv(os.path.join(a.data, "cocite.csv"), dtype={"field": str, "type": str, "source": str})
    if a.placebo:
        p = pd.read_csv(os.path.join(a.data, a.placebo), dtype={"field": str, "type": str, "source": str})
        # same pairs: real co-citation y and placebo y_placebo (target FE on i' for the placebo)
        real = p.copy()
        plac = p.copy()
        plac["i"] = plac["i_placebo"]
        plac["y"] = plac["y_placebo"]
        print(f"{a.placebo}: {len(p)} pairs")
        # within-pair comparison: real vs placebo count of the same citing work j,
        # E[Y] = exp(alpha_pair + b*real - g_diff*real*ln k): n_j, year, field, type
        # and the citing work's audience cancel; g_diff is the k-dependence specific
        # to passing on through the reference.
        q = p.drop_duplicates(["i", "j"]).reset_index(drop=True)
        q = q[q["n_j"] > 0].reset_index(drop=True)
        npair = len(q)
        yy = np.r_[np.minimum(q["y"], q["n_j"]), np.minimum(q["y_placebo"], q["n_j"])].astype(float)
        isreal = np.r_[np.ones(npair), np.zeros(npair)]
        lk = np.r_[np.log(q["k"].values), np.log(q["k"].values)]
        pair = np.r_[np.arange(npair), np.arange(npair)]
        clus = np.r_[q["i"].values, q["i"].values]
        rd = cpois(yy, np.column_stack([isreal, isreal * lk]), pair, [], cluster=clus)
        print(f"within-pair difference (real minus placebo): gamma_diff = {-rd['beta'][1]:.3f} ± {rd['se'][1]:.3f} "
              f"(n = {rd['n'] // 2} informative pairs); real/placebo ratio at median k = "
              f"{np.exp(rd['beta'][0] + rd['beta'][1] * np.median(np.log(q['k']))):.2f}")
        for name, dd in (("same pairs, real target", real), ("same pairs, placebo target (not cited by j)", plac)):
            dd = dd.drop_duplicates(["i", "j"]).copy()
            dd["field"] = dd["field"].fillna("")
            dd["type"] = dd["type"].fillna("").map(lambda s: s if s in ("article", "review") else "other")
            dd["year"] = dd["date"].str[:4].astype(int)
            dd = dd[dd["n_j"] > 0].reset_index(drop=True)
            dd["y"] = np.minimum(dd["y"], dd["n_j"])
            if (dd["y"] > 0).sum() < 200:
                print(f"{name}: mean y {dd['y'].mean():.4f}; share y>0 {(dd['y'] > 0).mean():.4f} "
                      f"-- too few non-zero counts to estimate gamma")
                continue
            rl = fit(dd, "log")
            print(f"{name}: gamma = {-rl['beta'][0]:.3f} ± {rl['se'][0]:.3f}; mean y {dd['y'].mean():.3f}; "
                  f"share y>0 {(dd['y'] > 0).mean():.3f}; n = {rl['n']}")
        return
    d = d.drop_duplicates(["i", "j"])
    d["field"] = d["field"].fillna("")
    d["type"] = d["type"].fillna("").map(lambda s: s if s in ("article", "review") else "other")
    d["year"] = d["date"].str[:4].astype(int)
    print(f"{len(d)} pairs, {d['i'].nunique()} targets; n_j>0: {(d['n_j'] > 0).mean():.3f}; "
          f"mean y {d['y'].mean():.3f}; share y>0 {(d['y'] > 0).mean():.3f}")
    d = d[d["n_j"] > 0].reset_index(drop=True)
    d["y"] = np.minimum(d["y"], d["n_j"])

    res = {}
    specs = {
        "main": d,
        "research articles only (no reviews)": d[d["type"] == "article"],
        "citing works up to 2015": d[d["year"] <= 2015],
        "n_j >= 10": d[d["n_j"] >= 10],
    }
    for name, dd in specs.items():
        dd = dd.reset_index(drop=True)
        rb = fit(dd, "bins")
        rl = fit(dd, "log")
        g, gse = -rl["beta"][0], rl["se"][0]
        res[name] = dict(bins=rb, loglin=rl, gamma=g, gamma_se=gse, n_pairs=len(dd))
        print(f"{name}: gamma = {g:.3f} ± {gse:.3f} (n = {rl['n']} informative pairs, "
              f"{rl['nunits']} targets; elasticity on n_j {rl['beta'][1]:.3f}; dispersion {rl['dispersion']:.2f})")

    st = shape_test(d)
    res["shape_test"] = st
    print(f"shape: power gamma {st['gamma']:.3f}; club alpha {st['club_alpha']:.4f} (k_c = {st['club_kc']:.0f}); "
          f"quasi-LR (club - power) = {st['qlr_club_minus_power']:.2f} (positive favours club; |QLR| < ~4 = no clear preference)")

    # per-bin effects relative to the reference bin, with median k per bin
    b = np.searchsorted(K_EDGES, d["k"].values, side="right") - 1
    kb = np.array([np.median(d["k"].values[b == q]) for q in range(len(LABELS))])
    rb = res["main"]["bins"]
    eff = np.zeros(len(LABELS))
    se = np.zeros(len(LABELS))
    j = 0
    for q in range(len(LABELS)):
        if q == REF:
            continue
        eff[q], se[q] = rb["beta"][j], rb["se"][j]
        j += 1
    res["bins_table"] = dict(labels=LABELS, k_median=kb.tolist(), log_effect=eff.tolist(), se=se.tolist(),
                             n=np.bincount(b, minlength=len(LABELS)).tolist())
    json.dump(res, open(os.path.join(a.out, "cocite_numbers.json"), "w"), indent=1, default=float)

    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.edgecolor": INK2, "legend.frameon": False, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    kk = np.geomspace(kb.min() * 0.8, kb.max() * 1.2, 100)
    ax.plot(kk, np.ones_like(kk), ":", color=INK2, lw=1.2, label="public, h = const")
    ax.plot(kk, kb[REF] / kk, "--", color=INK2, lw=1.2, label="fractional, h ∝ 1/k")
    g = res["main"]["gamma"]
    ax.plot(kk, (kk / kb[REF]) ** (-g), color=BLUE, lw=1.6,
            label=f"fit, h ∝ k^(−γ), γ = {g:.2f} ± {res['main']['gamma_se']:.2f}")
    lo, hi = np.exp(eff - 1.96 * se), np.exp(eff + 1.96 * se)
    ax.errorbar(kb, np.exp(eff), yerr=[np.exp(eff) - lo, hi - np.exp(eff)], fmt="o", ms=4.5, color=INK,
                ecolor=INK2, elinewidth=1, capsize=0, zorder=5, label="k-bins (95% CI)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10, 20, 50, 100, 200])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("reference-list length k of the citing work j")
    ax.set_ylabel("co-citation rate of i via j\n(relative to k = 26–35)")
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"fig_cocite.{ext}"))
    print("saved", os.path.join(a.out, "fig_cocite.png"))


if __name__ == "__main__":
    main()

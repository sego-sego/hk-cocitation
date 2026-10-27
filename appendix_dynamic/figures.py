"""Figure for the h(k) analysis from numbers.json (indirect-inference version).

Usage: python figures.py --res results
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

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.6, "savefig.dpi": 300, "figure.facecolor": "white"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    a = ap.parse_args()
    N = json.load(open(os.path.join(a.res, "numbers.json")))
    aux, ii = N["aux"], N["ii"]
    truth = N.get("truth")
    kb = np.array(aux["kb"])
    tab = ii["table"]
    g_tab = np.array(tab["gamma"])

    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw=dict(width_ratios=[1.1, 1, 1.15]))

    # (a) data versus simulations at matched level
    A = ax[0]
    for g, col, ls in ((0.0, INK2, ":"), (1.0, INK2, "--")):
        j = int(np.argmin(np.abs(g_tab - g)))
        A.plot(kb, tab["m"][j], color=col, ls=ls, lw=1.2,
               label=("public, γ = 0" if g == 0 else "fractional, γ = 1"))
    ms, sd = np.array(ii["m_sim_mean"]), np.array(ii["m_sim_sd"])
    A.fill_between(kb, ms - 1.96 * sd, ms + 1.96 * sd, color=BLUE, alpha=0.18, lw=0)
    A.plot(kb, ms, color=BLUE, label=f"fitted, γ = {ii['gamma']:.2f}")
    A.plot(kb, aux["m"], "o", ms=4.5, color=INK, zorder=5, label="data")
    A.set_xscale("log")
    A.set_xticks([10, 20, 50, 100, 200])
    A.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    A.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    A.set_xlabel("reference-list length k")
    A.set_ylabel("extra citations per citation, $\\hat m_b$")
    A.set_title("(a) auxiliary estimate", loc="left", fontsize=9)
    A.legend(fontsize=6, loc="upper right", handlelength=1.8)

    # (b) binding function
    B = ax[1]
    s = np.array(tab["slope"])
    ssd = np.array(tab["slope_sd"])
    B.errorbar(g_tab, s, yerr=1.96 * ssd, fmt="-o", ms=3.5, color=BLUE, ecolor=BLUE, elinewidth=1,
               capsize=0, label="simulated")
    B.axhline(aux.get("D", aux["slope"]), color=INK, lw=1, label="data")
    B.axvline(ii["gamma"], color=BLUE, lw=0.8, ls=":")
    if ii["se"] == ii["se"]:
        B.axvspan(ii["gamma"] - 1.96 * ii["se"], ii["gamma"] + 1.96 * ii["se"], color=BLUE, alpha=0.12, lw=0)
    if truth is not None:
        B.axvline(truth, color=ORANGE, lw=1.2, label=f"true γ = {truth}")
    B.set_xlabel("γ in h(k) = (k/k₀)^(−γ)")
    B.set_ylabel("d m̂ / d log k")
    B.set_title("(b) indirect inference", loc="left", fontsize=9)
    B.legend(fontsize=6.5)

    # (c) robustness
    C = ax[2]
    rows = [("main", ii["gamma"], ii["se"])]
    rows += [(k, v["gamma"], np.nan) for k, v in N["variants"].items()]
    rows += [(f"k shuffled ({i + 1})", p["gamma"], np.nan) for i, p in enumerate(N["permutation"])]
    yv = np.arange(len(rows))[::-1]
    for yy, (lab, g, se) in zip(yv, rows):
        col = BLUE if lab == "main" else (INK2 if "shuffled" in lab else INK)
        C.errorbar(g, yy, xerr=None if se != se else 1.96 * se, fmt="o", ms=4, color=col, ecolor=col,
                   elinewidth=1, capsize=0)
    if truth is not None:
        C.axvline(truth, color=ORANGE, lw=1.2)
    for x in (0, 1):
        C.axvline(x, color=INK2, lw=0.6, ls=":")
    C.set_yticks(yv)
    C.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    C.set_xlabel("γ")
    C.set_title("(c) variants and placebo", loc="left", fontsize=9)
    C.grid(axis="y", visible=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.res, f"fig_hk.{ext}"))
    print("saved", os.path.join(a.res, "fig_hk.png"))


if __name__ == "__main__":
    main()

"""Fig. 2: transfer per reference versus reference-list length.

Usage (from the repository root):
    python figures/make_fig2.py results_cocite/cocite_numbers.json
Writes figures/fig2_hk.{pdf,png} and figures/shape_fit.json (bin-level power and
club fits shown in the figure; the full-likelihood comparison is printed by
src/cocite_analyze.py).
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from scipy.optimize import least_squares

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.edgecolor": INK2, "legend.frameon": False, "savefig.dpi": 300})
N = json.load(open(sys.argv[1]))
B = N["bins_table"]
k, y, s = map(np.array, (B["k_median"], B["log_effect"], B["se"]))
ref = 2
m = np.arange(len(k)) != ref

def fit(f, p0):
    r = least_squares(lambda p: (f(p, k[m]) - f(p, k[[ref]])[0] - y[m]) / s[m], p0)
    return r.x, 2 * r.cost
power = lambda p, kk: -p[0] * np.log(kk)
club = lambda p, kk: np.log(-np.expm1(-np.exp(p[0]) * kk) / kk)
pp, chi_p = fit(power, [0.5])
pc, chi_c = fit(club, [np.log(0.03)])
alpha = np.exp(pc[0])
out = dict(power_gamma_bins=float(pp[0]), power_chi2=float(chi_p), club_alpha=float(alpha),
           club_kc=float(1 / alpha), club_chi2=float(chi_c), dof=int(m.sum() - 1))
json.dump(out, open("figures/shape_fit.json", "w"), indent=1)
print(out)

g = N["main"]["gamma"]; gse = N["main"]["gamma_se"]
fig, ax = plt.subplots(figsize=(3.4, 2.7))
kk = np.geomspace(9, 230, 200)
norm = lambda f, p: np.exp(f(p, kk) - f(p, k[[ref]])[0])
ax.plot(kk, np.ones_like(kk), ":", color=INK2, lw=1.1, label=r"public, $\gamma=0$")
ax.plot(kk, k[ref] / kk, "--", color=INK2, lw=1.1, label=r"fractional, $\gamma=1$")
ax.plot(kk, (kk / k[ref]) ** (-g), color=BLUE, lw=1.6,
        label=rf"power law, $\gamma={g:.2f}\pm{gse:.2f}$")
ax.plot(kk, norm(club, pc), color=ORANGE, lw=1.4, label=rf"club, $k_c={1/alpha:.0f}$")
e = np.exp(y)
ax.errorbar(k, e, yerr=[e - np.exp(y - 1.96 * s), np.exp(y + 1.96 * s) - e], fmt="o", ms=4,
            color=INK, ecolor=INK2, elinewidth=1, capsize=0, zorder=5, label="$k$ bins (95% CI)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([10, 20, 50, 100, 200]); ax.xaxis.set_major_formatter(mt.ScalarFormatter())
ax.xaxis.set_minor_formatter(mt.NullFormatter())
ax.set_yticks([0.2, 0.5, 1, 2]); ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel(r"reference-list length $k$ of the citing work")
ax.set_ylabel(r"$h(k)/h(k_{\rm ref})$")
ax.legend(fontsize=6.3, loc="lower left")
fig.tight_layout(); fig.savefig("figures/fig2_hk.pdf"); fig.savefig("figures/fig2_hk.eps"); fig.savefig("figures/fig2_hk.png")


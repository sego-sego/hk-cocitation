"""Fig. 1, two-source version: (a) the same article i cited from a short list
(k=3) and a long list (k=12); a larger share of j1's audience passes on to i.
(b) transfer per recipient h(k) in the regimes of the theory."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

BLUE, ORANGE, INK, INK2, GRID, MUTE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "savefig.dpi": 300,
                     "mathtext.fontset": "cm", "ps.fonttype": 42, "pdf.fonttype": 42,
                     # a TrueType font: OpenType/CFF fonts (e.g. Inter) embedded as Type 42 break
                     # EPS/PDF import in Illustrator
                     "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]})

fig = plt.figure(figsize=(7.0, 3.25))
A = fig.add_axes([0.0, 0.01, 0.6, 0.98])
B = fig.add_axes([0.70, 0.16, 0.285, 0.7])
A.set_xlim(-0.2, 11.0); A.set_ylim(-1.45, 7.2); A.set_aspect("equal"); A.axis("off")


def arrow(p, q, color=MUTE, lw=0.8, rad=0.0, sA=0, sB=4, z=1, ms=6):
    A.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=sA, shrinkB=sB, zorder=z))


def box(x, y, w, h, fc, ec, lw=1.0, z=3, r=0.06):
    A.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0.01,rounding_size={r}",
                               fc=fc, ec=ec, lw=lw, zorder=z))


# target i, shared by both sources
ix, iy = 1.2, 3.3
A.add_patch(Circle((ix, iy), 0.24, fc=BLUE, ec=BLUE, zorder=4))
A.text(ix - 0.38, iy, r"$i$", ha="right", va="center", color=BLUE, fontsize=11)

# sources
j1 = (4.6, 5.25); j2 = (4.6, 1.35)
for (x, y), lab in ((j1, r"$j_1$"), (j2, r"$j_2$")):
    box(x, y, 0.8, 0.56, "white", INK, lw=1.2)
    A.text(x, y, lab, ha="center", va="center", fontsize=10.5, zorder=4)
A.text(j1[0], j1[1] + 0.45, r"$k=3$", ha="center", va="bottom", fontsize=8, color=INK2)
A.text(j2[0], j2[1] - 0.42, r"$k=12$", ha="center", va="top", fontsize=8, color=INK2)

# other references of j1 (2) and j2 (11)
r1 = [(1.2, 5.9), (1.2, 5.0)]
r2 = [(1.2, -0.1 + 0.245 * n) for n in range(11)]
for (x, y) in r1:
    A.add_patch(Circle((x, y), 0.17, fc="white", ec=INK2, lw=0.9, zorder=3))
    A.plot([j1[0] - 0.4, x + 0.17], [j1[1], y], color=MUTE, lw=0.8, zorder=1)
for (x, y) in r2:
    A.add_patch(Circle((x, y), 0.1, fc="white", ec=INK2, lw=0.8, zorder=3))
    A.plot([j2[0] - 0.4, x + 0.1], [j2[1], y], color=MUTE, lw=0.6, zorder=1)
arrow((j1[0] - 0.4, j1[1]), (ix, iy), color=BLUE, lw=1.6, sB=8)
arrow((j2[0] - 0.4, j2[1]), (ix, iy), color=BLUE, lw=1.6, sB=8)

# audiences: 8 citers each; 4 of j1's and 2 of j2's also cite i
cx = 8.4
c1 = np.linspace(4.15, 6.45, 8)
c2 = np.linspace(0.25, 2.55, 8)
co1 = {0, 2, 5, 7}
co2 = {1, 6}
for ys, src, co, outward in ((c1, j1, co1, +1), (c2, j2, co2, -1)):
    for n_, y in enumerate(ys):
        hit = n_ in co
        box(cx, y, 0.34, 0.26, ORANGE if hit else "white", ORANGE if hit else INK2, lw=0.9, r=0.03)
        arrow((cx - 0.15, y), (src[0] + 0.4, src[1]), color=INK2, lw=0.6, sB=1, ms=4.5)
for ys, co in ((c1, co1), (c2, co2)):
    for n_ in co:
        A.text(cx, ys[n_], r"$i$", ha="center", va="center", fontsize=7, color="white", zorder=5)
# shares
A.text(cx + 0.45, c1.mean(), r"$Y/n=4/8$", ha="left", va="center", fontsize=8.5, color=ORANGE)
A.text(cx + 0.45, c2.mean(), r"$Y/n=2/8$", ha="left", va="center", fontsize=8.5, color=ORANGE)
A.text(cx + 0.45, c2.mean() - 0.42, "fractional: 1/8\npublic: 4/8", ha="left", va="top", fontsize=6.6,
       color=INK2, linespacing=1.3)
A.text(cx, 6.62, r"works citing $j$", ha="center", va="bottom", fontsize=7.2, color=INK2)
A.text(1.2, 6.55, "references", ha="center", va="bottom", fontsize=7.2, color=INK2)
# legend for the orange marker (top centre)
box(3.35, 6.95, 0.34, 0.26, ORANGE, ORANGE, lw=0.9, r=0.03)
A.text(3.35, 6.95, r"$i$", ha="center", va="center", fontsize=7, color="white", zorder=5)
A.text(3.65, 6.95, r"also cites $i$ (co-citation $Y_{ij}$)", ha="left", va="center", fontsize=7.2, color=INK2)
# time axis
ty = -0.85
A.annotate("", xy=(10.3, ty), xytext=(0.3, ty),
           arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=8))
for x, lab in ((1.2, r"$t_i$ (2010)"), (4.6, r"$t_j\leq2018$"), (cx, r"$t>t_j$")):
    A.plot([x, x], [ty - 0.09, ty + 0.09], color=INK2, lw=0.9)
    A.text(x, ty - 0.18, lab, ha="center", va="top", fontsize=7.2, color=INK2)
A.text(10.3, ty + 0.15, "time", ha="right", va="bottom", fontsize=7.2, color=INK2)
A.text(-0.15, 7.2, "(a)", fontsize=9, va="top", fontweight="bold")

# ---------------- (b) regimes ----------------
k = np.geomspace(1, 300, 300)
alpha = 1 / 50
B.plot(k, np.ones_like(k), ":", color=INK2, lw=1.2)
B.plot(k, 1 / k, "--", color=INK2, lw=1.2)
B.plot(k, -np.expm1(-alpha * k) / (alpha * k), color=ORANGE, lw=1.5)
B.plot(k, k ** -0.5, color=BLUE, lw=1.6)
B.axvline(1 / alpha, color=ORANGE, lw=0.7, ls=(0, (2, 2)))
B.text(1 / alpha * 1.1, 1.6e-3 * 1.4, r"$k_c=\alpha^{-1}$", color=ORANGE, fontsize=7)
B.text(1.25, 1.18, r"public, $\gamma=0$", fontsize=7, color=INK2)
B.text(1.3, 0.017, r"fractional, $\gamma=1$", fontsize=7, color=INK2, rotation=-33)
B.text(150, 0.42, "club", fontsize=7, color=ORANGE)
B.text(4, 0.62, r"$\gamma=\frac{1}{2}$", fontsize=7.5, color=BLUE)
B.set_xscale("log"); B.set_yscale("log")
B.set_xlim(1, 300); B.set_ylim(1.6e-3, 2.2)
B.set_xticks([1, 10, 100]); B.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
B.set_yticks([1e-2, 1e-1, 1]); B.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
B.xaxis.set_minor_formatter(mt.NullFormatter()); B.yaxis.set_minor_formatter(mt.NullFormatter())
B.spines[["top", "right"]].set_visible(False)
B.grid(True, color=GRID, lw=0.6)
B.set_xlabel(r"fan-out $k$")
B.set_ylabel(r"transfer per recipient $h(k)/h(1)$")
B.text(-0.36, 1.12, "(b)", transform=B.transAxes, fontsize=9, fontweight="bold")

fig.savefig("figures/fig1_scheme.pdf")          # TrueType fonts (pdf.fonttype 42): editable in Illustrator
fig.savefig("figures/fig1_scheme.png")
fig.savefig("figures/fig1_scheme.eps")          # TrueType (Type 42) fonts
with plt.rc_context({"svg.fonttype": "none"}):    # SVG with live, editable text
    fig.savefig("figures/fig1_scheme.svg")
print("saved")

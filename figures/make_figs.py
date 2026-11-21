"""Figures of the GRIT paper: (1) gates and rules; (2) oracle gates and the oracle risk R*(lambda)/r_B;
(3) frequentist risk curves of Bayes-tuned rules."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "grit"))

import json, numpy as np
OUT = pathlib.Path(__file__).resolve().parent / "Figs"; OUT.mkdir(exist_ok=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm
from gates import *
from bayes import prior_moments, oracle_gate_risk, yp

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False})
ink2 = "#52514e"; grid_c = "#e4e3df"

# ---------------- Figure 1
rules = [("soft", G_soft, "-"), ("garrote ($b=2$)", G_hybrid(2.0), "--"), ("firm/MCP ($a=2.5$)", G_firm(2.5), "-."),
         ("SCAD ($a=3.7$)", G_scad(3.7), ":"), ("OP ($c=1$, $b=3$)", G_op(1.0, 3.0), "-")]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
u = np.linspace(0, 1, 1001); y = np.linspace(0, 8, 1601); lam = 2.0
for i, (lab, G, ls) in enumerate(rules):
    ax[0].plot(u, G(u), color=C[i], lw=2, ls=ls, label=lab)
    ax[1].plot(y, grit(y, lam, G), color=C[i], lw=2, ls=ls, label=lab)
ax[1].plot(y, y, color=ink2, lw=0.8, ls=(0, (2, 2)))
ax[0].set_xlabel("$u=1-\\lambda/|y|$"); ax[0].set_ylabel("gate $G(u)$"); ax[0].set_title("(a) canonical gates", loc="left", fontsize=9.5)
ax[1].set_xlabel("$y$"); ax[1].set_ylabel("$\\delta(y)$"); ax[1].set_title("(b) rules, $\\lambda=2$", loc="left", fontsize=9.5)
for a in ax: a.grid(color=grid_c, lw=0.6); a.set_axisbelow(True)
ax[0].legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "gates_rules.pdf")

# ---------------- Figure 2
HERE = pathlib.Path(__file__).resolve().parent
d = json.load(open(HERE.parent / "bayes_risk" / "bayes_study.json"))
pri = [("laplace", "Laplace slab"), ("t3", "$t_3$ slab"), ("gauss", "Gaussian slab"), ("two4", "two-point $\\pm4$")]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for i, (k, lab) in enumerate(pri):
    P = prior_moments(k, 0.05); r, G, yy = oracle_gate_risk(P, 2.0); uu = 1 - 2.0/yy; sel = uu < 0.995
    ax[0].plot(uu[sel], G[sel], color=C[i], lw=2, label=lab)
    fr = np.array(d[f"{k}_0.05"]["frontier"]); B = d[f"{k}_0.05"]["bayes"]
    ax[1].plot(fr[:, 0], fr[:, 1]/B, color=C[i], lw=2, label=lab)
    for nm, mk in (("soft", "s"), ("firm", "D"), ("op", "o")):
        rr, ll, _ = d[f"{k}_0.05"][nm]
        ax[1].plot(ll, rr/B, marker=mk, ms=6, color=C[i], mec="white", mew=1, ls="none")
ax[0].plot(u, u, color=ink2, lw=0.8, ls=(0, (2, 2)))
_, lop, thop = d["laplace_0.05"]["op"]
ax[0].plot(u, G_op(*thop)(u), color=ink2, lw=1.4, ls="--", label=f"tuned OP gate, Laplace ($b={thop[1]:.1f}$)")
ax[0].set_xlabel("$u$"); ax[0].set_ylabel("oracle gate $G^*_\\lambda(u)$"); ax[0].set_title("(a) oracle gates, $\\lambda=2$, $\\epsilon=0.05$", loc="left", fontsize=9.5)
ax[1].set_xlabel("threshold $\\lambda$"); ax[1].set_ylabel("$R^*(\\lambda)/r_B$"); ax[1].set_ylim(0.98, 1.65)
ax[1].set_title("(b) oracle risk at threshold $\\lambda$, $\\epsilon=0.05$", loc="left", fontsize=9.5)
from matplotlib.lines import Line2D
hs = [Line2D([], [], marker=m, ls="none", color=ink2, mec="white", ms=6) for m in ("s", "D", "o")]
ax[1].legend(hs, ["tuned soft", "tuned firm/MCP", "tuned OP"], fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.33))
for a in ax: a.grid(color=grid_c, lw=0.6); a.set_axisbelow(True)
ax[0].legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "oracle_frontier.pdf")

# ---------------- Figure 3: frequentist risk curves, rules tuned for Laplace slab eps = 0.05
row = d["laplace_0.05"]; yg = np.linspace(-15, 25, 8001); hy = yg[1] - yg[0]
mk = {"soft": lambda th: G_soft, "hybrid": lambda th: G_hybrid(*th), "firm": lambda th: G_firm(*th), "scad": lambda th: G_scad(*th), "op": lambda th: G_op(*th)}
labs = {"soft": "soft", "hybrid": "power garrote", "firm": "firm/MCP", "scad": "SCAD", "op": "OP"}
mus = np.linspace(0, 8, 161)
fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.0))
for i, nm in enumerate(["soft", "hybrid", "firm", "scad", "op"]):
    r, lam, th = row[nm]; G = mk[nm](th); dlt = grit(yg, lam, G)
    rk = [np.sum((dlt - m)**2*norm.pdf(yg - m))*hy for m in mus]
    ax.plot(mus, rk, color=C[i], lw=2, ls=["-", "--", "-.", ":", "-"][i], label=f"{labs[nm]} ($\\lambda={lam:.2f}$)")
ax.axhline(1, color=ink2, lw=0.8, ls=(0, (2, 2)))
ax.set_xlabel("$\\mu$"); ax.set_ylabel("$E_\\mu\\{\\delta(Y)-\\mu\\}^2$"); ax.grid(color=grid_c, lw=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=7.5); fig.tight_layout(); fig.savefig(OUT / "risk_curves.pdf")
print("done")

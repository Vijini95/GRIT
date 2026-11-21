"""Exact Bayes-risk comparison of thresholding rules under sparse mixture priors.
For each prior: Bayes risk (posterior mean), posterior median, oracle GRIT frontier R*(lam), and tuned
soft, hard, hybrid (power garrote), firm (= MCP), SCAD and linear-entry odds-power (OP) rules.
Output: bayes_study.json"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "grit"))

import json, itertools, numpy as np
from scipy.optimize import minimize
from bayes import *
from gates import *

lams = np.round(np.arange(0.3, 5.01, 0.05), 3)
fam = {
    "soft": (lambda th: G_soft, [()]),
    "hard": (lambda th: G_hard, [()]),
    "hybrid": (lambda th: G_hybrid(th[0]), [(b,) for b in np.geomspace(1.0, 20.0, 25)]),
    "firm": (lambda th: G_firm(th[0]), [(a,) for a in np.geomspace(1.05, 20.0, 25)]),
    "scad": (lambda th: G_scad(th[0]), [(a,) for a in np.geomspace(2.05, 20.0, 25)]),
    "op": (lambda th: G_op(th[0], th[1]), [(c, b) for c in np.geomspace(0.01, 1.0, 15) for b in np.geomspace(1.0, 8.0, 12)]),
}
bounds = {"hybrid": [(1, 50)], "firm": [(1.01, 50)], "scad": [(2.01, 50)], "op": [(1e-3, 1.0), (1.0, 20)]}


def tune(P, name):
    mk, shapes = fam[name]; best = (np.inf, None, None)
    for th in shapes:
        G = mk(th); rs = [risk(P, grit(yp, l, G)) for l in lams]; i = int(np.argmin(rs))
        if rs[i] < best[0]: best = (rs[i], lams[i], th)
    if name in ("soft", "hard"): return best
    x0 = np.r_[best[1], best[2]]; bd = [(0.05, 8.0)] + bounds[name]
    def obj(x):
        x = np.clip(x, [b[0] for b in bd], [b[1] for b in bd])
        return risk(P, grit(yp, x[0], mk(x[1:])))
    res = minimize(obj, x0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-9, maxiter=400))
    x = np.clip(res.x, [b[0] for b in bd], [b[1] for b in bd])
    return (min(res.fun, best[0]), float(x[0]), tuple(float(v) for v in x[1:]))


out = {}
for kind, eps in itertools.product(("laplace", "t3", "gauss", "two4"), (0.01, 0.05, 0.2)):
    P = prior_moments(kind, eps); rB = P["rB"]; row = {"bayes": rB}
    row["postmed"] = risk(P, posterior_median(P))
    fr = [(l, oracle_gate_risk(P, l)[0]) for l in np.arange(0.5, 4.51, 0.25)]
    row["frontier"] = fr
    for name in fam:
        r, l, th = tune(P, name); row[name] = (r, float(l), [float(v) for v in th])
        row[name + "_frontier_at_lam"] = oracle_gate_risk(P, l)[0]
    out[f"{kind}_{eps}"] = row
    print(f"{kind:8s} eps={eps:<5} " + " ".join(f"{n} {row[n][0]/rB:.3f}" for n in fam) + f" postmed {row['postmed']/rB:.3f} | " +
          " ".join(f"{n}:lam={row[n][1]:.2f},th={np.round(row[n][2],2).tolist()}" for n in ("firm", "op", "hybrid")), flush=True)
json.dump(out, open("bayes_study.json", "w"), indent=1)

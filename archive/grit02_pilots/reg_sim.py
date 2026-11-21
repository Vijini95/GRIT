"""Pilot regression study: coordinate descent with GRIT proximal updates (standardized columns),
pathwise over lambda with warm starts, lambda chosen on an independent validation set.
Design: n = 100, p = 200, AR(1) correlation rho; 10 nonzero coefficients.
Usage: python reg_sim.py R rho"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "grit"))

import sys, json, time, numpy as np
from gates import G_soft, G_firm, G_scad, G_op, G_hybrid

R, rho = int(sys.argv[1]), float(sys.argv[2]); n, p, sig = 100, 200, 1.5
beta0 = np.zeros(p); idx = np.arange(5, 200, 20)[:10]
beta0[idx] = [3, 1.5, 2, -3, -1.5, -2, 3, 1.5, 2, -2]
methods = {"lasso": G_soft, "mcp": G_firm(3.0), "scad": G_scad(3.7), "garrote": G_hybrid(2.0), "op": G_op(1.0, 3.0)}
Lc = np.linalg.cholesky(rho**np.abs(np.subtract.outer(np.arange(p), np.arange(p))))


def prox(z, lam, G):
    a = abs(z)
    if a <= lam: return 0.0
    return z*G(np.array([1 - lam/a]))[0]


def cd_path(Q, c, lams, G, tol=1e-7, maxit=500):
    b = np.zeros(p); path = []
    for lam in lams:
        for it in range(maxit):
            # full sweep
            maxd = 0.0
            for j in range(p):
                z = c[j] - Q[j] @ b + b[j]; nb = prox(z, lam, G)
                if nb != b[j]: maxd = max(maxd, abs(nb - b[j])); b[j] = nb
            if maxd < tol: break
            # active-set sweeps
            for it2 in range(maxit):
                act = np.nonzero(b)[0]; maxd = 0.0
                for j in act:
                    z = c[j] - Q[j] @ b + b[j]; nb = prox(z, lam, G)
                    if nb != b[j]: maxd = max(maxd, abs(nb - b[j])); b[j] = nb
                if maxd < tol: break
        path.append(b.copy())
    return path


rng = np.random.default_rng(11); res = {m: {"err": [], "pe": [], "fp": [], "fn": [], "size": []} for m in methods}
t0 = time.time()
for r in range(R):
    X = rng.standard_normal((n, p)) @ Lc.T; mu_x = X.mean(0); sd_x = X.std(0); Xs = (X - mu_x)/sd_x
    y = X @ beta0 + sig*rng.standard_normal(n); ym = y.mean()
    Xv = rng.standard_normal((n, p)) @ Lc.T; yv = Xv @ beta0 + sig*rng.standard_normal(n)
    Xt = rng.standard_normal((1000, p)) @ Lc.T; yt = Xt @ beta0 + sig*rng.standard_normal(1000)
    Q = Xs.T @ Xs/n; c = Xs.T @ (y - ym)/n
    lmax = np.max(np.abs(c)); lams = lmax*np.geomspace(1, 0.02, 40)
    for m, G in methods.items():
        path = cd_path(Q, c, lams, G)
        best = None
        for bs in path:
            bh = bs/sd_x; a0 = ym - mu_x @ bh; e = np.mean((yv - a0 - Xv @ bh)**2)
            if best is None or e < best[0]: best = (e, bh, a0)
        _, bh, a0 = best
        res[m]["err"].append(float(np.sum((bh - beta0)**2))); res[m]["pe"].append(float(np.mean((yt - a0 - Xt @ bh)**2)))
        sel = bh != 0; res[m]["fp"].append(int(np.sum(sel & (beta0 == 0)))); res[m]["fn"].append(int(np.sum(~sel & (beta0 != 0))))
        res[m]["size"].append(int(sel.sum()))
    if r % 20 == 0: print(r, f"{time.time()-t0:.0f}s", flush=True)
out = {m: {k: [float(np.mean(v)), float(np.std(v)/np.sqrt(R))] for k, v in d.items()} for m, d in res.items()}
for m, d in out.items(): print(m, " ".join(f"{k} {v[0]:.3f}" for k, v in d.items()))
json.dump(out, open(f"reg_sim_{rho}.json", "w"), indent=1)

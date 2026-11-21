"""Shared simulation tools: rule families on grids with SURE, EB posterior median (Laplace prior),
Daubechies filters by spectral factorization, periodic DWT, Donoho-Johnstone test signals."""
import numpy as np
from math import comb
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from gates import G_soft, G_hybrid, G_firm, G_scad, G_op

# ------------------------------------------------------------------ rule families (canonical gates)
def rule_slope(y, lam, G, eps=1e-5):
    a = np.abs(y); d = np.zeros_like(y); s = a > lam
    u = 1 - lam/a[s]; d[s] = y[s]*G(u)
    # derivative by centered difference in |y| on the active set (piecewise smooth gates)
    up = 1 - lam/(a[s] + eps); um = 1 - lam/np.maximum(a[s] - eps, lam)
    ds = np.zeros_like(y); ds[s] = ((a[s] + eps)*G(up) - np.maximum(a[s] - eps, lam)*G(um))/(a[s] + eps - np.maximum(a[s] - eps, lam))
    return d, ds

FAMILIES = {
    "soft":   [("soft", G_soft)],
    "hybrid": [(f"b={b:.2f}", G_hybrid(b)) for b in np.geomspace(1.0, 12.0, 12)],
    "firm":   [(f"a={a:.2f}", G_firm(a)) for a in np.geomspace(1.2, 10.0, 12)],
    "scad":   [(f"a={a:.2f}", G_scad(a)) for a in np.geomspace(2.1, 10.0, 12)],
    "op1":    [(f"b={b:.2f}", G_op(1.0, b)) for b in np.geomspace(1.0, 12.0, 12)],
    "op2":    [(f"c={c:.2f},b={b:.2f}", G_op(c, b)) for c in (0.1, 0.2, 0.35, 0.5, 0.7, 1.0) for b in np.geomspace(1.0, 12.0, 8)],
}


def sure_select(y, fam, lams, mu=None):
    """returns estimate chosen by SURE, and (if mu given) the oracle-on-grid estimate and its loss."""
    best = (np.inf, None); orc = (np.inf, None)
    for name, G in FAMILIES[fam]:
        for lam in lams:
            d, ds = rule_slope(y, lam, G)
            s = np.sum((y - d)**2) + 2*np.sum(ds) - len(y)
            if s < best[0]: best = (s, d)
            if mu is not None:
                l = np.sum((d - mu)**2)
                if l < orc[0]: orc = (l, d)
    return best[1], orc


# ------------------------------------------------------------------ EB posterior median, Laplace prior (Johnstone-Silverman)
def eb_postmed(x, a=0.5):
    n = len(x); C = 0.5*a*np.exp(a*a/2)
    ax = np.abs(x)
    gplus = C*np.exp(-a*ax + norm.logcdf(ax - a))               # theta>0 part of g(|x|)
    gminus = C*np.exp(a*ax + norm.logsf(ax + a))
    g = gplus + gminus; ph = norm.pdf(x)
    tuniv = np.sqrt(2*np.log(n))
    # lower bound for w: universal threshold (as in EbayesThresh)
    def negll(w): return -np.sum(np.log((1 - w)*ph + w*g))
    wlo = 1e-6
    res = minimize_scalar(negll, bounds=(wlo, 1.0), method="bounded")
    w = res.x
    f = (1 - w)*ph + w*g
    # median for |x|: m = |x| - a - Phi^{-1}(z), z = f e^{a|x|} / (2 w C) ... computed on log scale
    logz = np.log(f) + a*ax - np.log(2*w*C)
    z = np.exp(np.minimum(logz, 0.0))
    med = np.where(logz >= 0, 0.0, np.maximum(0.0, ax - a - norm.ppf(np.clip(z, 1e-300, 1 - 1e-16))))
    # z refers to Phi(|x|-a-m); the median is positive only if P(theta>0|x) > 1/2
    return np.sign(x)*med


# ------------------------------------------------------------------ Daubechies filters and periodic DWT
def daubechies(N):
    P = [comb(N - 1 + k, k) for k in range(N)]
    yr = np.roots(P[::-1])                                            # roots in y = sin^2(w/2)
    zs = []
    for yk in yr:
        r = np.roots([1, -(2 - 4*yk), 1]); zs.append(r[np.argmin(np.abs(r))])
    poly = np.array([1.0])
    for _ in range(N): poly = np.convolve(poly, [0.5, 0.5])
    for zk in zs: poly = np.convolve(poly, [1, -zk])
    hh = np.real(poly); hh = hh*np.sqrt(2)/hh.sum()
    return hh


def dwt(x, hh, j0):
    gg = hh[::-1]*(-1)**np.arange(len(hh)); L = len(hh); a = x.copy(); det = {}
    J = int(np.log2(len(x)))
    for j in range(J - 1, j0 - 1, -1):
        n = len(a); idx = (2*np.arange(n//2)[:, None] + np.arange(L)[None, :]) % n
        det[j] = a[idx] @ gg; a = a[idx] @ hh
    return a, det


def idwt(a, det, hh, j0):
    gg = hh[::-1]*(-1)**np.arange(len(hh)); L = len(hh)
    for j in range(j0, max(det) + 1):
        n = 2*len(a); idx = (2*np.arange(n//2)[:, None] + np.arange(L)[None, :]) % n
        x = np.zeros(n)
        np.add.at(x, idx, a[:, None]*hh[None, :] + det[j][:, None]*gg[None, :]); a = x
    return a


def dj_signal(name, n):
    t = np.arange(1, n + 1)/n
    tj = np.array([.1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
    if name == "blocks":
        hj = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
        f = np.sum(hj[:, None]*(1 + np.sign(t[None, :] - tj[:, None]))/2, 0)
    elif name == "bumps":
        hj = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
        wj = np.array([.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005])
        f = np.sum(hj[:, None]*(1 + np.abs((t[None, :] - tj[:, None])/wj[:, None]))**-4, 0)
    elif name == "heavisine":
        f = 4*np.sin(4*np.pi*t) - np.sign(t - .3) - np.sign(.72 - t)
    elif name == "doppler":
        f = np.sqrt(t*(1 - t))*np.sin(2*np.pi*1.05/(t + .05))
    return f

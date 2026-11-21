"""Exact (numerical-integration) Bayes risks of thresholding rules under sparse mixture priors
theta ~ (1-eps) delta_0 + eps * slab, Y | theta ~ N(theta, 1).
Risk identity: r(delta) = r_Bayes + int (delta - m)^2 f dy, with m the posterior mean and f the marginal."""
import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import norm, t as tdist
from sklearn.isotonic import IsotonicRegression

h = 0.01; L = 300.0
grid = np.round(np.arange(-L, L + h/2, h), 6)
pos = grid >= 0
yp = grid[pos]                      # y >= 0 (risk integrals are symmetric)
wq = np.full(yp.shape, 2*h); wq[0] = h   # symmetric quadrature weights on y >= 0


def slab_density(kind):
    th = grid
    if kind == "laplace":            # (a/2) exp(-a|th|), a = 0.5
        a = 0.5; return 0.5*a*np.exp(-a*np.abs(th))
    if kind == "t3":                 # Student t, 3 df, scale 2
        return tdist.pdf(th/2.0, 3)/2.0
    if kind == "gauss":              # N(0, 3^2)
        return norm.pdf(th, scale=3.0)
    raise ValueError(kind)


def prior_moments(kind, eps):
    """returns f, m, V on y >= 0, and the Bayes risk."""
    phi_k = norm.pdf(np.arange(-12, 12 + h/2, h))
    if kind.startswith("two"):       # +-mu0 with equal probability
        mu0 = float(kind[3:])
        y = yp; f1 = 0.5*(norm.pdf(y - mu0) + norm.pdf(y + mu0))
        n1 = 0.5*mu0*(norm.pdf(y - mu0) - norm.pdf(y + mu0)); n2 = mu0**2*f1
    else:
        g = slab_density(kind); g = g/(g.sum()*h)
        f1 = np.convolve(g, phi_k, mode="same")*h
        n1 = np.convolve(g*grid, phi_k, mode="same")*h
        n2 = np.convolve(g*grid**2, phi_k, mode="same")*h
        f1, n1, n2 = f1[pos], n1[pos], n2[pos]
    f = (1 - eps)*norm.pdf(yp) + eps*f1
    ok = f > 1e-250
    m = np.where(ok, eps*n1/np.where(ok, f, 1), yp); V = np.where(ok, eps*n2/np.where(ok, f, 1) - m**2, 0.0)
    f = np.where(ok, f, 0.0); V = np.maximum(V, 0.0)
    rB = np.sum(V*f*wq)
    return dict(f=f, m=m, V=V, rB=rB, kind=kind, eps=eps)


def risk(P, d):
    """d: rule evaluated on yp (y >= 0)."""
    return P["rB"] + np.sum((d - P["m"])**2*P["f"]*wq)


def posterior_median(P, ymax=40.0):
    """posterior median on y in [0, ymax], posterior mean beyond."""
    kind, eps = P["kind"], P["eps"]
    out = P["m"].copy(); sel = np.where(yp <= ymax)[0]
    if kind.startswith("two"):
        mu0 = float(kind[3:]); y = yp[sel]
        w0 = (1 - eps)*norm.pdf(y); wp = 0.5*eps*norm.pdf(y - mu0); wm = 0.5*eps*norm.pdf(y + mu0)
        tot = w0 + wp + wm
        out[sel] = np.where(wp/tot > 0.5, mu0, np.where((wm + w0)/tot >= 0.5, np.where(wm/tot > 0.5, -mu0, 0.0), mu0))
        return out
    g = slab_density(kind); g = g/(g.sum()*h); th = grid
    for i0 in range(0, len(sel), 400):
        ii = sel[i0:i0+400]; y = yp[ii][:, None]
        lik = norm.pdf(y - th[None, :])*g[None, :]*h*eps          # continuous part
        atom = (1 - eps)*norm.pdf(yp[ii])
        tot = lik.sum(1) + atom
        cneg = np.cumsum(lik*(th[None, :] < 0), 1)[:, -1]/tot       # P(theta<0 | y)
        # median: if P(theta<0)+P(atom) >= 1/2 and P(theta<0) <= 1/2 -> 0 (for y>=0 the negative mass < 1/2)
        cdf = np.cumsum(lik, 1)/tot[:, None] + (th[None, :] >= 0)*(atom/tot)[:, None]
        idx = np.argmax(cdf >= 0.5, axis=1); out[ii] = th[idx]
        out[ii] = np.where((cneg + atom/tot >= 0.5) & (cneg <= 0.5), 0.0, out[ii])
    return out


def oracle_gate_risk(P, lam):
    """oracle GRIT gate at threshold lam: clipped weighted isotonic regression of m(y)/y on y > lam."""
    s = (yp > lam) & (P["f"] > 0); y = yp[s]; w = y**2*P["f"][s]*wq[s]; rho = P["m"][s]/y
    G = IsotonicRegression(y_min=0.0, y_max=1.0).fit(y, rho, sample_weight=w).predict(y)
    d = np.zeros_like(yp); d[s] = y*G
    return risk(P, d), G, y

"""GRIT rules in canonical (p=1) coordinates u = 1 - lam/|y| and classical comparators.
All rules are odd; functions take y (array) and return delta(y)."""
import numpy as np


def grit(y, lam, G):
    a = np.abs(y); out = np.zeros_like(y, dtype=float); s = a > lam
    u = 1.0 - lam/a[s]; out[s] = y[s]*G(u); return out


def G_soft(u): return u
def G_hard(u): return np.ones_like(u)
def G_hybrid(b): return lambda u: 1.0 - (1.0 - u)**b                       # power garrote; b=2 garrote
def G_firm(a): return lambda u: np.minimum(a*u/(a - 1.0), 1.0)              # firm = MCP (orthogonal)
def G_scad(a):
    def G(u):
        return np.where(u <= 0.5, u, np.where(u <= 1 - 1/a, (a*u - 1)/(a - 2), 1.0))
    return G
def G_op(c, b):                                                                # linear-entry odds-power gate
    return lambda u: u/(u + c*(1.0 - u)**b)


def slope(y, lam, G, eps=1e-6):
    """numerical derivative of the rule (for SURE)."""
    return (grit(y + eps, lam, G) - grit(y - eps, lam, G))/(2*eps)


# closed-form derivatives on the active set (canonical coordinates): delta' = G(u) + (1-u) g(u)
def dG_hybrid(b): return lambda u: b*(1 - u)**(b - 1)
def dG_op(c, b):
    def g(u):
        den = u + c*(1 - u)**b
        return (c*(1 - u)**b + c*b*u*(1 - u)**(b - 1))/den**2
    return g


def rule_and_slope(y, lam, G, g):
    a = np.abs(y); d = np.zeros_like(y, dtype=float); ds = np.zeros_like(y, dtype=float); s = a > lam
    u = 1.0 - lam/a[s]; d[s] = y[s]*G(u); ds[s] = G(u) + (1 - u)*g(u)
    return d, ds

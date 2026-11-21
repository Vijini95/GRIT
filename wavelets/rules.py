import numpy as np
from scipy.special import ndtr, ndtri, log_ndtr
# ---------- gates: return G(u), g(u) on u in (0,1) ----------
def gate(fam,u,s):
    if fam=="soft":   return u, np.ones_like(u)
    if fam=="op":     # c=1
        w=(1-u)**s; D=u+w; return u/D, (1-u)**(s-1)*((1-u)+s*u)/D**2
    if fam=="garrote":return 1-(1-u)**s, s*(1-u)**(s-1)
    if fam=="firm":
        k=s/(s-1); return np.minimum(k*u,1.0), np.where(u<1-1/s,k,0.0)
    if fam=="scad":
        a=s; G=np.where(u<=.5,u,np.where(u<=1-1/a,(a*u-1)/(a-2),1.0))
        g=np.where(u<=.5,1.0,np.where(u<=1-1/a,a/(a-2),0.0)); return G,g
def rule(fam,y,lam,s):
    ay=np.abs(y); act=ay>lam; u=np.where(act,1-lam/np.where(act,ay,1),0.0)
    G,_=gate(fam,u,s); return np.where(act,y*G,0.0)
def sure_grid(fam,Y,lams,s):
    """Y: (B,n) standardized; lams: (K,). returns SURE (B,K)."""
    ay=np.abs(Y)[:,:,None]; L=lams[None,None,:]; act=ay>L
    u=np.where(act,1-L/np.where(act,ay,1),0.0); G,g=gate(fam,u,s)
    val=np.where(act,ay**2*(1-G)**2+2*(G+(1-u)*g)-1, ay**2-1)
    return val.sum(1)
def loss_grid(fam,Y,th_std,lams,s):
    ay=np.abs(Y)[:,:,None]; L=lams[None,None,:]; act=ay>L
    u=np.where(act,1-L/np.where(act,ay,1),0.0); G,_=gate(fam,u,s)
    est=np.where(act,Y[:,:,None]*G,0.0); return ((est-th_std[:,:,None])**2).sum(1)
# ---------- empirical Bayes posterior median, Laplace prior a=0.5 (Johnstone-Silverman) ----------
A=0.5
def _logg(x):   # log of Laplace*phi marginal (density), x>=0 elementwise uses |x|
    x=np.abs(x); t1=-A*x+log_ndtr(x-A); t2=A*x+log_ndtr(-x-A)
    return np.log(A/2)+A*A/2+np.logaddexp(t1,t2)
def _logphi(x): return -0.5*x*x-0.5*np.log(2*np.pi)
def _ratio(x):  # Phi(x-a) / [Phi(x-a)+exp(2ax)Phibar(x+a)], x>=0
    l1=log_ndtr(x-A); l2=2*A*x+log_ndtr(-x-A); return np.exp(l1-np.logaddexp(l1,l2))
def w_from_t(t):
    beta=np.exp(_logg(t)-_logphi(t))-1; R=_ratio(t); return 1/(2*(beta+1)*R-beta)
def eb_median(Y):
    """Y: (B,n) standardized level coefficients. Returns posterior medians (MML weight per row)."""
    B,n=Y.shape; wlo=min(1.0,float(w_from_t(np.sqrt(2*np.log(n)))))
    ws=np.exp(np.linspace(np.log(wlo),0,200))
    lb=_logg(Y)-_logphi(Y)   # log(g/phi)
    # loglik(w)=sum log(1-w+w*exp(lb))
    lw=np.log(ws)[None,None,:]; l1w=np.log1p(-np.minimum(ws,1-1e-15))[None,None,:]
    ll=np.logaddexp(l1w,lw+lb[:,:,None]).sum(1)                 # sum log(1-w + w g/phi)
    w=ws[np.argmax(ll,1)][:,None]
    x=np.abs(Y); wpost=1/(1+np.exp(np.clip(-(lb+np.log(w)-np.log1p(-np.minimum(w,1-1e-15))),-700,700)))
    z=0.5*(1+np.exp(2*A*x+log_ndtr(-x-A)-log_ndtr(x-A)))/wpost  # = 0.5[Phi(x-a)+e^{2ax}Phibar(x+a)]/(wpost Phi(x-a))
    # median t solves Phi(x-a-t)=z*Phi(x-a)  -> t = x-a-ndtri(z*Phi(x-a))
    zz=z*ndtr(x-A); t=np.where(zz<1, x-A-ndtri(np.minimum(zz,1-1e-16)), 0.0)
    return np.sign(Y)*np.maximum(t,0.0)

"""Simulation engine for the GRIT wavelet study: periodized orthonormal DWT (Symmlet-4, db3, Haar by spectral factorization) and Donoho-Johnstone test signals."""
import numpy as np
from numpy.polynomial import polynomial as P

def daubechies(N):
    """Minimum-phase Daubechies filter with N vanishing moments (2N taps), via spectral factorization."""
    from scipy.special import comb
    q=np.array([comb(N-1+k,k) for k in range(N)])           # P(y)=sum C(N-1+k,k) y^k, y=sin^2(w/2)
    # roots in z: y=(2 - z - 1/z)/4 ; build polynomial in z of degree 2(N-1)
    poly=np.zeros(1)
    for k in range(N):
        t=np.array([1.0])
        for _ in range(k): t=P.polymul(t,np.array([-1/4,1/2,-1/4]))  # (2 - z - z^-1)/4 * z  -> coeffs in z
        t=np.concatenate([np.zeros(N-1-k),t,np.zeros(N-1-k)])
        poly=t*q[k] if poly.size==1 else poly+t*q[k]
    r=np.roots(poly[::-1]); r=r[np.abs(r)<1]                  # minimum phase: roots inside unit circle
    h=np.real(np.poly(np.concatenate([r,-np.ones(N)])))[::-1]
    h=h/np.sum(h)*np.sqrt(2); return h

SYM4=np.array([-0.07576571478927333,-0.02963552764599851,0.49761866763201545,0.8037387518059161,
               0.29785779560527736,-0.09921954357684722,-0.012603967262037833,0.0322231006040427])
HAAR=np.array([1,1])/np.sqrt(2)

def check(h,name):
    L=len(h); orth=max(abs(sum(h[n]*h[n+2*k] for n in range(L-2*k))-(k==0)) for k in range(L//2))
    g=np.array([(-1)**n*h[L-1-n] for n in range(L)])
    vm=[abs(sum(g[n]*n**p for n in range(L))) for p in range(L//2)]
    print(f"{name}: taps={L} sum={h.sum():.12f} orth_err={orth:.1e} vanishing moments err={max(vm):.1e}")

def dwt_per(x,h,j0):
    """Periodized orthonormal DWT down to level j0. Returns list [a_j0, d_j0, d_j0+1, ..., d_{J-1}]."""
    L=len(h); g=np.array([(-1)**n*h[L-1-n] for n in range(L)])
    a=x.astype(float); details=[]
    while a.shape[-1]>2**j0:
        m=a.shape[-1]; idx=(2*np.arange(m//2)[:,None]+np.arange(L)[None,:])%m
        A=a[...,idx]; details.append(A@g); a=A@h
    return [a]+details[::-1]

def signal(name,N):
    t=np.arange(1,N+1)/N
    pos=np.array([.1,.13,.15,.23,.25,.40,.44,.65,.76,.78,.81])
    if name=="Blocks":
        hb=np.array([4,-5,3,-4,5,-4.2,2.1,4.3,-3.1,2.1,-4.2])
        return (hb[None,:]*(1+np.sign(t[:,None]-pos[None,:]))/2).sum(1)
    if name=="Bumps":
        hb=np.array([4,5,3,4,5,4.2,2.1,4.3,3.1,5.1,4.2]); w=np.array([.005,.005,.006,.01,.01,.03,.01,.01,.005,.008,.005])
        return (hb[None,:]*(1+np.abs((t[:,None]-pos[None,:])/w[None,:]))**-4).sum(1)
    if name=="HeaviSine": return 4*np.sin(4*np.pi*t)-np.sign(t-.3)-np.sign(.72-t)
    if name=="Doppler": return np.sqrt(t*(1-t))*np.sin(2*np.pi*1.05/(t+.05))
WAVELET={"Blocks":"haar","Bumps":"db3","HeaviSine":"sym4","Doppler":"sym4"}
FILT={"haar":HAAR,"db3":daubechies(3),"sym4":SYM4}

def true_coeffs(name,N,rho,j0=3):
    f0=signal(name,N); f=f0*rho/np.std(f0)
    c=dwt_per(f,FILT[WAVELET[name]],j0)
    return c   # [a(8), d_3..d_{J-1}]

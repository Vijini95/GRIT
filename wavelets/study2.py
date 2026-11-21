"""Supplementary GRIT variants on the SAME noise realizations as study.py (same seed, same draw order).
Usage: python3 study2.py N SIGNAL RHO M SEED"""
import sys, time, numpy as np, engine as e, rules as r
N=int(sys.argv[1]); SIG=sys.argv[2]; RHO=float(sys.argv[3]); M=int(sys.argv[4]); SEED=int(sys.argv[5])
j0=3; h=e.FILT[e.WAVELET[SIG]]
if SIG=="Bumps": h=h[::-1]
f0=e.signal(SIG,N); f=f0*RHO/np.std(f0); levels=e.dwt_per(f,h,j0)[1:]
LAMG=lambda n: np.linspace(0,np.sqrt(2*np.log(n)),50)
BGRID=np.array([1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,8.])
METHODS=["GRIT-b2","GRIT-b5","GRIT-SUREglob","garrote-b2","firm-a3"]
rng=np.random.default_rng(SEED); loss=np.zeros((M,5)); selg=[]; t0=time.time(); B=100
for b0 in range(0,M,B):
    Bn=min(B,M-b0)
    za=rng.standard_normal((Bn,2**j0)); Z=[rng.standard_normal((Bn,len(d))) for d in levels]
    D=[d[None,:]+z for d,z in zip(levels,Z)]
    s=(np.median(np.abs(D[-1]),1)/0.67448975)[:,None]
    L=np.repeat((za**2).sum(1)[:,None],5,1)
    Stot=np.zeros((Bn,len(BGRID))); LOSS=np.zeros((Bn,len(BGRID)))
    for d,Dl in zip(levels,D):
        Y=Dl/s; lams=LAMG(len(d))
        for col,fam,b in [(0,"op",2.0),(1,"op",5.0),(3,"garrote",2.0),(4,"firm",3.0)]:
            S=r.sure_grid(fam,Y,lams,b); lam=lams[np.argmin(S,1)]
            L[:,col]+=((s*r.rule(fam,Y,lam[:,None],b)-d[None,:])**2).sum(1)
        for ib,b in enumerate(BGRID):
            S=r.sure_grid("op",Y,lams,b); k=np.argmin(S,1); Stot[:,ib]+=S[np.arange(Bn),k]
            LOSS[:,ib]+=((s*r.rule("op",Y,lams[k][:,None],b)-d[None,:])**2).sum(1)
    ib=np.argmin(Stot,1); L[:,2]+=LOSS[np.arange(Bn),ib]; selg.append(BGRID[ib])
    loss[b0:b0+Bn]=L/N
np.savez(f"res3/{SIG}_N{N}_r{int(RHO)}.npz",loss=loss,methods=METHODS,selg=np.concatenate(selg))
print(f"{SIG} N={N} rho={RHO:g} ({time.time()-t0:.0f}s): "+"  ".join(f"{m}={v:.4f}" for m,v in zip(METHODS,loss.mean(0))),flush=True)

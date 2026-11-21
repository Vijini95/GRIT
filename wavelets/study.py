"""GRIT wavelet denoising study (Section 6.3): four Donoho-Johnstone signals, N in {512,1024,2048}, root SNR in {3,5,7}.  Usage: python3 study.py N SIGNAL RHO M SEED"""
import sys, time, numpy as np, engine as e, rules as r
N=int(sys.argv[1]); SIG=sys.argv[2]; RHO=float(sys.argv[3]); M=int(sys.argv[4]); SEED=int(sys.argv[5])
J=int(np.log2(N)); j0=3
h=e.FILT[e.WAVELET[SIG]]
if SIG=="Bumps": h=h[::-1]                      # time-reversed db3 filter (MATLAB wavedec orientation)
f0=e.signal(SIG,N); f=f0*RHO/np.std(f0)
coef=e.dwt_per(f,h,j0); levels=coef[1:]          # d_3..d_{J-1}
LAMG=lambda n: np.linspace(0,np.sqrt(2*np.log(n)),50)
BGRID=np.array([1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,8.])
AGRID=np.array([1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,8,10.])
METHODS=["UnivHard","UnivSoft","SureShrink","SURE-soft","SCAD-SURE","GRIT-fixed","GRIT-SURE","garrote-SURE","firm-SURE","EB-median"]
rng=np.random.default_rng(SEED)
loss=np.zeros((M,len(METHODS))); orc={"GRIT-SURE":np.zeros(M),"SURE-soft":np.zeros(M)}
selb=[]; t0=time.time(); B=100
for b0 in range(0,M,B):
    Bn=min(B,M-b0)
    za=rng.standard_normal((Bn,2**j0)); Z=[rng.standard_normal((Bn,len(d))) for d in levels]
    D=[d[None,:]+z for d,z in zip(levels,Z)]
    sig=np.median(np.abs(D[-1]),1)/0.67448975; s=sig[:,None]
    L=np.repeat((za**2).sum(1)[:,None],len(METHODS),1)   # unshrunk scaling coefficients
    lu=np.sqrt(2*np.log(N))
    ob={k:np.zeros(Bn) for k in orc}
    for d,Dl in zip(levels,D):
        n=len(d); Y=Dl/s; th=d[None,:]/s; lams=LAMG(n); add=lambda k,est: L.__setitem__((slice(None),k),L[:,k]+((s*est-d[None,:])**2).sum(1))
        add(0,np.where(np.abs(Y)>lu,Y,0)); add(1,r.rule("soft",Y,lu,1))
        # SURE over lambda for one-parameter families
        def sure1(fam,sh):
            S=r.sure_grid(fam,Y,lams,sh); k=np.argmin(S,1); return lams[k],S[np.arange(Bn),k],S
        ls,_,Ss=sure1("soft",1); est_ss=r.rule("soft",Y,ls[:,None],1); add(3,est_ss)
        Lg=r.loss_grid("soft",Y,th,lams,1); ob["SURE-soft"]+=s[:,0]**2*Lg.min(1)
        # SureShrink hybrid: sparse levels -> level universal threshold
        sd2=((Y**2).sum(1)-n)/n; gam=np.log2(n)**1.5/np.sqrt(n); lev_u=np.sqrt(2*np.log(n))
        add(2,np.where((sd2<=gam)[:,None], r.rule("soft",Y,lev_u,1), est_ss))
        lsc,_,_=sure1("scad",3.7); add(4,r.rule("scad",Y,lsc[:,None],3.7))
        lf,_,_=sure1("op",3.0); add(5,r.rule("op",Y,lf[:,None],3.0))
        # two-parameter families
        for col,fam,grid in [(6,"op",BGRID),(7,"garrote",BGRID),(8,"firm",AGRID)]:
            best=np.full(Bn,np.inf); bl=np.zeros(Bn); bs=np.zeros(Bn); orl=np.full(Bn,np.inf)
            for sh in grid:
                S=r.sure_grid(fam,Y,lams,sh); k=np.argmin(S,1); v=S[np.arange(Bn),k]
                better=v<best; best=np.where(better,v,best); bl=np.where(better,lams[k],bl); bs=np.where(better,sh,bs)
                if col==6: orl=np.minimum(orl,r.loss_grid(fam,Y,th,lams,sh).min(1))
            est=np.zeros_like(Y)
            for sh in np.unique(bs):
                m=bs==sh; est[m]=r.rule(fam,Y[m],bl[m][:,None],sh)
            add(col,est)
            if col==6: ob["GRIT-SURE"]+=s[:,0]**2*orl; selb.append(bs)
        add(9,r.eb_median(Y))
    loss[b0:b0+Bn]=L/N
    for k in orc: orc[k][b0:b0+Bn]=(ob[k]+(za**2).sum(1))/N
np.savez(f"res/{SIG}_N{N}_r{int(RHO)}.npz",loss=loss,methods=METHODS,orc_grit=orc["GRIT-SURE"],orc_soft=orc["SURE-soft"],selb=np.concatenate(selb))
print(f"{SIG} N={N} rho={RHO:g} M={M} ({time.time()-t0:.0f}s): "+"  ".join(f"{m}={v:.4f}" for m,v in zip(METHODS,loss.mean(0))))

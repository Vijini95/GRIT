"""Regression example for GRIT06 Section 6.4: penalized least squares by cyclic coordinate descent,
vectorized over replications.  Usage: python3 regsim.py RHOX  -> res_<RHOX>.npz
Design: n=100 training, n_val=100 validation (independent, same distribution), p=200,
rows N(0,Sigma), Sigma_jk = rhoX^|j-k|; beta nonzero at positions 10,30,...,190 with values
(1.5,-1.5,1.5,-2,2,-2,2,-3,3,-3); noise sd 1.5.  Columns of the training X are centered and
scaled to x_j'x_j/n = 1, y is centered; validation and estimation use the original scale.
Objective (2n)^-1||y-Xb||^2 + sum_j P_lam(|b_j|); coordinate update b_j <- delta_lam(x_j'r/n + b_j).
Path: 40 thresholds log-spaced from lam_max = max|x_j'y|/n to 0.02 lam_max, warm starts.
Convergence: max_j |change in b_j| < 1e-7 within a sweep, at most 1000 sweeps per threshold.
Prediction error: sigma^2 + (bhat-beta)' Sigma (bhat-beta), exact."""
import sys, numpy as np, time
RHOX=float(sys.argv[1]); R=200; n=100; nv=100; p=200; sig=1.5; NL=40
rng=np.random.default_rng(int(20260 + 100*RHOX))
beta=np.zeros(p); pos=np.arange(10,200,20); beta[pos]=[1.5,-1.5,1.5,-2,2,-2,2,-3,3,-3]
Sig=RHOX**np.abs(np.subtract.outer(np.arange(p),np.arange(p))); Lc=np.linalg.cholesky(Sig)
X=rng.standard_normal((R,n,p))@Lc.T; Xv=rng.standard_normal((R,nv,p))@Lc.T
y=X@beta+sig*rng.standard_normal((R,n)); yv=Xv@beta+sig*rng.standard_normal((R,nv))
mx=X.mean(1,keepdims=True); sd=np.sqrt(((X-mx)**2).mean(1,keepdims=True)); Xs=(X-mx)/sd
my=y.mean(1,keepdims=True); yc=y-my
def rule(name,z,lam):
    a=np.abs(z); s=np.sign(z); out=np.zeros_like(z); act=a>lam
    if name=="lasso": out=s*np.maximum(a-lam,0)
    elif name=="mcp":   A=3.0; out=np.where(a<=A*lam, s*np.maximum(a-lam,0)/(1-1/A), z)
    elif name=="scad":
        A=3.7; out=np.where(a<=2*lam, s*np.maximum(a-lam,0), np.where(a<=A*lam, ((A-1)*z-s*A*lam)/(A-2), z))
    elif name=="garrote": out[act]=z[act]-lam[act]**2/z[act] if np.ndim(lam) else z[act]-lam**2/z[act]
    elif name=="op":
        c,b=1.0,3.0; lam_=np.broadcast_to(lam,z.shape); S=a[act]/lam_[act]
        out[act]=s[act]*lam_[act]*S*(S-1)/(S-1+c*S**(1-b))
    return out
METHODS=["lasso","mcp","scad","garrote","op"]
lmax=np.abs(np.einsum('rnp,rn->rp',Xs,yc)).max(1)/n            # per replication
grid=np.exp(np.linspace(0,np.log(0.02),NL))                      # multiples of lam_max
res={}
for meth in METHODS:
    t0=time.time(); b=np.zeros((R,p)); r=yc.copy(); best=np.full(R,np.inf); bbest=np.zeros((R,p)); capped=0; sweeps=[]
    for g in grid:
        lam=lmax*g
        for sw in range(1000):
            dmax=np.zeros(R)
            for j in range(p):
                xj=Xs[:,:,j]; old=b[:,j]; z=(xj*r).sum(1)/n+old
                new=rule(meth,z,lam); d=new-old
                if np.any(d): r-=xj*d[:,None]; b[:,j]=new
                np.maximum(dmax,np.abs(d),out=dmax)
            if dmax.max()<1e-7: break
        else: capped+=1
        sweeps.append(sw+1)
        bo=b/sd[:,0,:]; b0=my[:,0]-(mx[:,0,:]*bo).sum(1)          # original scale with intercept
        ve=((yv-(Xv@bo[:,:,None])[:,:,0]-b0[:,None])**2).mean(1)
        upd=ve<best; best[upd]=ve[upd]; bbest[upd]=bo[upd]
    e=bbest-beta; est=(e**2).sum(1); pe=sig**2+np.einsum('rp,pq,rq->r',e,Sig,e)
    nz=bbest!=0; FP=(nz&(beta==0)).sum(1); FN=((~nz)&(beta!=0)).sum(1)
    res[meth]=dict(est=est,pe=pe,FP=FP,FN=FN,size=nz.sum(1))
    print(f"rhoX={RHOX} {meth:8s} est {est.mean():.3f} ({est.std(ddof=1)/np.sqrt(R):.3f})  PE {pe.mean():.3f}  FP {FP.mean():.2f} FN {FN.mean():.2f} size {nz.sum(1).mean():.1f}  "
          f"max sweeps {max(sweeps)} capped {capped}  [{time.time()-t0:.0f}s]",flush=True)
np.savez(f"res_{RHOX}.npz",**{f"{m}_{k}":v for m,d in res.items() for k,v in d.items()})

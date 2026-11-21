import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "grit"))
import json, itertools, numpy as np
from scipy.optimize import minimize
from bayes import *; from gates import *
d=json.load(open('bayes_study.json')); lams=np.arange(0.3,5.0,0.05)
for kind,eps in itertools.product(("laplace","t3","gauss","two4"),(0.01,0.05,0.2)):
    P=prior_moments(kind,eps); best=(9e9,0,0)
    for b in np.geomspace(1,15,30):
        rs=[risk(P,grit(yp,l,G_op(1.0,b))) for l in lams]; i=int(np.argmin(rs))
        if rs[i]<best[0]: best=(rs[i],lams[i],b)
    obj=lambda x: risk(P,grit(yp,max(x[0],0.05),G_op(1.0,max(x[1],1.0))))
    res=minimize(obj,[best[1],best[2]],method='Nelder-Mead',options=dict(xatol=1e-4,fatol=1e-10))
    r=min(res.fun,best[0]); lam=max(res.x[0],0.05); b=max(res.x[1],1.0)
    d[f"{kind}_{eps}"]["op1"]=(float(r),float(lam),[1.0,float(b)])
    d[f"{kind}_{eps}"]["op1_frontier_at_lam"]=oracle_gate_risk(P,lam)[0]
    print(kind,eps,round(r/P['rB'],4),round(lam,2),round(b,2),flush=True)
json.dump(d,open('bayes_study.json','w'),indent=1)

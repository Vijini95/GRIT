"""Aggregate study.py (res/) and study2.py (res3/) on common realizations; write LaTeX table bodies to tables/."""
import numpy as np, os
SIGS=["Doppler","HeaviSine","Bumps","Blocks"]; NS=[512,1024,2048]; RH=[3,5,7]
os.makedirs("tables",exist_ok=True)
def load(s,N,r):
    a=np.load(f"res/{s}_N{N}_r{r}.npz",allow_pickle=True); b=np.load(f"res3/{s}_N{N}_r{r}.npz",allow_pickle=True)
    L=np.hstack([a["loss"],b["loss"]]); M=list(a["methods"])+list(b["methods"]); return L,M,a,b
_,MALL,_,_=load("Doppler",512,3)
LAB={"UnivHard":"hard, universal","UnivSoft":"soft, universal","SureShrink":"SureShrink","SURE-soft":"soft",
 "garrote-b2":"garrote ($b=2$)","firm-a3":"firm ($a=3$)","SCAD-SURE":"SCAD ($a=3.7$)","GRIT-b2":"OP ($b=2$)","GRIT-fixed":"OP ($b=3$)",
 "GRIT-b5":"OP ($b=5$)","garrote-SURE":"garrote, $b$ tuned","firm-SURE":"firm, $a$ tuned","GRIT-SURE":"OP, $b$ tuned by level",
 "GRIT-SUREglob":"OP, $b$ tuned globally","EB-median":"EB posterior median"}
ORDER=["UnivHard","UnivSoft","SureShrink","SURE-soft","garrote-b2","firm-a3","SCAD-SURE","GRIT-b2","GRIT-fixed","GRIT-b5",
       "garrote-SURE","firm-SURE","GRIT-SURE","GRIT-SUREglob","EB-median"]
A={(s,N,r):load(s,N,r)[0].mean(0) for s in SIGS for N in NS for r in RH}
Q={N:np.mean([A[(s,N,r)]/A[(s,N,r)][0] for s in SIGS for r in RH],0) for N in NS}
print("Q_N"); lines=[]
for m in ORDER:
    i=MALL.index(m); q=[Q[N][i] for N in NS]; print(f"{m:14s} "+" ".join(f"{v:.4f}" for v in q)+f"  {np.mean(q):.4f}")
    lines.append(f"{LAB[m]} & "+" & ".join(f"{v:.3f}" for v in q)+f" & {np.mean(q):.3f} \\\\")
open("tables/qn.tex","w").write("\n".join(lines))
COLS=["SURE-soft","SureShrink","garrote-b2","firm-a3","SCAD-SURE","GRIT-b2","GRIT-fixed","GRIT-SURE","EB-median"]
for N in NS:
    out=[]
    for s in SIGS:
        for k,r in enumerate(RH):
            v=[A[(s,N,r)][MALL.index(m)] for m in COLS]; best=min(v)
            cells=[(f"\\textbf{{{x:.4f}}}" if abs(x-best)<5e-5 else f"{x:.4f}") for x in v]
            out.append((s if k==0 else "")+f" & {r} & "+" & ".join(cells)+" \\\\")
        if s!="Blocks": out.append("\\midrule")
    open(f"tables/amse{N}.tex","w").write("\n".join(out))
# paired comparisons of OP(3) and OP(2) vs others: wins/ties/losses with |z|>2, and mean % difference
def paired(a,b):
    c=[0,0,0]; rel=[]
    for s in SIGS:
        for N in NS:
            for r in RH:
                L,M,_,_=load(s,N,r); d=L[:,M.index(a)]-L[:,M.index(b)]; z=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
                c[0 if z<-2 else (2 if z>2 else 1)]+=1; rel.append(L[:,M.index(a)].mean()/L[:,M.index(b)].mean()-1)
    rel=np.array(rel); return c,100*rel.mean(),100*rel.min(),100*rel.max()
lines=[]; print("\npaired")
for b in ["SURE-soft","SureShrink","garrote-b2","firm-a3","SCAD-SURE","GRIT-b2","GRIT-b5","GRIT-SURE","GRIT-SUREglob","garrote-SURE","firm-SURE","EB-median"]:
    c,mr,lo,hi=paired("GRIT-fixed",b); print(f"OP3 vs {b:14s} {c}  mean {mr:+.1f}% range [{lo:+.1f},{hi:+.1f}]")
    lines.append(f"{LAB[b]} & {c[0]} & {c[1]} & {c[2]} & ${mr:+.1f}$ & $[{lo:+.1f},{hi:+.1f}]$ \\\\")
open("tables/paired.tex","w").write("\n".join(lines).replace("+","{+}").replace("$-","$-"))
for a,b in [("GRIT-b2","SCAD-SURE"),("GRIT-b2","EB-median"),("firm-a3","SCAD-SURE"),("SCAD-SURE","EB-median"),("GRIT-b2","firm-a3")]:
    print(a,"vs",b,paired(a,b))
# by-signal relative to EB and SCAD (pooled over N, rho)
print("\nby signal mean % diff of OP3 vs EB / SCAD / soft; OP2 vs EB")
for s in SIGS:
    f=lambda a,b: 100*np.mean([A[(s,N,r)][MALL.index(a)]/A[(s,N,r)][MALL.index(b)]-1 for N in NS for r in RH])
    print(s, f"{f('GRIT-fixed','EB-median'):+.1f} {f('GRIT-fixed','SCAD-SURE'):+.1f} {f('GRIT-fixed','SURE-soft'):+.1f} {f('GRIT-b2','EB-median'):+.1f}")
# selection diagnostics
print("\nSURE/oracle-on-grid; selected b")
for N in NS:
    g=[];so=[]
    for s in SIGS:
        for r in RH:
            L,M,a,_=load(s,N,r); g.append(L[:,M.index("GRIT-SURE")].mean()/a["orc_grit"].mean()); so.append(L[:,M.index("SURE-soft")].mean()/a["orc_soft"].mean())
    print(N, f"OP1 {min(g):.2f}-{max(g):.2f} med {np.median(g):.2f} | soft {min(so):.2f}-{max(so):.2f} med {np.median(so):.2f}")
for s in SIGS:
    bl=np.concatenate([load(s,N,r)[2]["selb"] for N in NS for r in RH]); bg=np.concatenate([load(s,N,r)[3]["selg"] for N in NS for r in RH])
    print(s, f"level: P(b=1)={np.mean(bl==1):.2f} P(b>=5)={np.mean(bl>=5):.2f} P(b=8)={np.mean(bl==8):.2f} | global: median {np.median(bg)} P(b>=5)={np.mean(bg>=5):.2f}")

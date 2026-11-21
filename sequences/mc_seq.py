"""Pilot Monte Carlo, sparse normal means: n = 1000, k nonzero means equal to mu0, sigma = 1 known.
SURE-tuned families (threshold and shape on grids), oracle-on-grid tuning, EB posterior median.
Usage: python mc_seq.py R"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "grit"))

import sys, json, time, numpy as np
from sim_core import *
R = int(sys.argv[1]); n = 1000; lams = np.linspace(0.0, np.sqrt(2*np.log(n)), 50)
fams = ["soft", "hybrid", "firm", "scad", "op1", "op2"]
rng = np.random.default_rng(31); out = {}; t0 = time.time()
for mu0 in (3.0, 5.0):
    for k in (10, 50, 200):
        mu = np.zeros(n); mu[:k] = mu0
        L = {f: [] for f in fams}; Lo = {f: [] for f in fams}; L["eb"] = []
        for r in range(R):
            y = mu + rng.standard_normal(n)
            for f in fams:
                d, (lo, _) = sure_select(y, f, lams, mu)
                L[f].append(np.sum((d - mu)**2)); Lo[f].append(lo)
            L["eb"].append(np.sum((eb_postmed(y) - mu)**2))
        key = f"mu{mu0:g}_k{k}"
        out[key] = {f: [float(np.mean(L[f])), float(np.std(L[f])/np.sqrt(R))] for f in L}
        out[key].update({f + "_oracle": float(np.mean(Lo[f])) for f in fams})
        print(key, " ".join(f"{f} {out[key][f][0]:.1f}" for f in L), "| oracle", " ".join(f"{f} {out[key][f+'_oracle']:.1f}" for f in fams), f"[{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open("mc_seq.json", "w"), indent=1)

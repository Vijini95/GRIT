"""Pilot Monte Carlo, wavelet denoising of the Donoho-Johnstone test signals (SNR 7), periodic
Daubechies db4 transform, coarsest level 5, noise scale by MAD of the finest level, level-dependent
SURE tuning of threshold and gate shape.  Benchmarks: SureShrink (soft with the hybrid sparsity test)
and the level-wise EB posterior median (Laplace prior).  Usage: python mc_wav.py R n"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "grit"))

import sys, json, time, numpy as np
from sim_core import *
R, n = int(sys.argv[1]), int(sys.argv[2]); J = int(np.log2(n)); j0 = 5; hh = daubechies(4)
fams = ["soft", "hybrid", "firm", "scad", "op1", "op2"]
rng = np.random.default_rng(7); out = {}; t0 = time.time()


def denoise(y, method):
    a, det = dwt(y, hh, j0); sig = np.median(np.abs(det[J - 1]))/0.6745; new = {}
    for j, d in det.items():
        x = d/sig; nj = len(x); lu = np.sqrt(2*np.log(nj)); lams = np.linspace(0, lu, 40)
        if method == "sureshrink":
            s2 = (np.sum(x**2) - nj)/nj
            if s2 <= np.log2(nj)**1.5/np.sqrt(nj): est = np.sign(x)*np.maximum(np.abs(x) - lu, 0)
            else: est, _ = sure_select(x, "soft", lams)
        elif method == "eb": est = eb_postmed(x)
        else: est, _ = sure_select(x, method, lams)
        new[j] = est*sig
    return idwt(a, new, hh, j0)


for sig_name in ("blocks", "bumps", "heavisine", "doppler"):
    f = dj_signal(sig_name, n); f = 7*f/np.std(f)
    L = {m: [] for m in ["sureshrink", "eb"] + fams}
    for r in range(R):
        y = f + rng.standard_normal(n)
        for m in L: L[m].append(np.mean((denoise(y, m) - f)**2))
    out[sig_name] = {m: [float(np.mean(v)), float(np.std(v)/np.sqrt(R))] for m, v in L.items()}
    print(n, sig_name, " ".join(f"{m} {out[sig_name][m][0]:.4f}" for m in L), f"[{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(f"mc_wav_{n}.json", "w"), indent=1)

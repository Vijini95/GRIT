"""Write the table bodies of the Bayes-risk table (tab:bayes) and the sparse-sequence table (tab:seq)
from bayes_study.json and ../sequences/mc_seq.json."""
import json, pathlib
H = pathlib.Path(__file__).resolve().parent
d = json.load(open(H / "bayes_study.json"))
slab = {"laplace": "Laplace", "t3": "$t_3$", "gauss": "Gaussian", "two4": "two-point"}
rows = []; prev = None
for key, r in d.items():
    k, e = key.split("_"); B = r["bayes"]
    cells = [r[n][0]/B for n in ("soft", "hard", "hybrid", "firm", "scad", "op1", "op")] + [r["postmed"]/B, r["op1_frontier_at_lam"]/B]
    best = min(cells[:7])
    f = [("\\textbf{%.3f}" % x) if abs(x - best) < 5e-4 else "%.3f" % x for x in cells[:7]] + ["%.3f" % x for x in cells[7:]]
    if prev is not None and k != prev: rows.append("\\midrule")
    rows.append(f"{slab[k] if k != prev else ''} & {e} & {B:.4f} & " + " & ".join(f) + r" \\"); prev = k
(H / "tab_bayes_body.tex").write_text("\n".join(rows) + "\n")
s = json.load(open(H.parent / "sequences" / "mc_seq.json"))
fams = ["soft", "hybrid", "firm", "scad", "op1", "op2", "eb"]; rows = []
for key, r in s.items():
    mu0, k = key[2:].split("_k"); v = [r[f][0] for f in fams]; best = min(v)
    cells = [("\\textbf{%.1f}" % x) if abs(x - best) < 0.05 else "%.1f" % x for x in v]
    ratios = [r[f][0]/r[f + "_oracle"] for f in ("soft", "firm", "op1", "op2")]
    rows.append(f"{mu0} & {k} & " + " & ".join(cells) + " & " + " & ".join("%.2f" % x for x in ratios) + r" \\")
(H.parent / "sequences" / "tab_seq_body.tex").write_text("\n".join(rows) + "\n")
print("wrote tab_bayes_body.tex and ../sequences/tab_seq_body.tex")

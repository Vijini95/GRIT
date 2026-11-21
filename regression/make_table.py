"""Generate the GRIT07 regression table and paired checks from saved losses."""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = {
    "lasso": "LASSO (soft)",
    "mcp": "MCP (firm, $a=3$)",
    "scad": "SCAD ($a=3.7$)",
    "garrote": "garrote ($b=2$)",
    "op": "OP ($c=1$, $b=3$)",
}


def standard_error(values):
    return values.std(ddof=1) / np.sqrt(values.size)


def main():
    rows = []
    for rho in ("0.5", "0.8"):
        if rows:
            rows.append(r"\midrule")
        with np.load(HERE / f"res_{rho}.npz") as data:
            for i, (method, label) in enumerate(METHODS.items()):
                loss = data[f"{method}_est"]
                mean_se = f"{loss.mean():.3f} ({standard_error(loss):.3f})"
                # The manuscript omits the LASSO median.
                median = "--" if method == "lasso" else f"{np.median(loss):.3f}"
                metrics = [
                    f"{data[f'{method}_pe'].mean():.3f}",
                    f"{data[f'{method}_FP'].mean():.2f}",
                    f"{data[f'{method}_FN'].mean():.2f}",
                    f"{data[f'{method}_size'].mean():.1f}",
                ]
                rows.append(" & ".join([rho if i == 0 else "", label, mean_se, median, *metrics]) + r" \\")
            op = data["op_est"]
            print(f"rho={rho}: OP median {np.median(op):.3f}")
            for method in ("mcp", "scad"):
                diff = op - data[f"{method}_est"]
                print(f"  OP minus {method.upper()}: {diff.mean():.3f} (paired SE {standard_error(diff):.3f})")
    path = HERE / "tab_reg_body.tex"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote regression/{path.name}")


if __name__ == "__main__":
    main()

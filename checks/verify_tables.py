"""Compare the eight generated numerical table bodies with a GRIT07 TeX file."""
import argparse
from itertools import zip_longest
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "bayes": "bayes_risk/tab_bayes_body.tex",
    "seq": "sequences/tab_seq_body.tex",
    "qn": "wavelets/tables/qn.tex",
    "wav": "wavelets/tables/amse1024.tex",
    "paired": "wavelets/tables/paired.tex",
    "reg": "regression/tab_reg_body.tex",
    "wav512": "wavelets/tables/amse512.tex",
    "wav2048": "wavelets/tables/amse2048.tex",
}


def data_rows(tex):
    """Read printed rows, ignoring headers, boldface and TeX spacing."""
    for line in tex.splitlines():
        if "&" not in line or r"\\" not in line:
            continue
        cells = [
            re.sub(r"\s+|\\textbf|[{}$]", "", cell)
            for cell in line.split(r"\\", 1)[0].split("&")
        ]
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", cells[1]) or cells[0] in {"0.5", "0.8"}:
            yield cells
        elif len(cells) == 8 and re.fullmatch(r"\d+\.\d+\(\d+\.\d+\)", cells[2]):
            yield cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manuscript", type=Path, help="path to the GRIT07 TeX manuscript")
    args = parser.parse_args()
    tex = args.manuscript.read_text(encoding="utf-8")
    failed = False
    for label, path in TABLES.items():
        match = re.search(r"\\label\{tab:" + label + r"\}(.*?)\\end\{table\}", tex, re.S)
        if match is None:
            print(f"tab:{label}: table label missing")
            failed = True
            continue
        expected = list(data_rows(match.group(1)))
        observed = list(data_rows((ROOT / path).read_text(encoding="utf-8")))
        if expected and expected == observed:
            print(f"tab:{label}: {len(expected)} rows match")
            continue
        failed = True
        for i, (paper, generated) in enumerate(zip_longest(expected, observed), start=1):
            if paper != generated:
                print(f"tab:{label} row {i}\n  manuscript: {paper}\n  regenerated: {generated}")
        if not expected and not observed:
            print(f"tab:{label}: no data rows found")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# Regenerate the paper's numerical tables and figures from the saved results.
set -euo pipefail
cd "$(dirname "$0")"
python_cmd="${PYTHON:-python3}"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--tables | --full]" >&2
  exit 2
fi
case "${1:-}" in
  --full)
    (cd bayes_risk && "$python_cmd" bayes_study.py && "$python_cmd" op1_study.py)
    (cd sequences && "$python_cmd" mc_seq.py 200)
    (cd wavelets &&
      xargs -P "${JOBS:-2}" -L 1 "$python_cmd" study.py < jobs.txt &&
      xargs -P "${JOBS:-2}" -L 1 "$python_cmd" study2.py < jobs.txt)
    (cd regression && "$python_cmd" regsim.py 0.5 && "$python_cmd" regsim.py 0.8)
    ;;
  --tables|"") ;;
  *) echo "Usage: $0 [--tables | --full]" >&2; exit 2 ;;
esac

"$python_cmd" bayes_risk/make_tables.py
(cd wavelets && "$python_cmd" analyze2.py > analysis2.txt)
"$python_cmd" regression/make_table.py
if [[ "${1:-}" != "--tables" ]]; then
  "$python_cmd" figures/make_figs.py
  "$python_cmd" checks/oracle_check.py
fi

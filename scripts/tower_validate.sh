#!/usr/bin/env bash
# Run the external-bench validation for several configs and print ONLY aggregate metrics.
#
#   bash scripts/tower_validate.sh                      # rules_v0 hsmm_v1 hsmm_v1_rules (full)
#   bash scripts/tower_validate.sh --quick rules_v0     # quick run of one config
#
# Refuses to run on a dirty tree (the leaderboard row would say +dirty). Needs the data, so it
# belongs on the tower. Appends rows to results/leaderboard.csv; commit that file afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."

extra=()
if [ "${1:-}" = "--quick" ]; then extra=(--quick); shift; fi
configs=("$@")
[ ${#configs[@]} -gt 0 ] || configs=(rules_v0 hsmm_v1 hsmm_v1_rules)

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "dirty tree: commit first (results/leaderboard.csv rows must not say +dirty)" >&2
  exit 1
fi

for c in "${configs[@]}"; do
  [ -f "configs/$c.yaml" ] || { echo "missing configs/$c.yaml" >&2; exit 1; }
done

for c in "${configs[@]}"; do
  echo "== $c"
  python scripts/run_validation.py --config "configs/$c.yaml" "${extra[@]}" 2>&1 | tail -n 25
done

python - <<'PY'
import pandas as pd
d = pd.read_csv("results/leaderboard.csv")
cols = ["config", "git", "external_spell", "external_spell_lo", "external_spell_hi",
        "external_macro", "external_poss", "external_gt_poss", "external_matches", "full_poss"]
print(d[cols].tail(6).to_string(index=False))
PY

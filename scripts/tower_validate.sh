#!/usr/bin/env bash
# Run the external-bench validation for several configs and print ONLY aggregate metrics.
#
#   bash scripts/tower_validate.sh                      # rules_v0 hsmm_v1 hsmm_v1_rules (full)
#   bash scripts/tower_validate.sh --quick rules_v0     # quick run of one config
#
# Refuses to run on a dirty tree (the leaderboard row would say +dirty). Needs the data, so it
# belongs on the tower. Commits results/leaderboard.csv after each config (local commit, no push)
# so the next row is clean too. Run it inside the project's Python 3.11 venv.
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

# harness.run reads the git state per config, and each run appends to the tracked
# results/leaderboard.csv: without a commit in between, every config after the first is +dirty.
for c in "${configs[@]}"; do
  echo "== $c"
  python scripts/run_validation.py --config "configs/$c.yaml" "${extra[@]}" 2>&1 | tail -n 25
  if [ -n "$(git status --porcelain --untracked-files=no -- results/leaderboard.csv)" ]; then
    git add results/leaderboard.csv
    git commit -q -m "leaderboard: $c${extra:+ (quick)}"
  fi
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "tree dirty after $c (something besides the leaderboard changed): stopping" >&2
    exit 1
  fi
done

python - <<'PY'
import pandas as pd
d = pd.read_csv("results/leaderboard.csv")
cols = ["config", "git", "external_spell", "external_spell_lo", "external_spell_hi",
        "external_macro", "external_poss", "external_gt_poss", "external_matches", "full_poss"]
print(d[cols].tail(6).to_string(index=False))
PY

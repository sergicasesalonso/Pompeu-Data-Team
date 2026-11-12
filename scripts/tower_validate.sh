#!/usr/bin/env bash
# Run the external-bench validation for several configs and print ONLY aggregate metrics.
#
#   bash scripts/tower_validate.sh                      # rules_v0 hsmm_v1 hsmm_v1_rules (full)
#   bash scripts/tower_validate.sh --quick rules_v0     # quick run: prints, records nothing
#
# Needs the data, so it belongs on the tower; run it inside the project's Python 3.11 venv.
# Full runs: refuses a dirty tree and configs that are not committed (a row must be reproducible
# from its git hash), and commits results/leaderboard.csv after each config, because
# harness.run reads the git state per config and the appended row would make the next one
# +dirty. Those are LOCAL commits: bring them to `marti` with git (push or pull), never by
# copying the csv, or the rows point at hashes that exist nowhere else.
# Quick runs pass --no-save: CLAUDE.md says --quick is for iterating, never for deciding.
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
if [ "${1:-}" = "--quick" ]; then quick=1; shift; fi
configs=("$@")
[ ${#configs[@]} -gt 0 ] || configs=(rules_v0 hsmm_v1 hsmm_v1_rules)

for c in "${configs[@]}"; do
  [ -f "configs/$c.yaml" ] || { echo "missing configs/$c.yaml" >&2; exit 1; }
done

if [ "$quick" = 1 ]; then
  for c in "${configs[@]}"; do
    echo "== $c (quick, not recorded)"
    python scripts/run_validation.py --config "configs/$c.yaml" --quick --no-save 2>&1 | tail -n 25
  done
  exit 0
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "dirty tree: commit first (results/leaderboard.csv rows must not say +dirty)" >&2
  exit 1
fi
for c in "${configs[@]}"; do
  git ls-files --error-unmatch "configs/$c.yaml" >/dev/null 2>&1 \
    || { echo "configs/$c.yaml is not committed: commit it first" >&2; exit 1; }
done
git config user.name >/dev/null || { echo "git user.name is not set" >&2; exit 1; }

for c in "${configs[@]}"; do
  echo "== $c"
  python scripts/run_validation.py --config "configs/$c.yaml" 2>&1 | tail -n 25
  if [ -n "$(git status --porcelain --untracked-files=no -- results/leaderboard.csv)" ]; then
    git add results/leaderboard.csv
    git commit -q -m "leaderboard: $c"
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
        "external_macro", "external_poss", "external_gt_poss", "external_matches",
        "external_seeds", "full_poss"]
print(d[cols].tail(6).to_string(index=False))
PY

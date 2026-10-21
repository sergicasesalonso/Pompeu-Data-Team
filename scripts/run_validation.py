"""Run the four-bench validation harness on one or more configs and record them in the leaderboard.

    python scripts/run_validation.py --config configs/baseline_smooth50.yaml
    python scripts/run_validation.py --config configs/baseline_*.yaml
    python scripts/run_validation.py --config configs/x.yaml --quick     # 5 seeds, 200 draws
    python scripts/run_validation.py --config configs/x.yaml --no-save   # print, record nothing

Each run appends a row to results/leaderboard.csv (git hash and config hash included), writes the
full detail to results/runs/ and the full-match file to results/submissions/<name>.csv. See
pompeu/validate/harness.py for what each bench means and the acceptance rule.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.validate import harness  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", nargs="+", required=True, help="YAML config(s); globs work")
    parser.add_argument("--bootstrap", type=int, help="override validation.bootstrap_n")
    parser.add_argument("--seeds", type=int, help="override validation.stress.seeds")
    parser.add_argument("--quick", action="store_true",
                        help="5 seeds and 200 bootstrap draws, for iterating; not for deciding")
    parser.add_argument("--no-save", action="store_true", help="do not touch results/")
    args = parser.parse_args()

    paths = sorted({p for pattern in args.config for p in (glob.glob(pattern) or [pattern])})
    trackings = {}
    for path in paths:
        cfg = harness.load_config(path)
        if args.quick:
            cfg["validation"]["stress"]["seeds"] = 5
            cfg["validation"]["bootstrap_n"] = 200
        if args.seeds is not None:
            cfg["validation"]["stress"]["seeds"] = args.seeds
        if args.bootstrap is not None:
            cfg["validation"]["bootstrap_n"] = args.bootstrap

        # Configs on the same tracking share one load.
        if cfg["tracking"] not in trackings:
            trackings[cfg["tracking"]] = load_tracking(cfg["tracking"])
        result, _ = harness.run(cfg, trackings[cfg["tracking"]], save=not args.no_save,
                                verbose=False)
        print(harness.format_result(result))
        print()

    if not args.no_save:
        print("recorded in {}".format(os.path.relpath(harness.LEADERBOARD)))


if __name__ == "__main__":
    main()

"""Regenerate a submission file from the data and a config. Deterministic: same inputs, same bytes.

    python scripts/make_submission.py --config configs/baseline_smooth50.yaml
    python scripts/make_submission.py --config configs/final.yaml --out PompeuDataTeam.csv

The file is checked with the organisers' own validation (possession_metrics.validate_submission,
what evaluation/validate_submission.py and the official scorer run) and its sha256 printed, so two
machines can confirm they produced the same file.
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import pipeline  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.validate import harness  # noqa: E402
import possession_metrics as pm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", help="default results/submissions/<config name>.csv")
    args = parser.parse_args()

    cfg = harness.load_config(args.config)
    out = args.out or os.path.join(harness.SUBMISSIONS_DIR, "{}.csv".format(cfg["name"]))
    labels = pipeline.predict(load_tracking(cfg["tracking"]), cfg["pipeline"])

    folder = os.path.dirname(os.path.abspath(out))
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(
        out, index=False, lineterminator="\n")

    table = pm.read_label_csv(out)
    reason = pm.validate_submission(table, len(labels))
    if reason:
        sys.exit("FAIL {}: {}".format(out, reason))
    with open(out, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print("OK   {}  ({} rows, config {} {})".format(out, len(table), cfg["name"],
                                                     pipeline.config_hash(cfg)))
    print("     sha256 {}".format(digest))


if __name__ == "__main__":
    main()

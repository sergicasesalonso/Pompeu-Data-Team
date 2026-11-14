"""Run make_submission for a config in a temp folder and report only the sha256 and label counts.

    python scripts/experiments/hsmm_tcn_sha.py --config configs/hsmm_tcn.yaml

For the tower (torre.py): the submission file is written to a temp directory and deleted; only
the hash and the counts leave (TORRE_METRICS if set).
"""

import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    folder = tempfile.mkdtemp(prefix="hsmm_tcn_sha_")
    out = os.path.join(folder, "sub.csv")
    try:
        proc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_submission.py"),
                               "--config", args.config, "--out", out], cwd=ROOT,
                              capture_output=True, text=True)
        if proc.returncode:
            # Only the exception type and a short, path-free message leave (never the log).
            lines = [x for x in proc.stderr.splitlines() if x.strip()]
            last = lines[-1] if lines else ""
            kind, _, msg = last.partition(":")
            failure = {"failed": True, "error_type": kind.strip()[:60],
                       "error_msg": "".join(c if c.isalnum() or c in " _.-" else "~"
                                            for c in msg.strip()[:120])}
            print(json.dumps(failure))
            if os.environ.get("TORRE_METRICS"):
                with open(os.environ["TORRE_METRICS"], "w", encoding="utf-8") as handle:
                    json.dump(failure, handle)
            sys.exit("make_submission failed (exit {})".format(proc.returncode))
        with open(out, "rb") as handle:
            data = handle.read()
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    counts = collections.Counter(line.split(b",")[1].decode() for line in data.splitlines()[1:])
    metrics = {"sha256": hashlib.sha256(data).hexdigest(), "n_frames": sum(counts.values()),
               "n_H": counts["H"], "n_A": counts["A"], "n_D": counts["D"], "n_X": counts["X"],
               "metode": "hsmm_tcn", "config": os.path.basename(args.config)}
    print(json.dumps(metrics))
    if os.environ.get("TORRE_METRICS"):
        with open(os.environ["TORRE_METRICS"], "w", encoding="utf-8") as handle:
            json.dump(metrics, handle)


if __name__ == "__main__":
    main()

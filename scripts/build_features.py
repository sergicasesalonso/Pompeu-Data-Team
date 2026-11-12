"""Build the A2 feature matrix for the full match, cache it as parquet, and write its catalogue.

    python scripts/build_features.py                                 # cleaning with defaults
    python scripts/build_features.py --config configs/x.yaml         # that config's tracking + clean
    python scripts/build_features.py --force                         # rebuild even if cached

The matrix goes to cache/features_v<version>_<key>.parquet (gitignored: it is derived from the
match data). The key hashes the tracking spec, the cleaning settings and FEATURES_VERSION, so a
changed config never reads a stale matrix. The catalogue -- column names, group and NaN share,
nothing that reconstructs the data -- goes to results/features_report.md.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import paths  # noqa: E402
from pompeu.features import build  # noqa: E402
from pompeu.validate import harness  # noqa: E402

REPORT = os.path.join(paths.RESULTS_DIR, "features_report.md")


def write_report(frame, path, out):
    graded = frame["half"].to_numpy() > 0
    groups = frame.attrs.get("groups", {})
    lines = ["# Rasgos por frame (A2)", "",
             "`python scripts/build_features.py` — {} columnas, {:,} frames. Matriz en `{}` "
             "(no se sube a git). % NaN medido sobre los frames puntuados.".format(
                 frame.shape[1], frame.shape[0], os.path.relpath(path, paths.ROOT)), ""]
    order = ("ball", "proximity", "pitch_control", "collective", "context", "windows")
    for group in order:
        names = [c for c in frame.columns if groups.get(c) == group]
        if not names:
            continue
        lines += ["## {} ({} columnas)".format(group, len(names)), "",
                  "| columna | % NaN |", "|---|---|"]
        for name in names:
            share = 100.0 * float(np.mean(np.isnan(frame[name].to_numpy()[graded])))
            lines.append("| `{}` | {:.1f} |".format(name, share))
        lines.append("")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="take tracking and pipeline.clean from this config")
    parser.add_argument("--no-clean", action="store_true", help="features on the raw tracking")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", default=REPORT)
    args = parser.parse_args()

    tracking_spec, clean_cfg = "atd", {}
    if args.config:
        cfg = harness.load_config(args.config)
        tracking_spec, clean_cfg = cfg["tracking"], cfg["pipeline"].get("clean")
    if args.no_clean:
        clean_cfg = None

    started = time.time()
    frame, path = build.load_or_build(tracking_spec, clean_cfg, force=args.force)
    write_report(frame, path, args.report)
    print("{} x {}  ->  {}  ({:.1f} s total)".format(frame.shape[0], frame.shape[1], path,
                                                       time.time() - started))
    print("catalogue -> {}".format(args.report))


if __name__ == "__main__":
    main()

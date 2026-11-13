"""B1: build the ball pseudo-labels for the detector into cache/. Prints counts only.

    python scripts/build_pseudolabels.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.io import homography  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.vision import decode, pseudolabels  # noqa: E402
import games  # noqa: E402


def main():
    info = decode.probe(games.asset_paths()["video"], count_packets=False)
    tracking, _ = clean_tracking(load_tracking("atd"), {})
    H, err, valid = homography.load()
    table, report = pseudolabels.build(tracking, H, err, valid, info["width"], info["height"])
    path = pseudolabels.save(table)
    for key, value in report.items():
        print("{:<28} {}".format(key, value))
    print("written to cache/ ({})".format(os.path.basename(path)))


if __name__ == "__main__":
    main()

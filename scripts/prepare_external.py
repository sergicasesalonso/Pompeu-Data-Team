"""Download IDSSE (DFL) matches and convert them for the external bench (stage A6).

    python scripts/prepare_external.py                      # J03WMX and J03WPY, as validated
    python scripts/prepare_external.py --matches all        # all seven
    python scripts/prepare_external.py --matches J03WN1 J03WOH

Bassek, Weber, Rein & Memmert (2025), Scientific Data 12, 195, CC-BY 4.0. The official copy is
figshare doi:10.6084/m9.figshare.28196177, which answers scripted downloads with a bot challenge,
so the identical files come from the Hugging Face mirror pysport/idsse-data.

Files land in data/external/idsse/ and the converted matches in cache/external/: both gitignored.
The licensed data never enters this public repository; this script is how anyone rebuilds it.
"""

import argparse
import os
import shutil
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu.external import dfl  # noqa: E402

MIRROR = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/{}"
# Competition id in the file names: 000001 Bundesliga, 000002 2. Bundesliga.
COMPETITION = {"J03WMX": "000001", "J03WN1": "000001", "J03WPY": "000002", "J03WOH": "000002",
               "J03WQQ": "000002", "J03WOY": "000002", "J03WR9": "000002"}
DEFAULT = ("J03WMX", "J03WPY")


def file_names(match_id):
    competition = COMPETITION[match_id]
    return ["DFL_02_01_matchinformation_DFL-COM-{}_DFL-MAT-{}.xml".format(competition, match_id),
            "DFL_04_03_positions_raw_observed_DFL-COM-{}_DFL-MAT-{}.xml".format(competition,
                                                                               match_id)]


def download(name, folder):
    """One file, written aside and renamed so an interrupted download never looks complete."""
    target = os.path.join(folder, name)
    if os.path.isfile(target):
        return target
    request = urllib.request.Request(MIRROR.format(name), headers={"User-Agent": "pompeu/0.1"})
    with urllib.request.urlopen(request) as response, open(target + ".part", "wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    os.replace(target + ".part", target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="+", default=list(DEFAULT),
                        help="IDSSE match ids, or 'all'")
    args = parser.parse_args()
    matches = list(dfl.MATCHES) if args.matches == ["all"] else args.matches
    unknown = sorted(set(matches) - set(dfl.MATCHES))
    if unknown:
        sys.exit("unknown match ids {}; have {}".format(unknown, sorted(dfl.MATCHES)))

    os.makedirs(dfl.ROOT, exist_ok=True)
    for match_id in matches:
        started = time.time()
        for name in file_names(match_id):
            download(name, dfl.ROOT)
        tracking, labels = dfl.load(match_id)
        print("{}  {}  {} frames, halves {}  ({:.0f} s)".format(
            match_id, dfl.MATCHES[match_id], tracking.n_frames, tracking.meta["halves"],
            time.time() - started))


if __name__ == "__main__":
    main()

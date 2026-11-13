"""Fetch SkillCorner Open Data (A-League 2024/25, MIT licence) for the TCN's extra training matches.

    python scripts/prepare_skillcorner.py

Shallow clone of github.com/SkillCorner/opendata into data/external/skillcorner/opendata (the
folder pompeu.external.skillcorner reads), gitignored. Public data, training only (never val or
test): credit SkillCorner when it is used. Prints only counts.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import external  # noqa: E402
from pompeu.external import skillcorner  # noqa: E402

URL = "https://github.com/SkillCorner/opendata.git"


def main():
    target = os.path.dirname(os.path.dirname(skillcorner.ROOT))          # .../opendata
    if not os.path.isdir(os.path.join(target, ".git")):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", URL, target], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ids = [m for m in external.EXTRA_TRAIN if external.available(m)]
    print("skillcorner matches present: {} of {}".format(len(ids), len(external.EXTRA_TRAIN)))


if __name__ == "__main__":
    main()

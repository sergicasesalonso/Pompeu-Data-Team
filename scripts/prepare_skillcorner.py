"""Fetch SkillCorner Open Data (A-League 2024/25, MIT licence) for the TCN's extra training matches.

    python scripts/prepare_skillcorner.py

Shallow clone of github.com/SkillCorner/opendata into data/external/skillcorner/opendata (the
folder pompeu.external.skillcorner reads), gitignored. Public data, training only (never val or
test): credit SkillCorner when it is used. Prints only counts.
"""

import os
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import external  # noqa: E402
from pompeu.external import skillcorner  # noqa: E402

URL = "https://github.com/SkillCorner/opendata.git"
MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/{}/{}"


def _tracking_files():
    out = []
    for folder, _, names in os.walk(skillcorner.ROOT):
        out += [os.path.join(folder, n) for n in names if "tracking" in n]
    return out


def _pointers(target):
    """Files in the clone that are git-LFS pointers instead of data."""
    out = []
    for folder, dirs, names in os.walk(target):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in names:
            path = os.path.join(folder, name)
            if os.path.getsize(path) < 1024:
                with open(path, "rb") as handle:
                    if handle.read(40).startswith(b"version https://git-lfs"):
                        out.append(path)
    return out


def main():
    target = os.path.dirname(skillcorner.ROOT)                              # .../opendata
    if not os.path.isdir(os.path.join(target, ".git")):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", URL, target], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # The data files may be git-LFS pointers (a few hundred bytes of text) when git-lfs is not
    # installed: fetch the real file from GitHub's media endpoint (plain HTTPS), same repo, same data.
    branch = subprocess.run(["git", "-C", target, "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "master"
    for path in _pointers(target):
        rel = os.path.relpath(path, target).replace(os.sep, "/")
        url = MEDIA.format(branch, rel)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pompeu/0.1"})
                                    ) as response, open(path + ".part", "wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
        os.replace(path + ".part", path)
    small = _pointers(target)
    print("tracking files: {}, still LFS pointers: {}".format(len(_tracking_files()), len(small)))
    ids = [m for m in external.EXTRA_TRAIN if external.available(m)]
    print("skillcorner matches present: {} of {}".format(len(ids), len(external.EXTRA_TRAIN)))
    if os.environ.get("TORRE_METRICS"):
        import json
        with open(os.environ["TORRE_METRICS"], "w", encoding="utf-8") as handle:
            json.dump({"present": len(ids), "tracking_files": len(_tracking_files()),
                       "pointers": len(small)}, handle)


if __name__ == "__main__":
    main()

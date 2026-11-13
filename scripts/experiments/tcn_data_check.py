"""Availability check for the TCN recipe (counts only): which external matches are present, per
split and for the SkillCorner extra_train set. Writes TORRE_METRICS if set."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pompeu import external, paths  # noqa: E402


def have(m):
    return external.available(m) or os.path.isfile(os.path.join(
        paths.CACHE_DIR, "external", m.replace(":", "_") + "_v1.npz"))


metrics = {}
for split in ("train", "val", "test"):
    names = [external.canonical(m) for m in external.SPLITS[split]]
    metrics[split + "_present"], metrics[split + "_total"] = sum(map(have, names)), len(names)
extra = [external.canonical(m) for m in external.EXTRA_TRAIN]
metrics["extra_present"], metrics["extra_total"] = sum(map(have, extra)), len(extra)
metrics["tcn_artifact"] = int(os.path.isfile(os.path.join(paths.ROOT, "artifacts", "tcn_v1.pt")))
print(json.dumps(metrics))
if os.environ.get("TORRE_METRICS"):
    with open(os.environ["TORRE_METRICS"], "w", encoding="utf-8") as handle:
        json.dump(metrics, handle)

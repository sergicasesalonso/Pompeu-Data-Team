"""Repository locations, and the one place that makes the organisers' modules importable.

games.py sits at the repository root and possession_metrics.py in evaluation/, neither inside a
package. Putting both directories on sys.path here, once, means every module in pompeu can write
`import games` and `import possession_metrics as pm` and get the organisers' exact code -- the
metric is reused, never reimplemented.

Derived data (the parsed-tracking cache) goes to CACHE_DIR. It is gitignored, and it can be moved
off a synced folder with POMPEU_CACHE_DIR, the same way MTAH_DATA_DIR moves the data bundle.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALUATION_DIR = os.path.join(ROOT, "evaluation")
GT_DIR = os.path.join(EVALUATION_DIR, "gt")
CONFIGS_DIR = os.path.join(ROOT, "configs")
RESULTS_DIR = os.path.join(ROOT, "results")
CACHE_DIR = os.environ.get("POMPEU_CACHE_DIR") or os.path.join(ROOT, "cache")

for _path in (ROOT, EVALUATION_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

PUBLIC_GT = os.path.join(GT_DIR, "BAR-ATH_possession_gt_public.csv")

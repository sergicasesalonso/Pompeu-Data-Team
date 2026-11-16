"""Report which key packages import in the current venv (booleans only, to TORRE_METRICS)."""

import importlib
import json
import os

result = {}
for name in ("torch", "lightgbm", "numpy", "numba", "pandas", "yaml"):
    try:
        importlib.import_module(name)
        result[name] = 1
    except Exception:  # noqa: BLE001
        result[name] = 0
print(json.dumps(result))
if os.environ.get("TORRE_METRICS"):
    with open(os.environ["TORRE_METRICS"], "w", encoding="utf-8") as handle:
        json.dump(result, handle)

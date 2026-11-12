"""Training examples from the external matches: degraded, cleaned and featurised like BAR-ATH.

    from pompeu.external import examples
    F, y = examples.load("dfl:J03WOH", seed=0)       # features (A2) and aligned labels, per frame
    X, y, groups = examples.matrix(external.SPLITS["train"], seeds=range(4))

Every learned model of the plan (A3's live/dead classifier, A5's expert weights, A7's sequence
model) trains on these and nothing else. For a (match, seed):

    1. degrade the clean match like our ATD (pompeu.external.degrade, that seed): every seed is a
       different degradation of the same match, so seeds multiply the effective data;
    2. clean it with A1's defaults, exactly as the pipeline cleans BAR-ATH;
    3. build A2's features (all groups, windows included);
    4. labels: the source's own, convention-aligned (align_labels, 5 s), X outside the halves.

Cached as parquet under cache/external/examples/ (gitignored, derived from licensed data).

MODEL_COLUMNS drops what would not mean the same thing on BAR-ATH: the Smart Tagging columns are
NaN on every external match and real on BAR-ATH, so a model that learned them as "always missing"
would meet unseen values at prediction time. Tagging enters the decision through the HSMM's own
tagging expert instead.
"""

import os

import numpy as np
import pandas as pd

from pompeu import paths

# Params
VERSION = 1
MIN_POSSESSION_S = 5.0
CACHE = os.path.join(paths.CACHE_DIR, "external", "examples")
TAGGING_PREFIXES = ("st_", "sp_")
TAGGING_COLUMNS = ("goal_active",)


def _path(name, seed):
    safe = name.replace(":", "_")
    return os.path.join(CACHE, "{}_s{}_v{}.parquet".format(safe, seed, VERSION))


def model_columns(columns):
    """The feature columns a learned model may use (everything but Smart Tagging)."""
    return [c for c in columns
            if not c.startswith(TAGGING_PREFIXES) and c not in TAGGING_COLUMNS]


def build(name, seed, run_lengths=None):
    """(F, y): features of the degraded, cleaned match and its aligned labels (int8, X = 3)."""
    from pompeu import external
    from pompeu.clean import clean_tracking
    from pompeu.external import degrade
    from pompeu.features.build import build_features

    if run_lengths is None:
        from pompeu.io.contract import load_tracking
        from pompeu.validate.stress import missing_run_lengths
        run_lengths = missing_run_lengths(load_tracking("atd"))
    tracking, labels = external.load(name)
    graded = np.where(tracking.half() > 0, labels, 3).astype(np.int8)
    y = external.align_labels(graded, MIN_POSSESSION_S, tracking.meta.get("fps", 25))
    degraded = degrade.degrade(tracking, labels, seed, run_lengths)
    clean, _ = clean_tracking(degraded, {})
    F = build_features(clean, tagging=None)
    return F, y


def load(name, seed, run_lengths=None, cache=True):
    """(F, y) for one (match, seed), from the cache once built."""
    path = _path(name, seed)
    if cache and os.path.isfile(path):
        frame = pd.read_parquet(path, engine="pyarrow")
        y = frame.pop("__label__").to_numpy(dtype=np.int8)
        return frame, y
    F, y = build(name, seed, run_lengths)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        out = F.copy()
        out["__label__"] = y
        out.to_parquet(path + ".part", engine="pyarrow")
        os.replace(path + ".part", path)
    return F, y


def matrix(names, seeds, step=1, run_lengths=None):
    """(X DataFrame of model_columns, y (n,), groups (n,) 'match|seed') over graded frames, every
    `step`-th frame (consecutive frames are near-duplicates; thinning costs little)."""
    parts, labels, groups = [], [], []
    for name in names:
        for seed in seeds:
            F, y = load(name, seed, run_lengths)
            keep = np.flatnonzero(y != 3)[::step]
            parts.append(F.iloc[keep][model_columns(F.columns)])
            labels.append(y[keep])
            groups.append(np.full(len(keep), "{}|{}".format(name, seed)))
    return (pd.concat(parts, ignore_index=True), np.concatenate(labels),
            np.concatenate(groups))

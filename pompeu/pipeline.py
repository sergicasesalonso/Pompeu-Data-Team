"""Config -> labels. The one function the harness and make_submission call.

A config names a method and its parameters:

    pipeline:
      method: baseline_nearest
      params: {window: 50}

and every method is a function (tracking, **params) -> (N,) array of H/A/D/X labels, one per row
of the Tracking it was given. Methods never see a ground truth, and never assume the Tracking is a
full match: the stress bench hands them a cropped window (tracking.frame0 > 0).

New methods register themselves in METHODS -- rules_v0 (A4), hsmm_v1 (A5) and so on.
"""

import hashlib
import json

import numpy as np

from pompeu.models import baseline

METHODS = {
    "baseline_nearest": baseline.predict,
}


def predict(tracking, pipeline_cfg):
    method = pipeline_cfg["method"]
    if method not in METHODS:
        raise KeyError("unknown pipeline method {!r}; have {}".format(method, sorted(METHODS)))
    labels = np.asarray(METHODS[method](tracking, **(pipeline_cfg.get("params") or {})))
    if labels.shape != (tracking.n_frames,):
        raise ValueError("{} returned shape {}, expected ({},)".format(
            method, labels.shape, tracking.n_frames))
    return labels.astype("<U1")


def config_hash(cfg):
    """Short stable hash of a config: same content, same hash, whatever the key order."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]

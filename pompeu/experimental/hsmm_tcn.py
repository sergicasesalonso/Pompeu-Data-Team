"""Method `hsmm_tcn`: the HSMM with the dilated TCN (pompeu.experimental.tcn) as one more expert.

Same decoder and weights as hsmm_v1_fit_mbr; the TCN's per-frame team log-odds
(log P(H) - log P(A), weight `w_team`) and dead log-odds (log P(D) - log(P(H) + P(A)), weight
`w_dead`) enter through the HSMM's learned-expert slot, exactly as scripts/experiments/tcn_val.py
validated it (HSMM+TCN r2, 1/2, MBR).

TAGS. The TCN reads Smart Tagging context columns. They were trained on synthetic tags, which
only have possession phases and four set-piece types, so a real tagging is restricted with
tcn.like_synthetic (zone / goal / kick-off codes are out of distribution). A match with no
tagging (external matches) gets an empty table, as in the "no tags" ablation. The HSMM itself
runs WITHOUT tags (w_tagging 0, tagging injected as None), the condition that was validated.

The pipeline cleans the tracking before calling this (config `clean`).

Params: `hsmm` / `deadball` / `touches` as for hsmm_v1; `tcn` = {model: path, sha: expected
weights hash (tcn.state_hash, 16 hex; checked when given), w_team, w_dead}.
"""

import os

import numpy as np

from pompeu import paths
from pompeu.experimental import tcn

ARTIFACT = os.path.join(paths.ROOT, "artifacts", "tcn_v1.pt")
DEFAULTS = {"model": ARTIFACT, "sha": None, "w_team": 1.0, "w_dead": 2.0}


def tcn_log_probs(clean, F, model_path, sha=None):
    """(N, 3) per-frame log P(H), P(A), P(D) from the TCN with the like-synthetic real tags."""
    from pompeu.features.build import load_tagging_for
    model, norm = tcn.load(model_path)
    if sha is not None and tcn.state_hash(model) != sha:
        raise ValueError("TCN weights hash {} != expected {} ({})".format(
            tcn.state_hash(model), sha, model_path))
    tags = load_tagging_for(clean)
    tags = tcn.empty_tagging() if tags is None else tcn.like_synthetic(tags)
    X, names = tcn.inputs_from(clean, tags, F)
    return tcn.predict_frames(model, norm, X, names, clean.n_frames)


def decode(clean, hsmm=None, deadball=None, touches=None, tcn_params=None):
    """hsmm.decode_prepared result with the TCN expert filled in."""
    from pompeu.external import synthetic_tags as st
    from pompeu.features.build import build_features
    from pompeu.models import hsmm
    t = dict(DEFAULTS, **(tcn_params or {}))
    if not os.path.isabs(t["model"]):
        t["model"] = os.path.join(paths.ROOT, t["model"])
    game = clean.meta.get("game")
    with st.injected({game: None}):                     # no tagging reaches the HSMM
        prep = hsmm.prepare(clean, deadball or None, touches or None)
    F = build_features(clean, tagging=None)
    log_p = tcn_log_probs(clean, F, t["model"], t["sha"])
    prep["learned"] = {"team": log_p[:, 0] - log_p[:, 1],
                       "dead": log_p[:, 2] - np.logaddexp(log_p[:, 0], log_p[:, 1])}
    params = dict(hsmm or {}, w_tagging=0.0, w_learned_team=float(t["w_team"]),
                  w_learned_dead=float(t["w_dead"]))
    return hsmm.decode_prepared(prep, params)


def predict(tracking, hsmm=None, deadball=None, touches=None, tcn=None):
    """(N,) H/A/D labels: the pipeline entry point (method hsmm_tcn)."""
    return decode(tracking, hsmm, deadball, touches, tcn)["labels"]

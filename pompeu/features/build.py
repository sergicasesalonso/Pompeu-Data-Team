"""A2: the per-frame feature matrix F[frame, d], one row per row of the Tracking.

    from pompeu.features.build import build_features
    F = build_features(clean_tracking)          # pandas DataFrame, float32, index = video frame

Groups, each in its own module:

    kinematics.ball_features       ball speed, acceleration, turn, stillness, out, lines, gaps
    proximity.proximity_features   nearest players per team, counts within 2/5/10 m, streaks
    pitch_control                  Spearman control at the ball and on a 5 m ring
    collective                     team shape and motion; defined with or without the ball
    context                        clock, Smart Tagging votes and set pieces
    windows                        0.5 / 2 / 5 s centred mean, min, max of the key ones

WITH AND WITHOUT BALL. Every ball-dependent column is NaN where there is no ball, and
`ball_present` plus the gap timing (`ball_gap_age_s`, `ball_gap_left_s`, `ball_gap_len_s`) say so
explicitly; `ball_x_home_last` keeps the last known position. The collective and context groups
are defined on every frame. Gradient boosting and the sequence model take NaN natively, so nothing
is imputed here -- imputing would hide the very fact that the ball is missing.

The matrix is derived from the match data: it is cached under cache/ and never committed.
"""

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from pompeu import paths
from pompeu.clean.normalize import attack_flip
from pompeu.features import collective, context, kinematics, pitch_control, proximity, windows
import games

# Params
# Bump when any feature's definition changes, so cached matrices are rebuilt.
FEATURES_VERSION = 1


def load_tagging_for(tracking):
    """Smart Tagging for this match, or None when there is none (external datasets)."""
    from pompeu.io.tagging import load_tagging
    try:
        games.require(["smart_tagging"], tracking.meta.get("game", games.GAME))
    except (IOError, KeyError):
        return None
    return load_tagging(tracking.meta["team_names"], tracking.meta.get("game", games.GAME))


GROUPS = ("ball", "proximity", "pitch_control", "collective", "context")


def build_features(tracking, tagging="auto", with_windows=True, groups=GROUPS):
    """DataFrame (N, d) float32 indexed by video frame. tagging: DataFrame, None, or "auto".

    `groups` picks a subset (a detector that needs no pitch control skips its cost); the
    windows are only built when every windowed column is available.
    """
    if isinstance(tagging, str) and tagging == "auto":
        tagging = load_tagging_for(tracking) if "context" in groups else None
    fps = tracking.meta.get("fps", 25)
    players = kinematics.player_kinematics(tracking)
    ball = kinematics.ball_kinematics(tracking)
    makers = {
        "ball": lambda: kinematics.ball_features(tracking, ball, attack_flip(tracking, 0)),
        "proximity": lambda: proximity.proximity_features(tracking, players["speed"]),
        "pitch_control": lambda: pitch_control.pitch_control_features(tracking, players["vel"]),
        "collective": lambda: collective.collective_features(tracking, players["vel"],
                                                             players["speed"]),
        "context": lambda: context.context_features(tracking, tagging),
    }

    columns, column_groups = {}, {}
    for group in groups:
        values = makers[group]()
        columns.update(values)
        column_groups.update(dict.fromkeys(values, group))
    groups = column_groups

    frame = pd.DataFrame(columns, index=pd.Index(tracking.frames, name="frame"))
    frame = frame.astype(np.float32)
    if with_windows and set(windows.WINDOWED) <= set(frame.columns):
        windowed = windows.window_features(frame, fps).astype(np.float32)
        groups.update(dict.fromkeys(windowed.columns, "windows"))
        frame = pd.concat([frame, windowed], axis=1)
    frame.attrs["groups"] = groups  # column -> group, for reports; kept by to_parquet
    return frame


def features_path(key):
    return os.path.join(paths.CACHE_DIR, "features_v{}_{}.parquet".format(FEATURES_VERSION, key))


def save_features(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_parquet(path + ".part", engine="pyarrow")
    os.replace(path + ".part", path)
    return path


def load_features(path):
    return pd.read_parquet(path, engine="pyarrow")


def feature_key(tracking_spec, clean_cfg):
    blob = json.dumps({"tracking": tracking_spec, "clean": clean_cfg,
                       "version": FEATURES_VERSION}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def load_or_build(tracking_spec="atd", clean_cfg=None, force=False, verbose=True):
    """(features DataFrame, parquet path). clean_cfg None = no cleaning, {} = defaults."""
    path = features_path(feature_key(tracking_spec, clean_cfg))
    if os.path.isfile(path) and not force:
        return load_features(path), path
    from pompeu.clean import clean_tracking
    from pompeu.io.contract import load_tracking
    tracking = load_tracking(tracking_spec)
    if clean_cfg is not None:
        tracking, _ = clean_tracking(tracking, clean_cfg)
    started = time.time()
    frame = build_features(tracking)
    if verbose:
        print("built {} x {} in {:.1f} s".format(*frame.shape, time.time() - started))
    return frame, save_features(frame, path)

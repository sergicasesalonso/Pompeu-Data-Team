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

TEAM SWAP (the plan's symmetry augmentation, A7): swap=True exchanges home and away in the CLEAN
match -- slot teams flipped, the attacking direction reversed, labels H <-> A -- and only then
degrades, cleans and featurises it. Every feature is recomputed rather than translated column by
column, so the augmentation cannot be subtly wrong; it costs a second build.

TOKENS (A7's set encoder): per-player inputs at 5 Hz (every 5th frame) from the same cleaned
match -- see player_tokens().

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
# 2: the simulator blurs the ball (ball_smooth_s) -- examples built before are not comparable.
VERSION = 2
MIN_POSSESSION_S = 5.0
CACHE = os.path.join(paths.CACHE_DIR, "external", "examples")
TAGGING_PREFIXES = ("st_", "sp_")
TAGGING_COLUMNS = ("goal_active",)


TOKEN_STEP = 5            # frames per token step: 25 fps -> 5 Hz
TOKEN_FEATURES = ("team", "x", "y", "vx", "vy", "dx_ball", "dy_ball", "conf")
BALL_FEATURES = ("present", "x", "y", "vx", "vy", "speed")
PLAYER_VMAX = 12.0        # m/s
BALL_VMAX = 40.0          # m/s


def _path(name, seed, swap=False, kind="features"):
    safe = name.replace(":", "_") + ("_swap" if swap else "")
    if kind == "features":
        return os.path.join(CACHE, "{}_s{}_v{}.parquet".format(safe, seed, VERSION))
    return os.path.join(CACHE, "{}_s{}_v{}_{}.npz".format(safe, seed, VERSION, kind))


def swap_teams(tracking, labels):
    """(Tracking, labels) with home and away exchanged: the same match seen from the other side."""
    from dataclasses import replace
    team = tracking.players_team
    flipped = np.where(team >= 0, 1 - team, team).astype(np.int8)
    names = tracking.meta.get("team_names", ("home", "away"))
    meta = dict(tracking.meta, home_attack_sign=-int(tracking.meta["home_attack_sign"]),
                team_names=(names[1], names[0]))
    swapped = np.where(labels == 0, 1, np.where(labels == 1, 0, labels)).astype(np.int8)
    return replace(tracking, players_team=flipped, meta=meta), swapped


def model_columns(columns):
    """The feature columns a learned model may use (everything but Smart Tagging)."""
    return [c for c in columns
            if not c.startswith(TAGGING_PREFIXES) and c not in TAGGING_COLUMNS]


def cleaned(name, seed, swap=False, run_lengths=None):
    """(cleaned degraded Tracking, aligned labels) for one (match, seed, swap)."""
    from pompeu import external
    from pompeu.clean import clean_tracking
    from pompeu.external import degrade
    if run_lengths is None:
        from pompeu.io.contract import load_tracking
        from pompeu.validate.stress import missing_run_lengths
        run_lengths = missing_run_lengths(load_tracking("atd"))
    tracking, labels = external.load(name)
    if swap:
        tracking, labels = swap_teams(tracking, labels)
    graded = np.where(tracking.half() > 0, labels, 3).astype(np.int8)
    y = external.align_labels(graded, MIN_POSSESSION_S, tracking.meta.get("fps", 25))
    degraded = degrade.degrade(tracking, labels, seed, run_lengths)
    clean, _ = clean_tracking(degraded, {})
    return clean, y


def build(name, seed, run_lengths=None, swap=False):
    """(F, y): features of the degraded, cleaned match and its aligned labels (int8, X = 3)."""
    from pompeu.features.build import build_features
    clean, y = cleaned(name, seed, swap, run_lengths)
    return build_features(clean, tagging=None), y


def load(name, seed, run_lengths=None, cache=True, swap=False):
    """(F, y) for one (match, seed[, swap]), from the cache once built."""
    path = _path(name, seed, swap)
    if cache and os.path.isfile(path):
        frame = pd.read_parquet(path, engine="pyarrow")
        y = frame.pop("__label__").to_numpy(dtype=np.int8)
        return frame, y
    F, y = build(name, seed, run_lengths, swap)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        out = F.copy()
        out["__label__"] = y
        out.to_parquet(path + ".part", engine="pyarrow")
        os.replace(path + ".part", path)
    return F, y


def player_tokens(tracking):
    """{'players' (T, K, 8), 'mask' (T, K), 'ball' (T, 6)} at 5 Hz (TOKEN_FEATURES, BALL_FEATURES).

    Everything in the frame where the home side attacks +x (pompeu.clean.normalize), scaled to
    order one: positions as offsets from the centre spot over half the pitch, speeds over 10 m/s,
    offsets to the ball over 20 m (0 without a ball). Slot order carries no meaning -- the set
    encoder is permutation-invariant.
    """
    from pompeu.clean.normalize import attack_flip, rotate
    from pompeu.features.kinematics import ball_kinematics, player_kinematics
    length, width = tracking.meta["pitch"]
    flip = attack_flip(tracking, 0)
    rows = np.arange(0, tracking.n_frames, TOKEN_STEP)
    xy = rotate(tracking.players_xy, flip, (length, width))[rows]
    vel = player_kinematics(tracking)["vel"]
    # Clipped to what a body can do: a jump at a tracklet seam is not a 500 m/s sprint.
    vel = np.clip(np.where(flip[:, None, None], -vel, vel)[rows], -PLAYER_VMAX, PLAYER_VMAX)
    ball = rotate(tracking.ball_xy, flip, (length, width))[rows]
    bk = ball_kinematics(tracking)
    bvel = np.clip(np.where(flip[:, None], -bk["vel"], bk["vel"])[rows], -BALL_VMAX, BALL_VMAX)
    present = tracking.players_present[rows] & (tracking.players_conf[rows] > 0)
    team = np.where(tracking.players_team[rows] == 0, 1.0, -1.0)
    has_ball = ~np.isnan(ball[:, 0])
    dx = np.where(has_ball[:, None], (xy[..., 0] - ball[:, None, 0]) / 20.0, 0.0)
    dy = np.where(has_ball[:, None], (xy[..., 1] - ball[:, None, 1]) / 20.0, 0.0)
    players = np.stack([team, (xy[..., 0] - length / 2) / (length / 2),
                        (xy[..., 1] - width / 2) / (width / 2), vel[..., 0] / 10.0,
                        vel[..., 1] / 10.0, dx, dy, tracking.players_conf[rows]], axis=-1)
    players = np.where(present[..., None], np.nan_to_num(players), 0.0).astype(np.float32)
    ball_tokens = np.stack([has_ball.astype(float), (ball[:, 0] - length / 2) / (length / 2),
                            (ball[:, 1] - width / 2) / (width / 2), bvel[:, 0] / 10.0,
                            bvel[:, 1] / 10.0,
                            np.clip(bk["speed"][rows], 0, BALL_VMAX) / 10.0], axis=-1)
    ball_tokens = np.nan_to_num(np.where(has_ball[:, None], ball_tokens, 0.0)).astype(np.float32)
    return {"players": players, "mask": present, "ball": ball_tokens}


def tokens(name, seed, swap=False, run_lengths=None, cache=True):
    """player_tokens() of the cleaned degraded match plus 'y' at 5 Hz; cached as npz."""
    path = _path(name, seed, swap, kind="tokens")
    if cache and os.path.isfile(path):
        with np.load(path) as stored:
            return {key: stored[key] for key in stored.files}
    clean, y = cleaned(name, seed, swap, run_lengths)
    out = player_tokens(clean)
    out["y"] = y[::TOKEN_STEP]
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        with open(path + ".part", "wb") as handle:
            np.savez_compressed(handle, **out)
        os.replace(path + ".part", path)
    return out


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

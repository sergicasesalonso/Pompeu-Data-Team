"""B1: ball pseudo-labels for the detector (B2), produced by code from the ATD, never by a person.

    from pompeu.vision import pseudolabels
    table, report = pseudolabels.build(tracking, H, err, valid, width, height)
    pseudolabels.save(table)                       # cache/ball_pseudolabels_v1.parquet

Every graded frame where the CLEANED ATD (A1) has a ball observed by the tracker -- not filled by
interpolation -- is projected to pixels through that frame's homography. A frame is kept only if:

- the ball lands inside the image, with a margin so the detector's crop always contains it;
- the homography is trustworthy: reprojection error at or below a quantile of the graded frames'
  own errors (computed here, not tuned on any label);
- the ball moves plausibly: the cleaned speed is below a physical bound (the cleaning already
  removes > 40 m/s spikes; this drops what is left near the bound).

Frames whose ball is within `crowd_m` of a player are kept but flagged: the ball is real but may be
hidden behind a leg, so B2 can down-weight them.

The split is temporal, so the detector is validated on minutes it never saw: alternating blocks of
`block_s` seconds, one in `val_every` blocks for validation.

The table is per-frame data derived from the match: it lives in cache/ and is never committed.
`build` also returns a report of counts only, which is all that may be printed.
"""

import os

import numpy as np
import pandas as pd

from pompeu import paths
from pompeu.io.contract import SRC_ATD
from pompeu.vision.decode import project_frames

# Params
VERSION = 1
PARAMS = {
    "margin_px": 16.0,        # ball at least this far inside the image border
    "reproj_quantile": 0.95,  # keep frames whose homography error is <= this quantile
    "max_speed_ms": 35.0,     # physical bound on ball speed
    "crowd_m": 1.0,           # flag balls this close to any player
    "block_s": 60.0,          # temporal split block length
    "val_every": 5,           # one block in five goes to validation
}


def _speed(ball_xy, fps):
    """Ball speed in m/s from central differences; NaN where a neighbour is missing."""
    v = np.full(len(ball_xy), np.nan)
    d = np.linalg.norm(ball_xy[2:] - ball_xy[:-2], axis=1) * fps / 2.0
    v[1:-1] = d
    return v


def _nearest_player_m(players_xy, ball_xy):
    """Distance from the ball to the closest observed player, per frame (NaN if none)."""
    d = np.linalg.norm(players_xy - ball_xy[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    out = d.min(axis=1)
    return np.where(np.isfinite(out), out, np.nan)


def build(tracking, H, reproj_error, valid, width, height, params=None):
    """(table, report). `tracking` must already be cleaned (pompeu.clean.clean_tracking)."""
    p = {**PARAMS, **(params or {})}
    n = tracking.n_frames
    if len(H) != n:
        raise ValueError("homography has {} rows, tracking {}".format(len(H), n))
    fps = tracking.meta.get("fps", 25)
    half = tracking.half()
    graded = half > 0
    observed = (tracking.ball_source == SRC_ATD) & np.isfinite(tracking.ball_xy).all(axis=1)
    candidate = graded & observed & valid

    err_cut = float(np.quantile(reproj_error[graded & valid], p["reproj_quantile"]))
    speed = _speed(tracking.ball_xy, fps)
    rows = np.flatnonzero(candidate)
    px, py = project_frames(H[rows], tracking.ball_xy[rows, 0], tracking.ball_xy[rows, 1],
                            tracking.meta.get("game"))
    m = p["margin_px"]
    inside = (px >= m) & (px < width - m) & (py >= m) & (py < height - m)
    good_h = reproj_error[rows] <= err_cut
    plausible = ~(speed[rows] > p["max_speed_ms"])          # NaN speed (edge of a run) is kept
    keep = inside & good_h & plausible
    rows, px, py = rows[keep], px[keep], py[keep]

    crowded = _nearest_player_m(tracking.players_xy[rows], tracking.ball_xy[rows]) < p["crowd_m"]
    frames = tracking.frames[rows]
    # Absolute-frame blocks: halftime is far longer than a block, so none spans both halves.
    block = (frames // int(round(fps * p["block_s"]))).astype(np.int64)
    split = np.where(block % p["val_every"] == 0, "val", "train")

    table = pd.DataFrame({
        "frame": frames.astype(np.int64),
        "x_px": np.round(px, 1).astype(np.float32),
        "y_px": np.round(py, 1).astype(np.float32),
        "crowded": crowded,
        "split": split,
    })
    report = {
        "graded_frames": int(graded.sum()),
        "graded_with_observed_ball": int((graded & observed).sum()),
        "dropped_outside_image": int((~inside).sum()),
        "dropped_homography": int((inside & ~good_h).sum()),
        "dropped_speed": int((inside & good_h & ~plausible).sum()),
        "kept": int(len(table)),
        "kept_crowded": int(crowded.sum()),
        "train": int((split == "train").sum()),
        "val": int((split == "val").sum()),
        "reproj_cut": round(err_cut, 3),
    }
    return table, report


def cache_path():
    return os.path.join(paths.CACHE_DIR, "ball_pseudolabels_v{}.parquet".format(VERSION))


def save(table, path=None):
    path = path or cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_parquet(path, index=False)
    return path

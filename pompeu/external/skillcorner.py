"""A6 (extra training data): SkillCorner Open Data as Tracking + H/A/D labels from phases of play.

    from pompeu.external import skillcorner
    skillcorner.match_ids()                       # the A-League 2024/25 matches present
    tracking, labels = skillcorner.load(1886347)  # 25 fps, labels H=0 A=1 D=2 X=3

TRAINING ONLY. These matches are not in external.SPLITS (val/test stay as they are): they are
listed in external.EXTRA_TRAIN, and code opts in with external.resolve("extra_train").

THE DATA. github.com/SkillCorner/opendata (MIT licence, "if you use the data, we kindly ask that
you credit SkillCorner"): 20 Australian A-League 2024/25 matches, cloned to
data/external/skillcorner/opendata (gitignored; nothing derived from it is committed). Per match:
{id}_match.json (squads, periods, pitch size, home_team_side), {id}_tracking_extrapolated.jsonl,
{id}_phases_of_play.csv and {id}_dynamic_events.csv.

BROADCAST TRACKING, NOT OPTICAL. 10 fps, metres, origin at the centre spot, x along the length,
not flipped at half time (home_team_side says which way home attacks in each period). Each frame
lists the 22 players, `is_detected` true for the ones on screen (~13 of 22 on average) and an
extrapolated position for the others; on ~27% of the in-period frames (replays, close-ups --
mostly during dead ball) there is no tracking at all. The ball has a position on ~73% of the
in-period frames, `is_detected` on ~60%, and a height. Here: x + L/2, y + W/2 (so 0..L, 0..W
like the ATD); players_conf 1 for a detected player, EXTRAPOLATED_CONF for an extrapolated one
(features use conf > 0, so they still see them); the same for the ball.

LABELS FROM PHASES OF PLAY. SkillCorner's phases of play are "only defined when the ball is in
play", each carries team_in_possession_id, and two phases follow each other without a gap when
play goes on. So, at 10 fps, within each period:

    live         frames [frame_start, frame_end) of a phase -> its team in possession (H / A).
                 Consecutive phases of the same team (build-up -> create ...) merge; a change of
                 team between touching phases is a possession change without a stoppage.
    dead         in-period frames between two phases -> D. Measured on the 20 matches: ~95% of
                 these gaps sit next to a game interruption in the dynamic events (throw-in, goal
                 kick, free kick, corner, goal), a phase ends ~1 s after the last player
                 possession before a stoppage (the ball travelling out) and starts ~1.5 s before
                 the first reception after it (the restart itself), so the gaps are the stoppages.
    unknown      X before the first phase and after the last phase of each period (kick-off and
                 final whistle are not dated by a phase), and outside the periods.
    clipping     a phase is cut to its own period (on a few matches the last phase of the first
                 half runs a few seconds past the period's end frame).

Possession here is SkillCorner's team possession, which changes with control, not with every
touch; the labels still go through external.align_labels() like DFL's and Metrica's.

25 FPS, DETERMINISTICALLY. 25 fps row j is at time j / 25 s, i.e. source position 2j / 5 at
10 fps: lo = (2j) // 5 and weight w = ((2j) % 5) / 5. Positions are interpolated linearly between
source frames lo and lo + 1 when both have one, taken as they are when w = 0, NaN otherwise.
Labels and the period take source frame lo (no interpolation). Integer arithmetic only, so the
mapping is exact and the same on every machine. Row j of the Tracking is 25 fps frame j
(frame0 = 0); meta["halves"] holds each period's first and last 25 fps rows, and meta["fps"] 25.

kloppy 3.18 has a SkillCorner reader, but it drops `is_detected` (the one thing that says which
broadcast positions were seen); the JSONL is plain enough to read directly.
"""

import csv
import json
import os

import numpy as np

from pompeu import paths
from pompeu.io.contract import SRC_ATD, SRC_NONE, Tracking

# Params
VERSION = 1
ROOT = os.path.join(paths.ROOT, "data", "external", "skillcorner", "opendata", "data")
CACHE = os.path.join(paths.CACHE_DIR, "external")
SRC_FPS, FPS = 10, 25
EXTRAPOLATED_CONF = 0.5   # players_conf / ball_conf of a position SkillCorner did not see
H, A, D, X = 0, 1, 2, 3


def _folder(match_id, root):
    return os.path.join(root, "matches", str(match_id))


def _file(match_id, root, suffix):
    return os.path.join(_folder(match_id, root), "{}_{}".format(match_id, suffix))


def available(match_id, root=ROOT):
    return all(os.path.isfile(_file(match_id, root, s)) for s in
               ("match.json", "tracking_extrapolated.jsonl", "phases_of_play.csv"))


def match_ids(root=ROOT):
    """Sorted ids of the matches present on disk."""
    folder = os.path.join(root, "matches")
    if not os.path.isdir(folder):
        return []
    return sorted(int(name) for name in os.listdir(folder)
                  if name.isdigit() and available(int(name), root))


def read_meta(match_id, root=ROOT):
    with open(_file(match_id, root, "match.json"), encoding="utf-8") as handle:
        return json.load(handle)


def read_phases(match_id, root=ROOT):
    """[(frame_start, frame_end, period, team_in_possession_id)] at 10 fps."""
    out = []
    with open(_file(match_id, root, "phases_of_play.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.append((int(row["frame_start"]), int(row["frame_end"]), int(row["period"]),
                        int(row["team_in_possession_id"])))
    return out


def labels_from_phases(n, periods, phases, home_id):
    """(n,) H/A/D/X at the source rate (see the module docstring).

    periods: [(period, first, last)] inclusive source frames; phases as read_phases()."""
    labels = np.full(n, X, dtype=np.int8)
    for period, first, last in periods:
        own = sorted((max(s, first), min(e, last + 1), team) for s, e, p, team in phases
                     if p == period and min(e, last + 1) > max(s, first))
        if not own:
            continue
        labels[own[0][0]:own[-1][1]] = D
        for start, end, team in own:
            labels[start:end] = H if team == home_id else A
    return labels


def upsample_index(n_src):
    """(lo, w) for the 25 fps rows covering n_src source frames: row j sits at source 2j/5."""
    n = (5 * (n_src - 1)) // 2 + 1
    two_j = 2 * np.arange(n, dtype=np.int64)
    return two_j // 5, (two_j % 5) / 5.0


def resample(values, lo, w):
    """values (n_src, ...) float -> (len(lo), ...): linear between lo and lo+1 when both are
    present, the source value when w == 0, NaN otherwise."""
    values = np.asarray(values, dtype=np.float64)
    hi = np.minimum(lo + 1, len(values) - 1)
    shape = (-1,) + (1,) * (values.ndim - 1)
    ww = w.reshape(shape)
    out = values[lo] * (1 - ww) + values[hi] * ww
    exact = (w == 0).reshape(shape)
    return np.where(exact, values[lo], out)   # NaN propagates where a neighbour is missing


def read_tracking(match_id, slot_of, root=ROOT):
    """Source-rate arrays: period, players_xy (n,k,2), players_det, ball_xy, ball_z, ball_det.
    Coordinates are SkillCorner's (centred)."""
    rows = []
    with open(_file(match_id, root, "tracking_extrapolated.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["frame"])
    n = rows[-1]["frame"] + 1
    k = len(slot_of)
    period = np.zeros(n, dtype=np.int8)
    players_xy = np.full((n, k, 2), np.nan)
    players_det = np.zeros((n, k), dtype=bool)
    ball_xy = np.full((n, 2), np.nan)
    ball_z = np.full(n, np.nan)
    ball_det = np.zeros(n, dtype=bool)
    for r in rows:
        f = r["frame"]
        period[f] = r.get("period") or 0
        for p in r.get("player_data") or []:
            slot = slot_of.get(p.get("player_id"))
            if slot is None or p.get("x") is None or p.get("y") is None:
                continue
            players_xy[f, slot] = (p["x"], p["y"])
            players_det[f, slot] = bool(p.get("is_detected"))
        b = r.get("ball_data") or {}
        if b.get("x") is not None and b.get("y") is not None:
            ball_xy[f] = (b["x"], b["y"])
            ball_z[f] = b["z"] if b.get("z") is not None else np.nan
            ball_det[f] = bool(b.get("is_detected"))
    return period, players_xy, players_det, ball_xy, ball_z, ball_det


def convert(match_id, root=ROOT):
    """(Tracking at 25 fps, labels) from the files."""
    meta = read_meta(match_id, root)
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    players = sorted(meta["players"], key=lambda p: (p["team_id"] != home_id, p["id"]))
    slot_of = {p["id"]: i for i, p in enumerate(players)}
    slot_team = np.array([0 if p["team_id"] == home_id else 1 for p in players], dtype=np.int8)
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])

    period, pxy, pdet, bxy, bz, bdet = read_tracking(match_id, slot_of, root)
    n_src = len(period)
    periods = []
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        if len(rows):
            periods.append((half, int(rows[0]), int(rows[-1])))
    labels_src = labels_from_phases(n_src, periods, read_phases(match_id, root), home_id)

    lo, w = upsample_index(n_src)
    offset = np.array([length / 2, width / 2])
    players_xy = np.round(resample(pxy, lo, w) + offset, 2)
    ball_xy = np.round(resample(bxy, lo, w) + offset, 2)
    ball_z = resample(bz, lo, w).astype(np.float32)
    hi = np.minimum(lo + 1, n_src - 1)
    exact = w == 0
    p_seen = np.where(exact[:, None], pdet[lo], pdet[lo] & pdet[hi])
    b_seen = np.where(exact, bdet[lo], bdet[lo] & bdet[hi])
    observed = ~np.isnan(players_xy[:, :, 0])
    ball = ~np.isnan(ball_xy[:, 0])
    ball_z[~ball] = np.nan
    labels = labels_src[lo]

    halves = []
    for half, first, last in periods:
        rows = np.flatnonzero((lo >= first) & (lo <= last))
        halves.append((int(half), int(rows[0]), int(rows[-1])))
    side = meta.get("home_team_side") or ["left_to_right"]
    home_attack_sign = 1 if side[0] == "left_to_right" else -1

    tracking = Tracking(
        players_xy=players_xy,
        players_team=np.broadcast_to(slot_team, observed.shape),
        players_conf=np.where(observed, np.where(p_seen, 1.0, EXTRAPOLATED_CONF), 0.0)
        .astype(np.float32),
        ball_xy=ball_xy,
        ball_z=ball_z,
        ball_conf=np.where(ball, np.where(b_seen, 1.0, EXTRAPOLATED_CONF), 0.0).astype(np.float32),
        ball_source=np.where(ball, SRC_ATD, SRC_NONE).astype(np.int8),
        frame0=0,
        meta={"game": "SKILLCORNER-{}".format(match_id), "source": "skillcorner:{}".format(match_id),
              "fps": FPS, "source_fps": SRC_FPS, "pitch": (length, width),
              "team_names": (meta["home_team"].get("short_name", str(home_id)),
                             meta["away_team"].get("short_name", str(away_id))),
              "home_attack_sign": int(home_attack_sign), "halves": halves},
    )
    return tracking, labels


def _cache_path(match_id):
    return os.path.join(CACHE, "skillcorner_{}_v{}.npz".format(match_id, VERSION))


def load(match_id, root=ROOT, cache=True):
    """(Tracking, labels) for one match, from cache/external/ once built."""
    match_id = int(match_id)
    path = _cache_path(match_id)
    if cache and os.path.isfile(path):
        tracking = Tracking.load(path)
        with np.load(path) as stored:
            return tracking, stored["labels"].astype(np.int8)
    tracking, labels = convert(match_id, root)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        tracking.save(path)
        with np.load(path) as stored:
            arrays = {name: stored[name] for name in stored.files}
        with open(path + ".part", "wb") as handle:
            np.savez(handle, labels=labels, **arrays)
        os.replace(path + ".part", path)
    return tracking, labels

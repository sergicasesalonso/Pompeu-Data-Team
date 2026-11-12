"""A6: Metrica Sports' three open sample matches as Tracking + H/A/D labels derived from events.

    from pompeu.external import metrica
    tracking, labels = metrica.load(1)            # games 1, 2, 3

THE DATA. github.com/metrica-sports/sample-data: three anonymised professional matches, tracking at
25 fps for both teams and the ball, plus manually collected event data. Games 1 and 2 are CSV
(one file per team); game 3 is FIFA EPTS -- the very format of our ATD -- with events in JSON.
Same provider as the challenge's tracking. Cloned to data/external/metrica/sample-data
(gitignored).

LABELS FROM EVENTS -- there is no per-frame possession in these files, so the plan's rule is
applied: "the team of the last event with control, and D between the ball going out and the
restart".

    control events  PASS, CARRY, RECOVERY, SET PIECE, SHOT: the acting team has the ball from the
                    event's start frame. BALL LOST and CHALLENGE are not control: who ends up
                    with the ball is told by the next control event.
    dead ball       from a BALL OUT, a FAULT RECEIVED, or a SHOT ending in GOAL, until the start
                    frame of the next SET PIECE -- or of the next control event if play went on
                    without one (advantage).
    possession      changes at the start frame of the first control event of the other team,
                    which is where the challenge's ground truth dates a change too (the new team's
                    first touch).

These labels are coarser than DFL's -- events are collected by hand at about one per second,
so a possession boundary is only as precise as the event that opens it -- and they go through the
same align_labels() convention mapping as DFL.
"""

import csv
import glob
import json
import os

import numpy as np

from pompeu import paths
from pompeu.io.contract import SRC_ATD, SRC_NONE, Tracking

# Params
VERSION = 1
ROOT = os.path.join(paths.ROOT, "data", "external", "metrica", "sample-data", "data")
CACHE = os.path.join(paths.CACHE_DIR, "external")
GAMES = (1, 2, 3)
PITCH = (105.0, 68.0)
CONTROL = ("PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT")
H, A, D, X = 0, 1, 2, 3


def _folder(game, root):
    return os.path.join(root, "Sample_Game_{}".format(game))


def available(game, root=ROOT):
    folder = _folder(game, root)
    return bool(glob.glob(os.path.join(folder, "*Tracking*")) or
                glob.glob(os.path.join(folder, "*tracking*")))


def read_events(game, root=ROOT, team_ground=None):
    """[(start_frame, team 0/1, type, subtypes)] in time order, frames as in the tracking."""
    folder = _folder(game, root)
    out = []
    if game in (1, 2):
        path = os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(game))
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                team = 0 if row["Team"].strip().lower() == "home" else 1
                out.append((int(row["Start Frame"]), team, row["Type"].strip().upper(),
                            row["Subtype"].strip().upper()))
    else:
        with open(os.path.join(folder, "Sample_Game_3_events.json"), encoding="utf-8") as handle:
            data = json.load(handle)["data"]
        for event in data:
            subtypes = event.get("subtypes") or []
            if isinstance(subtypes, dict):
                subtypes = [subtypes]
            names = "-".join(s["name"] for s in subtypes if isinstance(s, dict))
            team = team_ground.get(event["team"]["id"], 1) if team_ground else 1
            out.append((int(event["start"]["frame"]), team, event["type"]["name"].upper(),
                        names.upper()))
    out.sort(key=lambda e: e[0])
    return out


def labels_from_events(events, frame_ids, period):
    """(N,) H/A/D/X codes on the tracking rows, from the events (see the module docstring)."""
    n = len(frame_ids)
    row_of = {int(f): i for i, f in enumerate(frame_ids)}
    labels = np.full(n, X, dtype=np.int8)
    team, dead = None, False
    marks = []  # (row, state) where state is 0/1 for a team or D
    for frame, side, kind, sub in events:
        row = row_of.get(frame)
        if row is None:
            continue
        if kind in ("BALL OUT", "FAULT RECEIVED") or (kind == "SHOT" and "GOAL" in sub):
            marks.append((row, D))
            dead = True
        elif kind in CONTROL:
            if dead or side != team:
                marks.append((row, side))
            team, dead = side, False
    marks.sort()
    for (row, state), nxt in zip(marks, [m[0] for m in marks[1:]] + [n]):
        labels[row:nxt] = state
    # Nothing is known before the first event of each period: X there, and never across periods.
    for half in np.unique(period):
        rows = np.flatnonzero(period == half)
        firsts = [r for r, _ in marks if rows[0] <= r <= rows[-1]]
        if firsts:
            labels[rows[0]:firsts[0]] = X
            # A label carried from the previous period is not this period's.
        else:
            labels[rows] = X
    return labels


def convert(game, root=ROOT):
    """(Tracking, labels) from the files through kloppy."""
    from kloppy import metrica as kloppy_metrica

    folder = _folder(game, root)
    if game in (1, 2):
        dataset = kloppy_metrica.load_tracking_csv(
            home_data=os.path.join(folder, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(game)),
            away_data=os.path.join(folder, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(game)))
    else:
        dataset = kloppy_metrica.load_tracking_epts(
            meta_data=os.path.join(folder, "Sample_Game_3_metadata.xml"),
            raw_data=os.path.join(folder, "Sample_Game_3_tracking.txt"))
    teams = dataset.metadata.teams
    home = next((t for t in teams if str(t.ground) == "home"), teams[0])
    players = [player for team in teams for player in team.players]
    slot_of = {player.player_id: k for k, player in enumerate(players)}
    slot_team = np.array([0 if player.team is home else 1 for player in players], dtype=np.int8)
    team_ground = {team.team_id: (0 if team is home else 1) for team in teams}

    records = dataset.records
    n, k = len(records), len(players)
    players_xy = np.full((n, k, 2), np.nan)
    ball_xy = np.full((n, 2), np.nan)
    frame_ids = np.empty(n, dtype=np.int64)
    period = np.zeros(n, dtype=np.int8)
    for row, frame in enumerate(records):
        frame_ids[row] = frame.frame_id
        period[row] = frame.period.id
        for player, data in frame.players_data.items():
            point = data.coordinates
            if point is not None and point.x is not None and not np.isnan(point.x):
                players_xy[row, slot_of[player.player_id]] = (point.x * PITCH[0],
                                                              point.y * PITCH[1])
        ball = frame.ball_coordinates
        if ball is not None and ball.x is not None and not np.isnan(ball.x):
            ball_xy[row] = (ball.x * PITCH[0], ball.y * PITCH[1])

    labels = labels_from_events(read_events(game, root, team_ground), frame_ids, period)
    halves = []
    for half in sorted(np.unique(period)):
        rows = np.flatnonzero(period == half)
        if half in (1, 2):
            halves.append((int(half), int(rows[0]), int(rows[-1])))
    first = halves[0][1]
    # Home's own half at the first kick-off; the first frames can still be empty, so look at the
    # first second with players on the pitch.
    seen = np.flatnonzero(~np.isnan(players_xy[first:first + 250, slot_team == 0, 0]).any(axis=1))
    probe = first + (int(seen[0]) if len(seen) else 0)
    home_attack_sign = 1 if np.nanmedian(players_xy[probe, slot_team == 0, 0]) < PITCH[0] / 2 \
        else -1

    ball = ~np.isnan(ball_xy[:, 0])
    observed = ~np.isnan(players_xy[:, :, 0])
    tracking = Tracking(
        players_xy=players_xy,
        players_team=np.broadcast_to(slot_team, observed.shape),
        players_conf=observed.astype(np.float32),
        ball_xy=ball_xy,
        ball_z=np.full(n, np.nan, dtype=np.float32),
        ball_conf=ball.astype(np.float32),
        ball_source=np.where(ball, SRC_ATD, SRC_NONE).astype(np.int8),
        frame0=0,
        meta={"game": "METRICA-{}".format(game), "source": "metrica:{}".format(game), "fps": 25,
              "pitch": PITCH, "team_names": ("Home", "Away"),
              "home_attack_sign": int(home_attack_sign), "halves": halves},
    )
    return tracking, labels


def _cache_path(game):
    return os.path.join(CACHE, "metrica_{}_v{}.npz".format(game, VERSION))


def load(game, root=ROOT, cache=True):
    """(Tracking, labels) for one sample game, from cache/external/ once built."""
    path = _cache_path(game)
    if cache and os.path.isfile(path):
        tracking = Tracking.load(path)
        with np.load(path) as stored:
            return tracking, stored["labels"].astype(np.int8)
    tracking, labels = convert(game, root)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        tracking.save(path)
        with np.load(path) as stored:
            arrays = {name: stored[name] for name in stored.files}
        with open(path + ".part", "wb") as handle:
            np.savez(handle, labels=labels, **arrays)
        os.replace(path + ".part", path)
    return tracking, labels

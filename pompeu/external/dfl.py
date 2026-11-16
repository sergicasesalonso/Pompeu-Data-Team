"""A6: the DFL / IDSSE open matches as Tracking + per-frame H/A/D labels.

    from pompeu.external import dfl
    tracking, labels = dfl.load("J03WPY")      # labels: (N,) int8 codes H=0 A=1 D=2 X=3

THE DATASET. Bassek, Weber, Rein & Memmert (2025), "An integrated dataset of spatiotemporal and
event data in elite soccer", Scientific Data 12, 195 (doi:10.1038/s41597-025-04505-y). Seven
Bundesliga / 2. Bundesliga 2022/23 matches, TRACAB optical tracking at 25 fps, CC-BY 4.0. The
official copy is on figshare (doi:10.6084/m9.figshare.28196177); figshare challenges scripted
downloads, so scripts/prepare_external.py fetches the identical files from the Hugging Face mirror
pysport/idsse-data. Files go to data/external/idsse/ (gitignored: licensed data stays out of the
public repository, and so does the cache built from it).

WHY IT FITS. Every frame carries BallStatus (alive / dead) and BallPossession (home / away): the
same three states as our ground truth, frame by frame, from corrected tracking -- which is also
how the challenge's ground truth was made. Labels: dead -> D; alive -> the possessing team; an
alive frame without a possessor keeps the previous label (it never happens on these files, but
the rule matches "possession stays with the last team that had it").

CONVENTIONS DIFFER. DFL's BallPossession changes on almost every touch: 8.4 possessions per
minute of live ball on these matches, against 4.2 in the challenge's public ground truth (17 in
4.0 live minutes), whose rule is that possession changes only with CONTROL. align_labels() maps
DFL onto that rule with one parameter, as stage A6 allows (at most 3, measured on the public
window): a possession shorter than min_possession_s between two of the other team's, with no
stoppage in between, is a touch without control and goes to the surrounding team. 5 s brings DFL
to 4.26 per live minute -- the public rate. That rate comes from 17 possessions (~ +/- 25%), so
the threshold is a measured convention, not a precise one.

THE CONVERSION. kloppy reads the files (only_alive=False: the dead frames are the D labels).
Rows are the frames in order, both halves back to back, frame0 = 0; meta["halves"] records where
each half starts and ends, which is all Tracking.half_blocks() needs. kloppy's 0-1 coordinates are
scaled by the pitch from the match information (105 x 68). One slot per player of the squad, team
by side, NaN where the player is not on the pitch. The ball's height goes into ball_z -- the ATD
has none, so nothing downstream relies on it yet. The home side's attacking direction is read off
the data (where the home outfield players stand at the first kick-off) and checked against
kloppy's orientation.

This is CLEAN data: complete tracks, no false ball, no gaps. pompeu.external.degrade makes it look
like our ATD before any method sees it.
"""

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

from pompeu import paths
from pompeu.io.contract import SRC_ATD, SRC_NONE, Tracking

# Params
# Bump when the conversion changes, so cached matches are rebuilt.
VERSION = 1
ROOT = os.path.join(os.environ.get("MTAH_DATA_DIR") or os.path.join(paths.ROOT, "data"), "external", "idsse")
CACHE = os.path.join(paths.CACHE_DIR, "external")
MATCHES = {
    "J03WMX": "1. FC Köln - FC Bayern München (Bundesliga)",
    "J03WN1": "VfL Bochum - Bayer 04 Leverkusen (Bundesliga)",
    "J03WPY": "Fortuna Düsseldorf - 1. FC Nürnberg (2. Bundesliga)",
    "J03WOH": "Fortuna Düsseldorf - SSV Jahn Regensburg (2. Bundesliga)",
    "J03WQQ": "Fortuna Düsseldorf - FC St. Pauli (2. Bundesliga)",
    "J03WOY": "Fortuna Düsseldorf - F.C. Hansa Rostock (2. Bundesliga)",
    "J03WR9": "Fortuna Düsseldorf - 1. FC Kaiserslautern (2. Bundesliga)",
}
H, A, D, X = 0, 1, 2, 3


def align_labels(labels, min_possession_s, fps=25):
    """Copy of labels where H-A-H (or A-H-A) possessions shorter than min_possession_s take the
    surrounding team. X frames separate runs (halves), and D frames are never touched."""
    from pompeu import paths  # noqa: F401  (possession_metrics on the path)
    import possession_metrics as pm
    out = np.asarray(labels, dtype=np.int8).copy()
    min_frames = int(round(min_possession_s * fps))
    if min_frames <= 0:
        return out
    runs = pm.valid_runs(out)
    while True:
        segments = pm.segments(out, runs)
        changed = False
        for before, (a, b, k), after in zip(segments, segments[1:], segments[2:]):
            if (k in (H, A) and before[2] == after[2] and before[2] in (H, A)
                    and before[2] != k and before[1] == a and after[0] == b
                    and b - a < min_frames):
                out[a:b] = before[2]
                changed = True
        if not changed:
            return out


def files(match_id, root=ROOT):
    """(match information xml, positions xml) for a match, or a readable error."""
    meta = glob.glob(os.path.join(root, "*matchinformation*{}.xml".format(match_id)))
    raw = glob.glob(os.path.join(root, "*positions_raw_observed*{}.xml".format(match_id)))
    if not meta or not raw:
        raise IOError("IDSSE match {} not found in {} -- run scripts/prepare_external.py".format(
            match_id, root))
    return meta[0], raw[0]


def pitch_size(meta_path):
    """(length, width) in metres from the match information."""
    general = ET.parse(meta_path).getroot().find(".//Environment")
    return float(general.get("PitchX")), float(general.get("PitchY"))


def _cache_path(match_id):
    return os.path.join(CACHE, "dfl_{}_v{}.npz".format(match_id, VERSION))


def convert(match_id, root=ROOT):
    """(Tracking, labels) straight from the XML through kloppy. Slow (~2 min a match)."""
    from kloppy import sportec

    meta_path, raw_path = files(match_id, root)
    length, width = pitch_size(meta_path)
    dataset = sportec.load_tracking(meta_data=meta_path, raw_data=raw_path, only_alive=False)
    teams = dataset.metadata.teams
    home = next(team for team in teams if str(team.ground) == "home")
    players = [player for team in teams for player in team.players]
    slot_of = {player.player_id: k for k, player in enumerate(players)}
    slot_team = np.array([0 if player.team is home else 1 for player in players], dtype=np.int8)

    records = dataset.records
    n, k = len(records), len(players)
    players_xy = np.full((n, k, 2), np.nan)
    ball_xy = np.full((n, 2), np.nan)
    ball_z = np.full(n, np.nan, dtype=np.float32)
    labels = np.full(n, X, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)
    for row, frame in enumerate(records):
        period[row] = frame.period.id
        for player, data in frame.players_data.items():
            point = data.coordinates
            if point is not None and player.player_id in slot_of:
                players_xy[row, slot_of[player.player_id]] = (point.x * length, point.y * width)
        ball = frame.ball_coordinates
        if ball is not None and ball.x is not None:
            ball_xy[row] = (ball.x * length, ball.y * width)
            ball_z[row] = getattr(ball, "z", np.nan) if getattr(ball, "z", None) is not None \
                else np.nan
        if "DEAD" in str(frame.ball_state).upper():
            labels[row] = D
        elif frame.ball_owning_team is not None:
            labels[row] = H if frame.ball_owning_team is home else A
    # Alive without a possessor: the last team that had it, within the half.
    for half in np.unique(period):
        rows = np.flatnonzero(period == half)
        for row in rows[1:]:
            if labels[row] == X and labels[row - 1] != X:
                labels[row] = labels[row - 1]

    halves = []
    for half in sorted(np.unique(period)):
        rows = np.flatnonzero(period == half)
        halves.append((int(half), int(rows[0]), int(rows[-1])))

    # Home attacking direction: home outfield players stand in their own half at kick-off.
    first = halves[0][1]
    home_x = players_xy[first, slot_team == 0, 0]
    home_attack_sign = 1 if np.nanmedian(home_x) < length / 2 else -1
    kloppy_says = {"HOME_AWAY": 1, "AWAY_HOME": -1}.get(dataset.metadata.orientation.name)
    if kloppy_says is not None and kloppy_says != home_attack_sign:
        raise ValueError("{}: attack direction from positions ({}) disagrees with kloppy ({})".format(
            match_id, home_attack_sign, kloppy_says))

    ball = ~np.isnan(ball_xy[:, 0])
    observed = ~np.isnan(players_xy[:, :, 0])
    tracking = Tracking(
        players_xy=players_xy,
        players_team=np.broadcast_to(slot_team, observed.shape),
        players_conf=observed.astype(np.float32),
        ball_xy=ball_xy,
        ball_z=ball_z,
        ball_conf=ball.astype(np.float32),
        ball_source=np.where(ball, SRC_ATD, SRC_NONE).astype(np.int8),
        frame0=0,
        meta={"game": "DFL-" + match_id, "source": "dfl:" + match_id, "fps": 25,
              "pitch": (length, width), "team_names": (home.name, next(
                  t.name for t in teams if t is not home)),
              "home_attack_sign": int(home_attack_sign), "halves": halves},
    )
    return tracking, labels


def load(match_id, root=ROOT, cache=True):
    """(Tracking, labels) for one IDSSE match, from cache/external/ when it has been built."""
    path = _cache_path(match_id)
    if cache and os.path.isfile(path):
        tracking = Tracking.load(path)
        with np.load(path) as stored:
            return tracking, stored["labels"].astype(np.int8)
    tracking, labels = convert(match_id, root)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        tracking.save(path)
        # Add the labels to the same file: one match, one file.
        with np.load(path) as stored:
            arrays = {name: stored[name] for name in stored.files}
        with open(path + ".part", "wb") as handle:
            np.savez(handle, labels=labels, **arrays)
        os.replace(path + ".part", path)
    return tracking, labels

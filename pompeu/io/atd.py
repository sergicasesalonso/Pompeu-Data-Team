"""Fast ATD loader: the raw FIFA/EPTS text straight into numpy, in metres, indexed by video frame.

    from pompeu.io.atd import load_atd
    atd = load_atd()                # ~6 s the first time, well under 1 s from the cache
    atd.P[frame, slot]              # (x, y, s): metres, metres, m/s -- NaN where not tracked
    atd.B[frame]                    # (x, y, s) for the ball
    atd.slot_team[slot]             # 0 home, 1 away

Why not kloppy: it builds a Python object per frame and per player, minutes and gigabytes for
199319 x 46 slots. Every line of the raw file has the same shape --

    <frame_id>:<x,y,s>;<x,y,s>;...;<x,y,s>:<x,y,s>
                 46 player slots              ball

-- so after turning ':' and ';' into ',' the file is a plain 142-column CSV, and pandas' C parser
reads it in one pass. float_precision="round_trip" makes every value bit-identical to Python's
float(), which is what starter/baseline.py uses: the two read the same numbers, so our baseline
reproduces the published one exactly (tests/test_baseline.py).

Index: row i of every array is VIDEO frame i, i.e. frame_id - 1 (DATA.md, "Frame alignment").
The loader checks the ids are exactly 1..n and refuses anything else rather than shifting.

Coordinates: the file is normalised 0-1 with Metrica's orientation; x is multiplied by the pitch
length and y by the width from games.py (105.3 x 68.0, not the 105 x 68 the XML claims). No axis
is flipped. Speed is passed through as the file has it -- it is empty on most frames, and A1
recomputes it.
"""

import hashlib
import io
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pompeu import paths
import games

# Params
# Bump when the parse or the cached layout changes, so stale caches are rebuilt rather than read.
PARSER_VERSION = 1
CHANNELS = 3  # x, y, s per object
HOME, AWAY = 0, 1
# XML attack_direction_first_half -> sign of the x direction that team attacks in the 1st half.
DIRECTION_SIGN = {"left_to_right": 1, "right_to_left": -1}


@dataclass
class ATD:
    """The tracking exactly as provided, in metres. Line A reads it through Tracking, not this."""

    P: np.ndarray            # (N, K, 3) float64: x m, y m, speed m/s
    B: np.ndarray            # (N, 3) float64
    slot_team: np.ndarray    # (K,) int8, HOME or AWAY
    team_names: tuple        # (home, away)
    home_attack_sign: int    # +1 if home attacks towards +x in the 1st half, -1 otherwise
    game: str
    source: str

    @property
    def n_frames(self):
        return self.P.shape[0]

    @property
    def n_slots(self):
        return self.P.shape[1]


def parse_meta(meta_path):
    """(slot_team, team_names, home_attack_sign) from the metadata XML.

    The first <Team> is home -- the FIFA/EPTS convention, the one kloppy and starter/baseline.py
    follow, and for this match Barcelona, which DATA.md states independently. Slots are in
    <Player> order, which is the channel order of the raw file.
    """
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")

    slot_team = np.array([HOME if player.get("teamId") == home_id else AWAY
                          for player in root.iter("Player")], dtype=np.int8)
    if not len(slot_team):
        raise ValueError("{}: no Player entries".format(meta_path))

    direction = None
    for parameter in teams[0].iter("ProviderParameter"):
        if parameter.findtext("Name") == "attack_direction_first_half":
            direction = parameter.findtext("Value")
    if direction not in DIRECTION_SIGN:
        raise ValueError("{}: home attack_direction_first_half is {!r}".format(meta_path, direction))

    names = tuple(team.findtext("Name") for team in teams)
    return slot_team, names, DIRECTION_SIGN[direction]


def parse_raw(raw_path, n_slots, pitch):
    """(frame_ids, P, B) from the raw text. P and B already in metres."""
    with open(raw_path, "rb") as handle:
        text = handle.read().replace(b":", b",").replace(b";", b",")
    table = pd.read_csv(io.BytesIO(text), header=None, dtype=np.float64,
                        float_precision="round_trip").to_numpy()

    n_columns = 1 + CHANNELS * (n_slots + 1)
    if table.shape[1] != n_columns:
        raise ValueError("{}: {} columns per line, expected {} for {} slots".format(
            raw_path, table.shape[1], n_columns, n_slots))

    frame_ids = table[:, 0].astype(np.int64)
    objects = table[:, 1:].reshape(len(table), n_slots + 1, CHANNELS)
    objects[:, :, 0] *= pitch[0]
    objects[:, :, 1] *= pitch[1]
    return frame_ids, np.ascontiguousarray(objects[:, :n_slots]), objects[:, n_slots].copy()


def _cache_path(raw_path, meta_path, pitch, cache_dir):
    """Cache file keyed on both inputs' size and mtime, the pitch and PARSER_VERSION."""
    key = repr([(os.path.getsize(p), os.stat(p).st_mtime_ns) for p in (raw_path, meta_path)]
               + [tuple(pitch), PARSER_VERSION]).encode()
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    return os.path.join(cache_dir, "{}_v{}_{}.npz".format(
        stem, PARSER_VERSION, hashlib.sha1(key).hexdigest()[:10]))


def load_atd(game=games.GAME, source=games.DEFAULT_SOURCE, data_dir=None, cache=True,
             cache_dir=None, verbose=False):
    """The ATD of one recording as an ATD. Parsed once, then read from an .npz cache."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir, source)
    pitch = games.GAMES[game]["pitch"]
    slot_team, names, home_sign = parse_meta(meta_path)

    path = _cache_path(raw_path, meta_path, pitch, cache_dir or paths.CACHE_DIR)
    if cache and os.path.isfile(path):
        with np.load(path) as stored:
            P, B = stored["P"], stored["B"]
    else:
        started = time.time()
        frame_ids, P, B = parse_raw(raw_path, len(slot_team), pitch)
        expected = games.source_frames(source, game)
        if not np.array_equal(frame_ids, np.arange(1, expected + 1)):
            raise ValueError("{}: frame ids are not 1..{} -- refusing to guess an alignment"
                             .format(raw_path, expected))
        if verbose:
            print("parsed {} in {:.1f} s".format(os.path.basename(raw_path), time.time() - started))
        if cache:
            # Written aside and renamed, so an interrupted run never leaves a truncated cache.
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as handle:
                np.savez(handle, P=P, B=B)
            os.replace(path + ".part", path)

    return ATD(P=P, B=B, slot_team=slot_team, team_names=names, home_attack_sign=home_sign,
               game=game, source=source)

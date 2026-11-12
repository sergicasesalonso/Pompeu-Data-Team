"""Smart Tagging as a table on video frames, with Team already mapped to our team codes.

Thin on purpose: starter/load_smart_tagging.py does the parsing and its time base is verified
there. What this adds is the club-name -> side mapping, read from the ATD metadata rather than
hard-coded, and N/A kept as unknown instead of silently dropped.
"""

import pandas as pd

from pompeu import paths  # noqa: F401
from pompeu.io.contract import TEAM_AWAY, TEAM_HOME, TEAM_UNKNOWN
import games
from starter import load_smart_tagging


def team_code(name, team_names):
    """0 home, 1 away, -1 for N/A or anything unrecognised. Case- and space-insensitive."""
    wanted = (name or "").strip().lower()
    for code, club in ((TEAM_HOME, team_names[0]), (TEAM_AWAY, team_names[1])):
        if wanted == club.strip().lower():
            return code
    return TEAM_UNKNOWN


def load_tagging(team_names, game=games.GAME, data_dir=None):
    """DataFrame: code, start_frame, end_frame (exclusive), team, type, half, plus all tags."""
    rows = []
    for event in load_smart_tagging.load(game, data_dir):
        tags = event["tags"]
        rows.append({
            "code": (event["code"] or "").strip().upper(),
            "start_frame": event["start_frame"],
            "end_frame": event["end_frame"],
            "team": team_code(tags.get("Team"), team_names),
            "type": (tags.get("Type") or "").strip().lower() or None,
            "half": {"1st Half": 1, "2nd Half": 2}.get(tags.get("Half"), 0),
            "tags": tags,
        })
    return pd.DataFrame(rows)

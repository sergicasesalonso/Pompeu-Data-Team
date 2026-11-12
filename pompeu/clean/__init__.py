"""A1: tracking cleaning. Tracking in, cleaned Tracking out -- the contract does not change.

    from pompeu.clean import clean_tracking
    clean, report = clean_tracking(tracking)            # defaults below
    clean, report = clean_tracking(tracking, {"stitching": {"enabled": False}})

Order: ball (spikes, fragments, short gaps) independently of players; players frozen positions ->
duplicates -> stitching. Each step can be switched off or retuned from a config's
`pipeline.clean` block, and `report` counts what each one did (scripts/cleaning_report.py writes it
to results/cleaning_report.md).

Velocities are not stored in the Tracking: pompeu.features.kinematics recomputes them from the
cleaned positions (Savitzky-Golay, window 7, order 2). Coordinates stay in the ATD frame;
pompeu.clean.normalize gives the "team attacks +x" view when a feature wants it.
"""

import copy
from dataclasses import replace

import numpy as np

from pompeu.clean import ball, players, stitching
from pompeu.io.contract import SRC_INTERP, SRC_NONE

DEFAULTS = {
    "ball": {"enabled": True, "vmax": ball.VMAX, "min_run": ball.MIN_RUN,
             "interp_max_s": ball.INTERP_MAX_S, "interp_conf": ball.INTERP_CONF},
    "frozen": {"enabled": True, "window_s": players.FROZEN_WINDOW_S,
               "max_disp_m": players.FROZEN_MAX_DISP_M,
               "field_min_speed": players.FIELD_MIN_SPEED,
               "action": "flag",  # flag: keep at low confidence; drop: remove
               "conf": players.FROZEN_CONF},
    "duplicates": {"enabled": True, "max_dist_m": players.DUP_MAX_DIST_M,
                   "min_s": players.DUP_MIN_S},
    "stitching": {"enabled": True, "max_gap_s": stitching.MAX_GAP_S, "vmax": stitching.VMAX,
                  "margin_m": stitching.MARGIN_M, "fill_max_s": stitching.FILL_MAX_S,
                  "fill_conf": stitching.FILL_CONF},
}


def _settings(cfg):
    out = copy.deepcopy(DEFAULTS)
    for step, values in (cfg or {}).items():
        if step not in out:
            raise KeyError("unknown clean step {!r}; have {}".format(step, sorted(out)))
        out[step].update(values or {})
    return out


def slot_teams(tracking):
    """(K,) team per slot: the most frequent observed team, -1 if never observed."""
    out = np.full(tracking.n_slots, -1, dtype=np.int8)
    present = tracking.players_present
    for slot in range(tracking.n_slots):
        values = tracking.players_team[present[:, slot], slot]
        values = values[values >= 0]
        if len(values):
            out[slot] = np.bincount(values).argmax()
    return out


def clean_tracking(tracking, cfg=None):
    """(cleaned Tracking, report dict)."""
    settings = _settings(cfg)
    fps = tracking.meta.get("fps", 25)
    report = {"frames": tracking.n_frames}

    # -- ball ----------------------------------------------------------------------------------
    ball_xy, ball_conf = tracking.ball_xy, tracking.ball_conf
    ball_z, ball_source = tracking.ball_z, tracking.ball_source
    if settings["ball"]["enabled"]:
        s = settings["ball"]
        ball_xy, masks = ball.clean_ball(tracking.ball_xy, fps, s["vmax"], s["min_run"],
                                         s["interp_max_s"])
        removed = masks["spike"] | masks["fragment"]
        ball_conf = tracking.ball_conf.copy()
        ball_source = tracking.ball_source.copy()
        ball_z = tracking.ball_z.copy()
        ball_conf[removed], ball_source[removed], ball_z[removed] = 0, SRC_NONE, np.nan
        ball_conf[masks["interp"]] = s["interp_conf"]
        ball_source[masks["interp"]] = SRC_INTERP
        report["ball"] = {
            "present_before": int(tracking.ball_present.sum()),
            "spikes_removed": int(masks["spike"].sum()),
            "fragment_points_removed": int(masks["fragment"].sum()),
            "interpolated_frames": int(masks["interp"].sum()),
            "present_after": int((~np.isnan(ball_xy[:, 0])).sum()),
        }

    # -- players -------------------------------------------------------------------------------
    xy = tracking.players_xy.copy()
    conf = tracking.players_conf.copy()
    team = slot_teams(tracking)
    report["players"] = {"observations_before": int(tracking.players_present.sum())}

    if settings["frozen"]["enabled"]:
        s = settings["frozen"]
        frozen = players.frozen_mask(xy, fps, s["window_s"], s["max_disp_m"], s["field_min_speed"])
        if s["action"] == "drop":
            xy[frozen] = np.nan
            conf[frozen] = 0
        elif s["action"] == "flag":
            conf[frozen] = np.minimum(conf[frozen], s["conf"])
        else:
            raise ValueError("frozen.action must be 'flag' or 'drop'")
        per_slot = frozen.sum(axis=0)
        report["frozen"] = {
            "frames": int(frozen.sum()), "action": s["action"],
            "top_slots": {int(k) + 1: int(per_slot[k]) for k in np.argsort(-per_slot)[:3]
                          if per_slot[k]},
        }

    if settings["duplicates"]["enabled"]:
        s = settings["duplicates"]
        drop, stretches = players.duplicate_mask(xy, team, fps, s["max_dist_m"], s["min_s"])
        xy[drop] = np.nan
        conf[drop] = 0
        report["duplicates"] = {"stretches": stretches, "frames_dropped": int(drop.sum())}

    players_team = np.broadcast_to(team, xy.shape[:2])
    if settings["stitching"]["enabled"]:
        s = settings["stitching"]
        xy, conf, team, report["stitching"] = stitching.stitch(
            xy, conf, team, fps, s["max_gap_s"], s["vmax"], s["margin_m"], s["fill_max_s"],
            s["fill_conf"])
        players_team = np.broadcast_to(team, xy.shape[:2])

    report["players"]["observations_after"] = int((~np.isnan(xy[:, :, 0])).sum())
    cleaned = replace(tracking, players_xy=xy, players_conf=conf, players_team=players_team,
                      ball_xy=ball_xy, ball_z=ball_z, ball_conf=ball_conf,
                      ball_source=ball_source, meta=dict(tracking.meta, cleaned=True))
    return cleaned, report

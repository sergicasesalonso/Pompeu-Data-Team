"""A4: touch and control events on synthetic scenes, the possession state machine case by case,
and rules_v0 on the real match."""

import numpy as np
import pandas as pd
import pytest

from pompeu.events import touches
from pompeu.models import rules
from synthetic import FPS, make_tracking, static_players

FIRST_HALF = 20000


# -- events ------------------------------------------------------------------------------------

def redirect_scene(n=100):
    """Ball rolls along +x at 8 m/s and an away player at (50, 30) turns it 90 degrees at frame 50."""
    t = np.arange(n) / FPS
    ball = np.empty((n, 2))
    ball[:50] = np.column_stack((50 - 8 * (t[50] - t[:50]), np.full(50, 30.0)))
    ball[50:] = np.column_stack((np.full(n - 50, 50.0), 30 + 8 * (t[50:] - t[50])))
    players = static_players(n, [(50.0, 30.3), (20, 10), (80, 60), (20, 60)])
    return make_tracking(players, np.array([1, 0, 0, 1]), ball, FIRST_HALF)


def test_touch_at_the_redirect():
    events = touches.detect_events(redirect_scene())
    touched = events[events["kind"] == "touch"]
    assert len(touched) == 1
    row = touched.iloc[0]
    assert abs(row["frame"] - (FIRST_HALF + 50)) <= 2
    assert row["team"] == 1 and row["p_home"] < 0.1
    # Savitzky-Golay smooths over the turn, so the speeds either side are close to, not at, 8.
    assert row["speed_in"] == pytest.approx(8.0, abs=1.0)
    assert row["speed_out"] == pytest.approx(8.0, abs=1.0)


def test_no_touch_without_a_player_near():
    tr = redirect_scene()
    players = tr.players_xy.copy()
    players[:, 0] = (70.0, 60.0)  # the only player near the turn moves away: a ball cannot turn alone
    events = touches.detect_events(make_tracking(players, tr.players_team[0], tr.ball_xy, FIRST_HALF))
    assert (events["kind"] != "touch").all()


def test_control_when_a_player_keeps_the_ball():
    n = 100
    t = np.arange(n) / FPS
    carrier = np.column_stack((30 + 3 * t, np.full(n, 30.0)))
    players = static_players(n, [(0, 0), (80, 60), (20, 60), (90, 10)])
    players[:, 0] = carrier
    ball = carrier + (0.6, 0.0)  # dribbled 0.6 m ahead for 4 s
    events = touches.detect_events(make_tracking(players, np.array([0, 1, 1, 0]), ball, FIRST_HALF))
    control = events[events["kind"] == "control"]
    assert len(control) == 1
    assert control.iloc[0]["team"] == 0 and control.iloc[0]["end"] - control.iloc[0]["frame"] == n


# -- the state machine -------------------------------------------------------------------------

def event(frame, team, kind="touch", end=None, p=None, speed_in=5.0, speed_out=5.0):
    p_home = (0.95 if team == 0 else 0.05) if p is None else p
    return {"frame": frame, "end": end or frame + 1, "kind": kind, "slot": 0, "distance": 0.5,
            "p_home": p_home, "team": 0 if p_home >= 0.5 else 1,
            "speed_in": speed_in, "speed_out": speed_out}


def changes(rows, opening=0, nearest=None, **params):
    p = dict(rules.PARAMS, backdate="touch")
    p.update(params)
    rows = list(pd.DataFrame(rows).itertuples(index=False))
    return rules.possession_changes(rows, 0, 1000, opening, FPS, p, nearest)


def test_deflection_does_not_change_possession():
    assert changes([event(100, 0), event(200, 1), event(220, 0)]) == [(0, 0)]


def test_two_rival_touches_change_at_the_first():
    assert changes([event(100, 0), event(200, 1), event(240, 1)]) == [(0, 0), (200, 1)]


def test_rival_touches_too_far_apart_do_not_chain():
    assert changes([event(200, 1), event(200 + 4 * FPS, 1)]) == [(0, 0)]


def test_rival_control_dates_back_to_its_touch():
    assert changes([event(200, 1), event(230, 1, kind="control", end=260)]) == [(0, 0), (200, 1)]
    assert changes([event(230, 1, kind="control", end=260)]) == [(0, 0), (230, 1)]


def test_fly_by_and_uncertain_events_are_ignored():
    fly = [event(200, 1, speed_in=25, speed_out=22), event(220, 1, speed_in=24, speed_out=20)]
    assert changes(fly) == [(0, 0)]
    unsure = [event(200, 1, p=0.45), event(220, 1, p=0.45)]
    assert changes(unsure) == [(0, 0)]


def test_unknown_opening_takes_the_first_event():
    assert changes([event(300, 1), event(400, 1)], opening=-1) == [(0, 1)]


def test_streak_backdate_stops_at_own_last_event():
    nearest = np.full(1000, -1)
    nearest[150:200] = 1    # away nearest from 150 ...
    nearest[120:150] = 0
    rows = [event(100, 0, end=130), event(200, 1), event(240, 1)]
    assert changes(rows, nearest=nearest, backdate="streak") == [(0, 0), (150, 1)]
    nearest[120:150] = 1    # ... or from 120, but home's own event ended at 130
    assert changes(rows, nearest=nearest, backdate="streak") == [(0, 0), (130, 1)]


def test_unknown_params_rejected():
    with pytest.raises(KeyError, match="unknown rules params"):
        rules.predict(redirect_scene(), rules={"chain_gap": 2})


# -- end to end --------------------------------------------------------------------------------

def test_interception_end_to_end():
    """Home carries the ball for 4 s, then an away player takes it off him and carries it away."""
    n = 400
    t = np.arange(n) / FPS
    home = np.column_stack((30 + 3 * t, np.full(n, 30.0)))
    away = np.column_stack((np.full(n, 42.6), np.full(n, 30.5)))
    away[110:] = np.column_stack((42.6 - 3 * (t[110:] - t[110]), np.full(n - 110, 30.5)))
    ball = home + (0.6, 0.0)
    ball[100:] = away[100:] + (-0.6, 0.0)
    others = static_players(n, [(10, 10), (90, 60), (10, 60), (90, 10)])
    players = np.concatenate((home[:, None], away[:, None], others), axis=1)
    # Running players so the dead-ball detector sees live play.
    players[:, 2:, 0] += 4 * np.sin(2 * np.pi * t[:, None] / 6 + np.arange(4))
    tr = make_tracking(players, np.array([0, 1, 0, 1, 0, 1]), ball, FIRST_HALF)
    labels = rules.predict(tr)
    assert (labels[:90] == "H").all()
    assert (labels[110:] == "A").all()


def test_real_match(tracking):
    from pompeu.clean import clean_tracking
    import possession_metrics as pm
    clean, _ = clean_tracking(tracking)
    labels = rules.predict(clean)
    table = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    assert pm.validate_submission(table, 199319) is None
    graded = clean.half() > 0
    codes = pm.label_codes(labels)
    runs = pm.valid_runs(np.where(graded, codes, pm.X))
    stats = pm.possession_stats(codes, runs)
    assert 150 <= stats["n_spells"] <= 500
    assert not np.isin(labels[graded], ["X"]).any()


def test_labels_from_matches_predict():
    """hsmm_v1 reuses rules_v0 through labels_from(); it must be the very same decision."""
    from pompeu.events.deadball import detect
    from pompeu.features.build import build_features
    tr = redirect_scene(300)
    F = build_features(tr, with_windows=False,
                       groups=("ball", "proximity", "pitch_control", "collective", "context"))
    dead = detect(tr, F=F)
    events = touches.detect_events(tr, F)
    before = events.copy()
    np.testing.assert_array_equal(rules.labels_from(tr, dead, events), rules.predict(tr))
    pd.testing.assert_frame_equal(events, before)  # the events it was given are not modified

"""A2: features on synthetic scenes where the answer is known, then the full match."""

import time

import numpy as np
import pandas as pd
import pytest

from pompeu.features import build, context, pitch_control, proximity
from synthetic import FPS, make_tracking, static_players

FIRST_HALF = 20000    # a frame0 inside the 1st half: home attacks +x
SECOND_HALF = 120000  # inside the 2nd half: home attacks -x
TEAM = np.array([0, 0, 1, 1])


def features(tracking, tagging=None):
    return build.build_features(tracking, tagging=tagging)


def far_away(n):
    """Two home and two away players; the away pair far from everything at (90, 10) / (90, 60)."""
    return static_players(n, [(30, 30), (50, 30), (90, 10), (90, 60)])


# -- the three scenes the plan asks for --------------------------------------------------------

def test_ball_still_next_to_a_player():
    n = 100
    ball = np.tile([31.0, 30.0], (n, 1))     # 1 m from home player 0, for 4 s
    F = features(make_tracking(far_away(n), TEAM, ball, FIRST_HALF))
    late = F.iloc[50:]
    assert (late["ball_still"] == 1).all() and late["ball_still_s"].iloc[-1] == pytest.approx(4.0)
    assert late["ball_speed"].abs().max() < 1e-6
    assert (late["nearest_team"] == 0).all() and (late["n2_H"] == 1).all()
    assert (late["n10_A"] == 0).all()
    assert late["d1_diff"].iloc[0] == pytest.approx(1.0 - np.hypot(59, 20))
    assert (late["pc_ball_H"] > 0.99).all()
    assert late["near_streak_s_H"].iloc[-1] == pytest.approx(4.0)
    assert late["near_streak_fwd_s_H"].iloc[-1] == pytest.approx(1 / FPS)


def test_pass_between_teammates():
    """Ball from home player 0 at (30, 30) to home player 1 at (50, 30) at 15 m/s."""
    n = 60
    t = np.arange(n) / FPS
    ball = np.column_stack((np.clip(30.5 + 15 * t, None, 49.5), np.full(n, 30.0)))
    F = features(make_tracking(far_away(n), TEAM, ball, FIRST_HALF))
    flight = F.iloc[10:25]
    np.testing.assert_allclose(flight["ball_speed"], 15.0, atol=1e-6)
    assert flight["ball_turn_deg"].max() < 1.0              # a straight pass does not turn
    assert (F["nearest_team"].iloc[[0, -1]] == 0).all()
    assert F["ball_x_home"].iloc[-1] == pytest.approx(49.5)  # 1st half: home frame = ATD frame
    assert (F["ball_out"] == 0).all()


def test_ball_out_over_the_touchline():
    n = 50
    t = np.arange(n) / FPS
    ball = np.column_stack((np.full(n, 40.0), 60.0 + 10 * t))  # crosses y = 68 at t = 0.8 s
    F = features(make_tracking(far_away(n), TEAM, ball, FIRST_HALF))
    crossing = np.argmax(ball[:, 1] > 68.0 + 0.5)
    assert (F["ball_out"].iloc[:crossing] == 0).all()
    assert (F["ball_out"].iloc[crossing:] == 1).all()
    assert F["ball_to_touchline_min"].iloc[-1] == pytest.approx(68.0 - ball[-1, 1])
    assert F["ball_edge_dist"].iloc[-1] < 0


# -- the other groups --------------------------------------------------------------------------

def test_no_ball_gives_nan_and_gap_timing():
    n = 100
    ball = np.tile([31.0, 30.0], (n, 1))
    ball[40:60] = np.nan
    F = features(make_tracking(far_away(n), TEAM, ball, FIRST_HALF))
    gap = F.iloc[40:60]
    assert gap["d1_H"].isna().all() and gap["pc_ball_H"].isna().all()
    assert (gap["ball_present"] == 0).all()
    assert gap["ball_gap_age_s"].iloc[0] == pytest.approx(1 / FPS)
    assert gap["ball_gap_len_s"].iloc[0] == pytest.approx(21 / FPS)
    assert (gap["ball_x_home_last"] == 31.0).all()
    assert gap["field_speed_mean"].notna().all()  # collective needs no ball


def test_pitch_control_symmetric_and_monotone():
    targets = np.array([[50.0, 34.0]])
    tau_home = pitch_control.arrival_times(np.array([[[45.0, 34.0]]]), np.zeros((1, 1, 2)), targets)
    tau_away = pitch_control.arrival_times(np.array([[[55.0, 34.0]]]), np.zeros((1, 1, 2)), targets)
    assert pitch_control.control(tau_home, tau_away, np.zeros(1))[0] == pytest.approx(0.5)
    closer = pitch_control.arrival_times(np.array([[[48.0, 34.0]]]), np.zeros((1, 1, 2)), targets)
    assert pitch_control.control(closer, tau_away, np.zeros(1))[0] > 0.7
    # Running towards the ball beats standing at the same distance.
    running = pitch_control.arrival_times(np.array([[[55.0, 34.0]]]), np.array([[[-5.0, 0.0]]]),
                                          targets)
    assert pitch_control.control(tau_home, running, np.zeros(1))[0] < 0.4


def test_collective_reads_the_same_in_both_halves():
    """Home pushing forward at 3 m/s: +x in the 1st half, -x in the 2nd. Same features."""
    n = 100
    t = np.arange(n)[:, None] / FPS
    for frame0, direction in ((FIRST_HALF, 1.0), (SECOND_HALF, -1.0)):
        xy = far_away(n)
        xy[:, :2, 0] = xy[0, :2, 0] + direction * 3.0 * t
        F = features(make_tracking(xy, TEAM, np.full((n, 2), np.nan), frame0)).iloc[10:-10]
        np.testing.assert_allclose(F["cvx_H"], 3.0, atol=1e-6)
        assert (F["n_press_H"] == 2).all() and (F["n_recover_H"] == 0).all()
        np.testing.assert_allclose(F["cvx_A"], 0.0, atol=1e-9)
        assert (F["n_vis_H"] == 2).all()


def test_streaks_switch_with_the_nearest_team():
    n = 50
    ball = np.tile([31.0, 30.0], (n, 1))
    ball[25:] = (89.0, 10.0)  # jumps next to an away player at frame 25
    F = features(make_tracking(far_away(n), TEAM, ball, FIRST_HALF))
    assert F["nearest_team"].iloc[24] == 0 and F["nearest_team"].iloc[25] == 1
    assert F["near_streak_s_H"].iloc[24] == pytest.approx(25 / FPS)
    assert F["near_streak_s_H"].iloc[25] == 0
    assert F["near_streak_s_A"].iloc[25] == pytest.approx(1 / FPS)
    assert F["near_streak_fwd_s_H"].iloc[0] == pytest.approx(25 / FPS)


def test_tagging_votes():
    n = 300
    tagging = pd.DataFrame({
        "code": ["BUILD UP", "DEFENDING IN MIDDLE THIRD", "SET PIECES", "BALL IN THE BOX"],
        "start_frame": [FIRST_HALF + 100, FIRST_HALF + 150, FIRST_HALF + 260, FIRST_HALF + 0],
        "end_frame": [FIRST_HALF + 200, FIRST_HALF + 250, FIRST_HALF + 290, FIRST_HALF + 50],
        "team": [0, 1, -1, 1],
        "type": [None, None, "throw-in", None],
    })
    tr = make_tracking(far_away(n), TEAM, np.full((n, 2), np.nan), FIRST_HALF)
    F = features(tr, tagging)
    assert F["st_vote"].iloc[120] == 1     # home builds up
    assert F["st_vote"].iloc[170] == 2     # ... and away is defending: two votes for home
    assert F["st_vote"].iloc[220] == 1     # only the defending phase left
    assert F["st_zone_team"].iloc[10] == -1 and F["st_vote"].iloc[10] == 0
    assert F["sp_throw_in"].iloc[270] == 1 and F["sp_team"].iloc[270] == 0
    assert F["sp_next_s"].iloc[0] == pytest.approx(260 / FPS)
    assert F["st_prev_vote"].iloc[180] == 1   # last phase started was the defending one, away
    assert F["half"].iloc[0] == 1 and F["minute"].iloc[0] > 0

    blank = features(tr, None)
    assert blank[list(context.TAGGING_COLUMNS)].isna().all().all()
    assert list(blank.columns) == list(F.columns)  # same schema with or without tagging


def test_windows_are_centred():
    n = 100
    t = np.arange(n) / FPS
    ball = np.column_stack((30.5 + np.where(t < 2, 0.0, 10 * (t - 2)), np.full(n, 30.0)))
    F = features(make_tracking(far_away(n), TEAM, ball, FIRST_HALF))
    # 0.5 s window = 12 frames centred: just before the kick it already sees the ball moving.
    assert F["ball_speed_w0.5_max"].iloc[46] > 5 and F["ball_speed"].iloc[44] < 1


# -- the full match ----------------------------------------------------------------------------

def test_full_match(tracking):
    from pompeu.clean import clean_tracking
    clean, _ = clean_tracking(tracking)
    started = time.time()
    F = build.build_features(clean)
    elapsed = time.time() - started
    assert elapsed < 300                       # the plan's budget: under 5 minutes on CPU
    assert F.shape[0] == 199319 and F.shape[1] >= 190
    assert (F.index == np.arange(199319)).all()
    graded = F["half"].to_numpy() > 0
    assert not F[graded].isna().all().any()    # no column is empty on graded frames
    assert set(F.attrs["groups"].values()) == {"ball", "proximity", "pitch_control",
                                               "collective", "context", "windows"}
    assert F["ball_present"][graded].mean() == pytest.approx(0.592, abs=0.01)
    assert proximity.RADII == (2, 5, 10)

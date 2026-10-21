"""Stage 0.2: the fast ATD loader reads exactly what is in the file, on the video-frame index."""

import numpy as np
import pytest

import games
from pompeu.io.atd import load_atd, parse_raw

N_FRAMES = 199319
N_SLOTS = 46
BALL_PRESENT = 113488  # DATA.md: 56.9% of all frames
BALL_PRESENT_GRADED = 94214  # DATA.md: 59.3% of graded frames


def test_shape(atd):
    assert atd.P.shape == (N_FRAMES, N_SLOTS, 3)
    assert atd.B.shape == (N_FRAMES, 3)
    assert atd.slot_team.shape == (N_SLOTS,)


def test_teams(atd):
    assert atd.team_names == ("Barcelona", "Atletic Club")
    assert int((atd.slot_team == 0).sum()) == 24
    assert int((atd.slot_team == 1).sum()) == 22


def test_ball_presence(atd):
    present = ~np.isnan(atd.B[:, 0])
    assert int(present.sum()) == BALL_PRESENT
    assert round(100 * present.mean(), 1) == 56.9
    graded = np.zeros(N_FRAMES, dtype=bool)
    for _, first, last in games.half_video_blocks(games.GAME):
        graded[first:last + 1] = True
    assert int(present[graded].sum()) == BALL_PRESENT_GRADED


@pytest.mark.parametrize("frame_id", [1, 11190, 113743, N_FRAMES])
def test_frame_matches_text(atd, frame_id):
    """Line `frame_id` of the raw file is row frame_id - 1, value for value, in metres."""
    raw_path = games.require(["atd_raw"])[0]
    with open(raw_path) as handle:
        for index, line in enumerate(handle, start=1):
            if index == frame_id:
                break
    head, players, ball = line.strip().split(":")
    assert int(head) == frame_id
    length, width = games.GAMES[games.GAME]["pitch"]
    scale = np.array([length, width, 1.0])
    expected = np.array([[float(v) for v in group.split(",")]
                         for group in players.split(";")]) * scale
    expected_ball = np.array([float(v) for v in ball.split(",")]) * scale
    row = frame_id - 1
    np.testing.assert_array_equal(atd.P[row], expected)
    np.testing.assert_array_equal(atd.B[row], expected_ball)


def test_attack_direction(atd):
    """Barcelona (home) attacks left to right in the 1st half: at each kickoff the home side sits
    on the half it defends -- left (x < L/2) in the 1st half, right in the 2nd."""
    length = games.GAMES[games.GAME]["pitch"][0]
    assert atd.home_attack_sign == 1
    for kickoff, home_on_left in ((11185, True), (113742, False)):
        home_x = atd.P[kickoff, atd.slot_team == 0, 0]
        assert (np.nanmedian(home_x) < length / 2) == home_on_left


def test_cache_roundtrip(atd, tmp_path):
    fresh = load_atd(cache=True, cache_dir=str(tmp_path))
    cached = load_atd(cache=True, cache_dir=str(tmp_path))
    assert len(list(tmp_path.iterdir())) == 1
    np.testing.assert_array_equal(fresh.P, cached.P)
    np.testing.assert_array_equal(cached.P, atd.P)
    np.testing.assert_array_equal(cached.B, atd.B)


def test_rejects_wrong_column_count(tmp_path):
    raw = tmp_path / "bad.txt"
    raw.write_text("1:0.1,0.2,0.3;0.1,0.2,0.3:0.5,0.5,0\n")
    with pytest.raises(ValueError, match="columns"):
        parse_raw(str(raw), n_slots=3, pitch=(105.3, 68.0))

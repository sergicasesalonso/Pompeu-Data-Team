"""SkillCorner Open Data loader on a tiny synthetic match (no real data is needed or committed)."""

import json

import numpy as np
import pytest

from pompeu import external
from pompeu.external import skillcorner as sc

H, A, D, X = 0, 1, 2, 3
HOME, AWAY = 10, 20
MID = 999


def write_match(root, n=60, periods=((1, 2, 29), (2, 32, 59)), phases=None):
    """Synthetic match: 2 players per team, the home ones moving 1 m per source frame along x."""
    folder = root / "matches" / str(MID)
    folder.mkdir(parents=True)
    players = [{"id": pid, "team_id": team} for pid, team in
               ((1, HOME), (2, HOME), (3, AWAY), (4, AWAY))]
    meta = {"id": MID, "home_team": {"id": HOME, "short_name": "Home"},
            "away_team": {"id": AWAY, "short_name": "Away"}, "players": players,
            "pitch_length": 104, "pitch_width": 68,
            "home_team_side": ["right_to_left", "left_to_right"]}
    (folder / "{}_match.json".format(MID)).write_text(json.dumps(meta), encoding="utf-8")
    period_of = {}
    for half, first, last in periods:
        for f in range(first, last + 1):
            period_of[f] = half
    with open(folder / "{}_tracking_extrapolated.jsonl".format(MID), "w", encoding="utf-8") as fh:
        for f in range(n):
            empty = f in (10, 11)          # a replay: no tracking at all
            pdata = [] if empty else [
                {"player_id": 1, "x": float(f), "y": 0.0, "is_detected": True},
                {"player_id": 2, "x": -10.0, "y": 5.0, "is_detected": f % 2 == 0},
                {"player_id": 3, "x": 10.0, "y": -5.0, "is_detected": True}]
            ball = {"x": None, "y": None, "z": None, "is_detected": None} if empty else \
                {"x": 0.5 * f, "y": 1.0, "z": 0.2, "is_detected": True}
            fh.write(json.dumps({"frame": f, "period": period_of.get(f), "ball_data": ball,
                                 "possession": {"player_id": None, "group": None},
                                 "player_data": pdata}) + "\n")
    if phases is None:
        phases = [(5, 12, 1, HOME), (12, 20, 1, AWAY), (24, 31, 1, AWAY),   # runs past period 1
                  (35, 50, 2, HOME)]
    lines = ["frame_start,frame_end,period,team_in_possession_id"]
    lines += ["{},{},{},{}".format(*p) for p in phases]
    (folder / "{}_phases_of_play.csv".format(MID)).write_text("\n".join(lines) + "\n",
                                                              encoding="utf-8")
    return root


def test_labels_from_phases_rules():
    periods = [(1, 2, 29), (2, 32, 59)]
    phases = [(5, 12, 1, HOME), (12, 20, 1, AWAY), (24, 31, 1, AWAY), (35, 50, 2, HOME)]
    lab = sc.labels_from_phases(60, periods, phases, HOME)
    assert (lab[:5] == X).all()                 # before the first phase of the period: unknown
    assert (lab[5:12] == H).all() and (lab[12:20] == A).all()   # touching phases: no stoppage
    assert (lab[20:24] == D).all()              # gap between phases: dead ball
    assert (lab[24:30] == A).all()              # clipped to period 1 (last frame 29)
    assert (lab[30:35] == X).all()              # half-time and before the second half's phase
    assert (lab[35:50] == H).all() and (lab[50:] == X).all()


def test_upsample_index_is_exact():
    lo, w = sc.upsample_index(11)               # 1 s at 10 fps -> 0..1 s at 25 fps
    assert len(lo) == 26 and lo[-1] == 10 and w[-1] == 0
    np.testing.assert_allclose(lo + w, np.arange(26) * 0.4)
    values = np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0])
    lo, w = sc.upsample_index(6)
    out = sc.resample(values, lo, w)
    np.testing.assert_allclose(out[:3], [0.0, 0.4, 0.8])
    assert np.isnan(out[3:8]).all()               # source frame 2 is missing: NaN around it
    np.testing.assert_allclose(out[8:], [3.2, 3.6, 4.0, 4.4, 4.8])


def test_convert_synthetic_match(tmp_path):
    root = write_match(tmp_path)
    assert sc.available(MID, str(root)) and sc.match_ids(str(root)) == [MID]
    tr, labels = sc.convert(MID, str(root))
    assert tr.meta["fps"] == 25 and tr.n_frames == (5 * 59) // 2 + 1 == len(labels)
    assert tr.meta["home_attack_sign"] == -1 and tr.attack_sign(0, 2) == 1
    # Halves: the 25 fps rows whose source frame lies in the period.
    assert tr.meta["halves"] == [(1, 5, 74), (2, 80, 147)]
    # Metres from the corner, interpolated: player 1 at source x = f, row j at f = 0.4 j.
    j = 50
    np.testing.assert_allclose(tr.players_xy[j, 0], (0.4 * j + 52.0, 34.0))
    np.testing.assert_allclose(tr.ball_xy[j], (0.2 * j + 52.0, 35.0))
    assert (tr.players_team[0] == [0, 0, 1, 1]).all()
    # Player 4 never appears; player 2 is detected one frame in two.
    assert not tr.players_present[:, 3].any()
    assert set(np.unique(tr.players_conf[tr.players_present[:, 1], 1])) == {0.5, 1.0}
    # The replay frames 10-11 (rows 25-29, plus the rows interpolating into them) have nothing.
    assert not tr.players_present[25:30].any() and not tr.ball_present[25:30].any()
    # Labels follow the source frame (2j)//5.
    lo, _ = sc.upsample_index(60)
    np.testing.assert_array_equal(labels, sc.labels_from_phases(
        60, [(1, 2, 29), (2, 32, 59)],
        sc.read_phases(MID, str(root)), HOME)[lo])
    assert labels[(5 * 5 + 1) // 2] == H and labels[(5 * 20 + 1) // 2] == D


def test_convert_is_deterministic(tmp_path):
    root = write_match(tmp_path)
    a, la = sc.convert(MID, str(root))
    b, lb = sc.convert(MID, str(root))
    np.testing.assert_array_equal(la, lb)
    np.testing.assert_array_equal(a.players_xy, b.players_xy)
    np.testing.assert_array_equal(a.ball_xy, b.ball_xy)


def test_registry_keeps_skillcorner_out_of_the_splits():
    assert not any(name.startswith("skillcorner:") for name in external.ALL)
    for split in external.SPLITS.values():
        assert not any(name.startswith("skillcorner:") for name in split)
    assert len(external.EXTRA_TRAIN) == len(set(external.EXTRA_TRAIN)) == 20
    module, key = external.parse("skillcorner:1886347")
    assert module is sc and key == 1886347
    assert set(external.resolve("extra_train")) <= set(external.EXTRA_TRAIN)


@pytest.mark.skipif(not sc.match_ids(), reason="SkillCorner open data not cloned")
def test_real_match_converts():
    match = sc.match_ids()[0]
    tr, labels = sc.load(match)
    assert tr.meta["fps"] == 25 and len(labels) == tr.n_frames
    graded = tr.half() > 0
    live = np.isin(labels[graded], (H, A)).mean()
    assert 0.4 < live < 0.75

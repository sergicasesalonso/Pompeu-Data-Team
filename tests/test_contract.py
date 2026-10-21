"""Stage 0.3: the Tracking contract holds its invariants, and the ATD satisfies it."""

import numpy as np
import pytest

from pompeu.io.contract import (BALL_SOURCES, SRC_ATD, SRC_NONE, Tracking, load_tracking)


def synthetic(n=10, k=4):
    rng = np.random.default_rng(0)
    players_xy = rng.uniform(0, 60, (n, k, 2))
    players_xy[0, 1] = np.nan
    ball_xy = rng.uniform(0, 60, (n, 2))
    ball_xy[3] = np.nan
    ball = ~np.isnan(ball_xy[:, 0])
    return Tracking(
        players_xy=players_xy,
        players_team=np.tile([0, 0, 1, 1], (n, 1)),
        players_conf=(~np.isnan(players_xy[:, :, 0])).astype(float),
        ball_xy=ball_xy,
        ball_z=np.full(n, np.nan),
        ball_conf=ball.astype(float),
        ball_source=np.where(ball, SRC_ATD, SRC_NONE),
        frame0=11180,
        meta={"game": "BAR-ATH", "fps": 25, "home_attack_sign": 1},
    )


def test_valid_synthetic():
    tr = synthetic()
    assert (tr.n_frames, tr.n_slots) == (10, 4)
    assert tr.ball_present.sum() == 9
    assert tr.players_xy.dtype == np.float64 and tr.players_team.dtype == np.int8


@pytest.mark.parametrize("breakage, message", [
    (lambda f: f.update(ball_z=np.zeros(3)), "ball_z has shape"),
    (lambda f: f.update(players_team=np.full_like(f["players_team"], 2)), "players_team"),
    (lambda f: f["players_conf"].__setitem__((0, 1), 1.0), "players_conf must be 0"),
    (lambda f: f["ball_source"].__setitem__(3, SRC_ATD), "ball_source must be 'none'"),
    (lambda f: f["ball_source"].__setitem__(0, SRC_NONE), "ball_source must be 'none'"),
    (lambda f: f["ball_conf"].__setitem__(0, 1.5), r"ball_conf must be in \[0, 1\]"),
    (lambda f: f["players_xy"].__setitem__((2, 0, 1), np.nan), "x without y"),
])
def test_invariants_enforced(breakage, message):
    tr = synthetic()
    fields = {name: getattr(tr, name).copy() for name in (
        "players_xy", "players_team", "players_conf", "ball_xy", "ball_z", "ball_conf",
        "ball_source")}
    breakage(fields)
    with pytest.raises(ValueError, match=message):
        Tracking(frame0=tr.frame0, meta=tr.meta, **fields)


def test_crop_keeps_absolute_frames():
    tr = synthetic()
    crop = tr.crop(11183, 11186)
    assert crop.frame0 == 11183 and crop.n_frames == 3
    np.testing.assert_array_equal(crop.ball_xy, tr.ball_xy[3:6])
    assert tr.crop(0, 11182).n_frames == 2      # clipped at the start
    assert tr.crop(11188, 99999).n_frames == 2  # clipped at the end


def test_half_and_attack_sign():
    tr = synthetic()
    np.testing.assert_array_equal(tr.half(), [0] * 5 + [1] * 5)  # 1st half starts at 11185
    assert tr.attack_sign(0, 1) == 1 and tr.attack_sign(1, 1) == -1
    assert tr.attack_sign(0, 2) == -1 and tr.attack_sign(1, 2) == 1


def test_without_ball():
    tr = synthetic()
    mask = np.zeros(tr.n_frames, dtype=bool)
    mask[:2] = True
    out = tr.without_ball(mask)
    assert not out.ball_present[:2].any()
    assert (out.ball_source[:2] == SRC_NONE).all() and (out.ball_conf[:2] == 0).all()
    assert tr.ball_present[:2].all()  # the original is untouched
    np.testing.assert_array_equal(out.players_xy, tr.players_xy)


def test_save_load_roundtrip(tmp_path):
    tr = synthetic()
    path = tr.save(str(tmp_path / "t.npz"))
    back = load_tracking(path)
    assert back.frame0 == tr.frame0 and back.meta == tr.meta
    for name in ("players_xy", "players_team", "players_conf", "ball_xy", "ball_z",
                 "ball_conf", "ball_source"):
        np.testing.assert_array_equal(getattr(back, name), getattr(tr, name))


def test_atd_satisfies_contract(tracking, atd):
    assert tracking.n_frames == 199319 and tracking.n_slots == 46 and tracking.frame0 == 0
    assert (tracking.players_team == atd.slot_team[None, :]).all()
    assert set(np.unique(tracking.ball_source)) == {SRC_NONE, SRC_ATD}
    assert np.isnan(tracking.ball_z).all()
    assert BALL_SOURCES[SRC_ATD] == "atd"
    counts = np.bincount(tracking.half(), minlength=3)
    assert counts.tolist() == [40462, 79467, 79390]  # DATA.md: ungraded, 1st half, 2nd half

"""A6: convention alignment, the degradation simulator, the DFL conversion and the external bench."""

import numpy as np
import pytest

from pompeu.external import degrade, dfl
from pompeu.validate import harness
from synthetic import make_tracking

H, A, D, X = 0, 1, 2, 3
FPS = 25


def labels_of(*runs):
    """Concatenate (label, frames) runs."""
    return np.concatenate([np.full(n, label, dtype=np.int8) for label, n in runs])


# -- convention alignment ----------------------------------------------------------------------

def test_short_sandwiched_possession_goes_to_the_surrounding_team():
    lab = labels_of((X, 1), (H, 100), (A, 30), (H, 100), (X, 1))
    out = dfl.align_labels(lab, 2.0)
    assert (out[1:-1] == H).all()


def test_long_or_unsandwiched_possessions_stay():
    lab = labels_of((X, 1), (H, 100), (A, 60), (H, 100), (X, 1))
    np.testing.assert_array_equal(dfl.align_labels(lab, 2.0), lab)       # 2.4 s >= 2 s
    lab = labels_of((X, 1), (H, 100), (A, 30), (D, 50), (A, 100), (X, 1))
    np.testing.assert_array_equal(dfl.align_labels(lab, 2.0), lab)       # ends in a stoppage
    lab = labels_of((X, 1), (A, 30), (H, 100), (X, 1))
    np.testing.assert_array_equal(dfl.align_labels(lab, 2.0), lab)       # first of the half


def test_alignment_never_crosses_halves_and_iterates():
    lab = labels_of((X, 1), (H, 100), (A, 10), (H, 10), (A, 10), (H, 100), (X, 5), (A, 10),
                    (H, 100), (X, 1))
    out = dfl.align_labels(lab, 2.0)
    assert (out[1:231] == H).all()        # A-H-A of 0.4 s pieces collapse into home
    assert (out[236:246] == A).all()      # the 2nd half's opening A is not touched


# -- degradation -------------------------------------------------------------------------------

def clean_match(n=6000, seed=0):
    """A clean synthetic 'external' match: 22 players always seen, ball always seen, two halves."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    start = rng.uniform(10, 95, (22, 2)) * (1.0, 0.6)
    players = start[None] + 3 * np.sin(t[:, None, None] / 3 + rng.uniform(0, 6, (1, 22, 2)))
    ball = players[:, 0] + 0.5
    team = np.array([0] * 11 + [1] * 11)
    tr = make_tracking(players, team, ball, frame0=0)
    tr.meta.update(halves=[(1, 0, n // 2 - 1), (2, n // 2, n - 1)], source="test")
    labels = np.where((t % 60) < 20, D, H).astype(np.int8)   # dead 1/3 of the time
    return tr, labels


def test_degrade_hits_the_atd_statistics():
    tr, labels = clean_match()
    run_lengths = np.array([25, 100, 250, 500])
    out = degrade.degrade(tr, labels, seed=0, run_lengths=run_lengths)
    out.validate()
    missing = 1 - out.ball_present.mean()
    assert abs(missing - 0.41) < 0.02
    dead = labels == D
    assert (1 - out.ball_present[dead].mean()) > (1 - out.ball_present[~dead].mean())
    visible = out.players_present.sum(axis=1).mean() / 2
    assert 9.5 < visible < 10.8                                   # ~10.2 of 11 per team
    assert out.n_slots < 40                                       # slots are reused, not one each
    assert set(np.unique(out.players_team)) == {0, 1}


def test_degrade_is_seeded():
    tr, labels = clean_match()
    lengths = np.array([25, 100, 250])
    a = degrade.degrade(tr, labels, 3, lengths)
    b = degrade.degrade(tr, labels, 3, lengths)
    c = degrade.degrade(tr, labels, 4, lengths)
    np.testing.assert_array_equal(a.ball_xy, b.ball_xy)
    np.testing.assert_array_equal(a.players_xy, b.players_xy)
    assert not np.array_equal(a.ball_present, c.ball_present)


def test_degrade_unknown_params_rejected():
    tr, labels = clean_match(500)
    with pytest.raises(KeyError, match="unknown degrade params"):
        degrade.degrade(tr, labels, 0, np.array([25]), {"ball_missin": 0.3})


# -- the external bench ------------------------------------------------------------------------

def test_bench_off_and_match_listing():
    cfg = harness.load_config(harness.paths.CONFIGS_DIR + "/rules_v0.yaml")
    cfg["validation"]["external"]["datasets"] = []
    assert harness.bench_external(cfg, np.array([25]), 0)["status"].startswith("pending")
    assert harness.external_matches(["J03WPY"]) == ["J03WPY"]
    assert set(harness.external_matches("auto")) <= set(dfl.MATCHES)


# -- the real conversion (needs scripts/prepare_external.py) ----------------------------------

def _have(match_id):
    try:
        dfl.files(match_id)
        return True
    except IOError:
        return False


@pytest.mark.skipif(not _have("J03WPY"), reason="IDSSE match J03WPY not downloaded")
def test_dfl_conversion():
    tr, labels = dfl.load("J03WPY")
    halves = tr.meta["halves"]
    assert [h for h, _, _ in halves] == [1, 2]
    assert halves[0][1] == 0 and halves[-1][2] == tr.n_frames - 1
    assert len(labels) == tr.n_frames and not (labels == X).any()
    assert 0.35 < np.mean(labels == D) < 0.5
    assert tr.meta["pitch"] == (105.0, 68.0)
    # Home attacks the way meta says: at the first kick-off home stands in its own half.
    home_x = np.nanmedian(tr.players_xy[0, tr.players_team[0] == 0, 0])
    assert (home_x < 52.5) == (tr.meta["home_attack_sign"] == 1)
    assert set(np.unique(labels)) == {H, A, D}
    assert tr.ball_present.mean() > 0.99            # clean data: the simulator removes the ball

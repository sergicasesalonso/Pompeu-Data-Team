"""Stage 0.4: stress masks, intervals and the harness."""

import numpy as np
import pytest

from pompeu import paths
from pompeu.io.gt import load_gt_codes
from pompeu.validate import bootstrap, harness, stress
import possession_metrics as pm


# -- stress ------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", [0.3, 0.5, 0.7])
def test_stress_mask_hits_target_exactly(target):
    present = np.ones(7500, dtype=bool)
    present[100:900] = False  # ~11% already missing, as in the public window
    erase = stress.stress_mask(present, target, np.array([30, 270, 500, 2600]),
                               np.random.default_rng(0))
    assert not (erase & ~present).any()  # never "erases" what is already missing
    assert int((~present | erase).sum()) == round(target * len(present))


def test_stress_mask_nothing_when_already_above():
    present = np.zeros(100, dtype=bool)
    assert not stress.stress_mask(present, 0.5, np.array([10]), np.random.default_rng(0)).any()


def test_stressed_trackings_only_touch_window_and_are_seeded(tracking):
    window = (11185, 18685)
    crop = tracking.crop(window[0] - 1500, window[1] + 1500)
    lengths = stress.missing_run_lengths(tracking)
    assert 10.0 < np.median(lengths) / 25 < 11.5  # the measured ~10.7 s

    def masks():
        return [(level, seed, achieved, s.ball_present.copy()) for level, seed, achieved, s in
                stress.stressed_trackings(crop, window, (0.5,), 3, run_lengths=lengths)]

    first, second = masks(), masks()
    for (level, seed, achieved, present), (_, _, _, again) in zip(first, second):
        np.testing.assert_array_equal(present, again)  # same seed, same mask
        outside = np.ones(crop.n_frames, dtype=bool)
        outside[1500:-1500] = False
        np.testing.assert_array_equal(present[outside], crop.ball_present[outside])
        assert abs(achieved - 0.5) < 1e-3
    assert not np.array_equal(first[0][3], first[1][3])  # different seeds, different masks


def test_missing_run_lengths_excludes_half_edges():
    from pompeu.io.contract import Tracking, SRC_ATD, SRC_NONE
    import games
    n = games.GAMES[games.GAME]["n_video_frames"]
    ball_xy = np.zeros((n, 2))
    ball_xy[11185:11200] = np.nan   # touches the 1st-half start: dropped
    ball_xy[20000:20250] = np.nan   # inside: kept
    ball = ~np.isnan(ball_xy[:, 0])
    tr = Tracking(players_xy=np.zeros((n, 1, 2)), players_team=np.zeros((n, 1)),
                  players_conf=np.ones((n, 1)), ball_xy=ball_xy, ball_z=np.full(n, np.nan),
                  ball_conf=ball.astype(float), ball_source=np.where(ball, SRC_ATD, SRC_NONE),
                  meta={"game": games.GAME, "fps": 25})
    assert stress.missing_run_lengths(tr).tolist() == [250]


# -- bootstrap ---------------------------------------------------------------------------------

def _public_pred(name="baseline_smooth50"):
    import os
    return pm.label_codes(pm.read_label_csv(os.path.join(
        paths.EVALUATION_DIR, "results", name + ".csv")))


def test_per_draw_formulas_equal_the_metric_without_resampling():
    gt = load_gt_codes()
    runs = pm.valid_runs(gt)
    for name in ("baseline_raw", "baseline_smooth50", "baseline_smooth125"):
        pred = _public_pred(name)
        official = pm.evaluate(gt, pred)
        flags, n_pred = bootstrap._spell_tables(gt, pred, runs)
        assert bootstrap.spell_f1_from_flags(flags, n_pred, np.arange(flags.shape[1])) == \
            pytest.approx(official["spell_f1_mean"], abs=1e-12)
        gt_of_segment, contributions = bootstrap._segment_tables(gt, pred, runs)
        assert bootstrap.macro_f1_from_segments(gt_of_segment, contributions,
                                                np.arange(len(gt_of_segment))) == \
            pytest.approx(official["macro_f1"], abs=1e-12)


def test_single_prediction_interval_is_the_official_one():
    gt, pred = load_gt_codes(), _public_pred()
    official = pm.evaluate(gt, pred, bootstrap_n=300)
    ours = bootstrap.intervals(gt, pred, 300)
    assert ours["spell"] == (official["spell_bootstrap_low"], official["spell_bootstrap_high"])
    assert ours["macro"] == (official["bootstrap_low"], official["bootstrap_high"])


def test_identical_seeds_interval_contains_point():
    gt, pred = load_gt_codes(), _public_pred()
    ci = bootstrap.intervals(gt, [pred, pred, pred], 300)
    point = pm.evaluate(gt, pred)
    assert ci["spell"][0] <= point["spell_f1_mean"] <= ci["spell"][1]
    assert ci["macro"][0] <= point["macro_f1"] <= ci["macro"][1]


# -- harness -----------------------------------------------------------------------------------

def test_harness_runs_and_matches_published_public_score(tracking, tmp_path):
    cfg = harness.load_config(paths.CONFIGS_DIR + "/baseline_smooth50.yaml")
    cfg["validation"]["stress"]["seeds"] = 2
    cfg["validation"]["bootstrap_n"] = 50
    result, row = harness.run(cfg, tracking, save=False, verbose=False)
    # README: baseline_smooth50 scores spellF1 0.3256 and macroF1 0.6207 on the public window.
    assert round(result["public_clean"]["spell"], 4) == 0.3256
    assert round(result["public_clean"]["macro"], 4) == 0.6207
    assert set(result["public_stress"]) == {0.3, 0.5, 0.7}
    assert result["external"]["status"].startswith("pending")
    assert result["full_match_sanity"]["x_on_graded"] == 0
    assert row["config_hash"] == harness.pipeline.config_hash(cfg)

    path = str(tmp_path / "leaderboard.csv")
    harness.append_leaderboard(row, path)
    harness.append_leaderboard(dict(row, extra=1), path)
    import pandas as pd
    board = pd.read_csv(path)
    assert len(board) == 2 and "extra" in board.columns

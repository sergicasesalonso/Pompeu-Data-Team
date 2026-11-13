"""scripts/min_possession_sensitivity.py: possession counts, ranks, ranking-change detection."""

import importlib.util
import os

import numpy as np

from pompeu import external

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts",
                     "min_possession_sensitivity.py")
_spec = importlib.util.spec_from_file_location("min_possession_sensitivity", _PATH)
mps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mps)

H, A, D, X = 0, 1, 2, 3


def test_n_possessions_counts_team_segments_and_merging_lowers_it():
    # H 10 s, A 1 s, H 10 s, D, A 10 s | X | H 10 s
    codes = np.array([H] * 250 + [A] * 25 + [H] * 250 + [D] * 50 + [A] * 250 + [X] + [H] * 250,
                     dtype=np.int8)
    assert mps.n_possessions(codes) == 5
    assert mps.n_possessions(external.align_labels(codes, 0.0)) == 5
    assert mps.n_possessions(external.align_labels(codes, 2.0)) == 3


def test_ranks_best_is_one_and_ties_share():
    assert mps.ranks({"a": 0.5, "b": 0.7, "c": 0.5}) == {"a": 2, "b": 1, "c": 2}


def test_ranking_changes():
    same = {0.0: {"a": 0.6, "b": 0.5}, 5.0: {"a": 0.7, "b": 0.4}}
    flip = {0.0: {"a": 0.6, "b": 0.5}, 5.0: {"a": 0.4, "b": 0.5}}
    assert mps.ranking_changes(same)[1] is False
    orders, changed = mps.ranking_changes(flip)
    assert changed and orders[5.0] == ("b", "a")


def test_means_average_over_units():
    rows = [dict(value=0.0, config="a", spell=0.2), dict(value=0.0, config="a", spell=0.4)]
    assert abs(mps.means(rows, "spell")[0.0]["a"] - 0.3) < 1e-12

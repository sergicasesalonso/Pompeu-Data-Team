import numpy as np
import pytest

from pompeu.vision.ball_det import decode_heatmaps, make_heatmap


def test_heatmap_and_candidate_contract():
    heat = make_heatmap(24, 32, 10, 12)
    table = decode_heatmaps(heat[None], [42], top_k=1)
    assert list(table.columns) == ["frame", "x_px", "y_px", "conf"]
    assert table.iloc[0].tolist() == [42, 10, 12, 1]


def test_empty_multiple_and_deterministic_ties():
    maps = np.zeros((2, 20, 20), np.float32)
    maps[1, 2, 3] = maps[1, 15, 16] = 0.9
    first = decode_heatmaps(maps, [5, 6])
    assert len(first) == 2
    assert first.frame.tolist() == [6, 6]
    assert first.x_px.tolist() == [3, 16]
    assert first.equals(decode_heatmaps(maps, [5, 6]))
    assert decode_heatmaps(maps[:1], [5]).empty


def test_invalid_probabilities():
    with pytest.raises(ValueError):
        decode_heatmaps(np.full((1, 2, 2), np.nan), [0])
    with pytest.raises(ValueError):
        make_heatmap(10, 10, -1, 2)

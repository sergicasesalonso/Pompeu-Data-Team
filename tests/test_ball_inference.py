import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="separate B2 CPU environment required")
from pompeu.vision.ball_det import detect_ball  # noqa: E402
from pompeu.vision.ball_det.checkpoint import save_checkpoint  # noqa: E402
from pompeu.vision.ball_det.model import TemporalHeatmapNet, heatmap_targets, set_deterministic  # noqa: E402
from pompeu.vision.ball_det.predict import BallDetector  # noqa: E402
from pompeu.vision.ball_det.train import evaluate  # noqa: E402


class BrightPoint(torch.nn.Module):
    config = {"n_frames": 3}

    def forward(self, clips):
        return clips[:, 1, :1]*20-10


def test_tiled_inference_absolute_pixels_overlap_and_gaps():
    set_deterministic()
    video = np.zeros((3, 20, 28, 3), np.uint8)
    video[:, 10, 12] = 255
    detector = BallDetector(BrightPoint(), crop_shape=(16, 20), overlap=8, batch_size=1)
    result = detector.detect_ball([(100+i, image) for i, image in enumerate(video)])
    assert result.iloc[0].tolist() == [101, 12, 10, 1]
    assert len(result) == 1
    assert result.equals(detector.detect_ball([(100+i, image) for i, image in enumerate(video)]))
    assert detector.detect_ball([(100, video[0]), (102, video[1]), (103, video[2])]).empty
    assert detector.detect_ball(video*0).empty
    with pytest.raises(ValueError):
        detector.detect_ball([(100, video[0]), (100, video[1])])


def test_public_entry_point_uses_checkpoint_and_cpu(tmp_path):
    set_deterministic()
    path = tmp_path/"model.pt"
    digest = save_checkpoint(TemporalHeatmapNet(width=4), path, {"synthetic": True})
    result = detect_ball(np.zeros((3, 16, 20, 3), np.uint8), checkpoint=path,
                         expected_sha256=digest, crop_shape=(16, 20), overlap=4)
    assert list(result.columns) == ["frame", "x_px", "y_px", "conf"]
    assert result.empty


def test_validation_pixel_tolerances_are_distinct():
    set_deterministic()
    clips = torch.zeros(1, 3, 3, 24, 24)
    clips[:, :, :, 8, 12] = 1
    batch = {"clip": clips, "target": heatmap_targets([[5, 8]], 24, 24),
             "point": torch.tensor([[5, 8]]), "weight": torch.ones(1)}
    metrics = evaluate(BrightPoint(), [batch])
    assert metrics["crop_pseudo_precision_5px"] == metrics["crop_pseudo_recall_5px"] == 0
    assert metrics["crop_pseudo_precision_10px"] == metrics["crop_pseudo_recall_10px"] == 1


def test_constructor_does_not_change_threads_or_random_state():
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        state = torch.get_rng_state()
        deterministic = torch.are_deterministic_algorithms_enabled()
        BallDetector(BrightPoint())
        assert torch.get_num_threads() == 2
        assert torch.equal(state, torch.get_rng_state())
        assert deterministic == torch.are_deterministic_algorithms_enabled()
    finally:
        torch.set_num_threads(old_threads)


def test_canonical_api_requires_hash_and_rejects_cuda(tmp_path):
    path = tmp_path/"absent.pt"
    with pytest.raises(ValueError, match="CPU"):
        detect_ball([], checkpoint=path, expected_sha256="0"*64, device="cuda")
    with pytest.raises(ValueError, match="pin"):
        detect_ball([], checkpoint=path)


def test_full_frame_validation_counts_distant_false_positives():
    import pandas as pd
    from pompeu.vision.ball_det.validation import evaluate_frames
    set_deterministic()
    video = np.zeros((3, 24, 32, 3), np.uint8)
    video[:, 8, 5] = video[:, 8, 24] = 255
    table = pd.DataFrame({"frame": [1], "x_px": [5.0], "y_px": [8.0],
                          "crowded": [False], "split": ["val"]})
    detector = BallDetector(BrightPoint(), crop_shape=(16, 20), overlap=8)
    report = evaluate_frames(detector, video, table, block_frames=10)
    assert report["full_frame_val_frames"] == 1 and report["full_frame_candidates"] == 2
    assert report["full_frame_pseudo_precision_5px"] == 0.5
    assert report["full_frame_pseudo_recall_5px"] == 1
    empty = evaluate_frames(detector, video*0, table, block_frames=10)
    assert empty["full_frame_pseudo_recall_5px"] == 0
    with pytest.raises(ValueError, match="contexts"):
        evaluate_frames(detector, video[:2], table, block_frames=10)

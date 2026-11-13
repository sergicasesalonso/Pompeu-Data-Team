import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="B2 requires the separate vision environment")
from pompeu.vision.ball_det.model import (  # noqa: E402
    TemporalHeatmapNet, focal_heatmap_loss, heatmap_targets, set_deterministic,
)


@pytest.mark.parametrize("frames", [3, 5])
def test_temporal_shape_and_small_parameter_budget(frames):
    set_deterministic()
    model = TemporalHeatmapNet(n_frames=frames)
    logits = model(torch.zeros(2, frames, 3, 25, 33))
    assert logits.shape == (2, 1, 25, 33)
    assert sum(p.numel() for p in model.parameters()) < 150_000


def test_heatmap_targets_and_focal_gradients():
    target = heatmap_targets([[5.2, 8.1], [np.nan, np.nan]], 16, 20)
    assert target[0, 0, 8, 5] == 1 and target[1].sum() == 0
    logits = torch.zeros_like(target, requires_grad=True)
    loss = focal_heatmap_loss(logits, target, [0.5, 1.0])
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    better = torch.full_like(target, -8)
    better[0, 0, 8, 5] = 8
    assert focal_heatmap_loss(better, target) < loss
    with pytest.raises(ValueError):
        heatmap_targets([[-1, 3]], 16, 20)


def test_training_repeats_exactly_and_updates_weights():
    def run():
        set_deterministic(7)
        model = TemporalHeatmapNet(width=4)
        clips = torch.zeros(2, 3, 3, 16, 20)
        clips[:, :, :, 8, 5] = 1
        target = heatmap_targets([[5, 8], [5, 8]], 16, 20)
        initial = model.head.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = focal_heatmap_loss(model(clips), target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert not torch.equal(initial, model.head.weight)
        assert losses[-1] < losses[0]
        return losses, model(clips).detach()
    loss_a, output_a = run()
    loss_b, output_b = run()
    assert loss_a == loss_b
    assert torch.equal(output_a, output_b)


def test_separable_variant_supports_backward_at_native_resolution():
    set_deterministic()
    model = TemporalHeatmapNet(width=16, separable=True)
    assert sum(p.numel() for p in model.parameters()) < 20_000
    target = heatmap_targets([[12, 10]], 24, 32)
    loss = focal_heatmap_loss(model(torch.zeros(1, 3, 3, 24, 32)), target)
    loss.backward()
    assert torch.isfinite(loss)

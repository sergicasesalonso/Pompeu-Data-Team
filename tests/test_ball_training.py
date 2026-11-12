import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="separate B2 CPU environment required")
from pompeu.vision.ball_det.checkpoint import load_model  # noqa: E402
from pompeu.vision.ball_det.dataset import B1ClipDataset  # noqa: E402
from pompeu.vision.ball_det.model import TemporalHeatmapNet, set_deterministic  # noqa: E402
from pompeu.vision.ball_det.train import fit  # noqa: E402


def labels():
    return pd.DataFrame({"frame": [8, 9, 10, 11, 14], "x_px": [10.2]*5,
                         "y_px": [8.3]*5, "crowded": [False]*5,
                         "split": ["train", "train", "val", "val", "val"]})


def read_clip(frame, n_frames):
    clip = np.zeros((n_frames, 20, 24, 3), np.uint8)
    clip[:, 8, 10] = 255
    return clip


def dataset(split, augment=False):
    return B1ClipDataset(labels(), read_clip, split, crop_shape=(16, 20),
                         block_frames=10, augment=augment)


def test_split_guard_and_repeatable_augmentation():
    train, val = dataset("train", True), dataset("val")
    assert train.table.frame.tolist() == [8]
    assert val.table.frame.tolist() == [11, 14]
    assert set(train.table.frame).isdisjoint(val.table.frame)
    assert torch.equal(train[0]["clip"], train[0]["clip"])
    train.set_epoch(1)
    assert not torch.equal(dataset("train", True)[0]["clip"], train[0]["clip"])
    assert val[0]["target"].max() == 1
    with pytest.raises(ValueError):
        dataset("val", True)
    with pytest.raises(ValueError):
        val.set_epoch(1)


def test_train_save_reload_and_identical_checkpoint_bytes(tmp_path):
    def run(path):
        set_deterministic(3)
        model = TemporalHeatmapNet(width=4)
        history = fit(model, dataset("train", True), dataset("val"), path,
                      seed=3, epochs=2, batch_size=1)
        restored, info = load_model(path)
        assert info["seed"] == 3
        assert history[-1]["val_examples"] == 2
        sample = dataset("val")[0]["clip"][None]
        return history, path.read_bytes(), restored(sample).detach()
    a, bytes_a, out_a = run(tmp_path/"a.pt")
    b, bytes_b, out_b = run(tmp_path/"b.pt")
    assert a == b and bytes_a == bytes_b and torch.equal(out_a, out_b)
    with pytest.raises(ValueError, match="sha256"):
        load_model(tmp_path/"a.pt", "0"*64)

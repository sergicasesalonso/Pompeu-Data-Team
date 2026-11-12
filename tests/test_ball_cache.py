import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="separate B2 CPU environment required")
from pompeu.vision.ball_det.cache import CachedB1Dataset, prepare_cache  # noqa: E402
from pompeu.vision.ball_det.model import set_deterministic  # noqa: E402
from pompeu.vision.ball_det.video import iter_video  # noqa: E402


def test_stream_crop_cache_keeps_b1_splits_and_deterministic_crops(tmp_path):
    set_deterministic()
    table = pd.DataFrame({"frame": [8, 9, 10, 11, 14], "x_px": [10.2]*5,
                          "y_px": [8.3]*5, "crowded": [False]*5,
                          "split": ["train", "train", "val", "val", "val"]})
    video = np.zeros((16, 20, 24, 3), np.uint8)
    video[:, 8, 10] = 255
    directory = tmp_path/"clips"
    counts = prepare_cache(table, enumerate(video), directory, stride=1,
                           crop_shape=(16, 20), block_frames=10)
    assert counts == {"train": 1, "val": 2}
    train, val = CachedB1Dataset(directory, "train", augment=True), CachedB1Dataset(directory, "val")
    assert train.table.frame.tolist() == [8] and val.table.frame.tolist() == [11, 14]
    assert torch.equal(train[0]["clip"], train[0]["clip"])
    assert val[0]["clip"].shape == (3, 3, 16, 20)
    assert val[0]["target"].max() == 1
    with pytest.raises(FileExistsError):
        prepare_cache(table, enumerate(video), directory, stride=1, block_frames=10)


class Decoder:
    def __init__(self, data, returncode=0):
        self.stdout, self.returncode, self.terminated = io.BytesIO(data), returncode, False

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True


def test_video_frame_index_and_decode_errors():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    info = {"width": 3, "height": 2, "frames_header": 2}
    with patch("shutil.which", return_value="ffmpeg"), patch("subprocess.Popen", return_value=Decoder(image.tobytes()*2)):
        frames = list(iter_video("synthetic.mp4", info))
    assert [frame for frame, _ in frames] == [0, 1]
    np.testing.assert_array_equal(frames[1][1], image)
    with patch("shutil.which", return_value="ffmpeg"), patch("subprocess.Popen", return_value=Decoder(b"short")):
        with pytest.raises(RuntimeError, match="incomplete"):
            list(iter_video("synthetic.mp4", info))
    with patch("shutil.which", return_value="ffmpeg"), patch("subprocess.Popen", return_value=Decoder(image.tobytes())):
        with pytest.raises(RuntimeError, match="count"):
            list(iter_video("synthetic.mp4", info))
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="FFmpeg"):
            list(iter_video("synthetic.mp4", info))

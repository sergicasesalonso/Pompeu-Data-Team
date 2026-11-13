import errno
from unittest.mock import patch

import pandas as pd
import pytest

from pompeu.vision.ball_det.runtime import check_lock, runtime_status


def test_lock_normalises_names_and_reports_missing_or_incompatible_versions(tmp_path):
    lock = tmp_path/"test.lock"
    lock.write_text("# synthetic\n--extra-index-url https://example.invalid\nPy_YAML==1.0\ntorch==2.9.1+cu126\n")
    result = check_lock(lock, {"py-yaml": "1.0", "torch": "2.6.0+cu124"})
    assert result["missing_packages"] == []
    assert result["version_mismatches"] == [{"package": "torch", "expected": "2.9.1+cu126", "actual": "2.6.0+cu124"}]
    assert check_lock(lock, {})["missing_packages"] == ["py-yaml", "torch"]


def test_readonly_cache_is_not_ready(tmp_path):
    lock = tmp_path/"test.lock"
    lock.write_text("torch==2.9.1+cpu\n")
    with patch("tempfile.TemporaryFile", side_effect=OSError(errno.EROFS, "synthetic read-only disk")):
        report = runtime_status(lock, tmp_path/"cache")
    assert not report["cache_writable"] and not report["ready"]


from pompeu.vision.ball_det.runtime import input_plan
from pompeu.vision.ball_det.smoke import training_smoke


def small_config():
    return {"seed": 7, "model": {"n_frames": 3, "width": 4},
            "data": {"crop_shape": [16, 20], "stride": 1, "negative_crops": 1},
            "train": {"batch_size": 1, "epochs": 1, "lr": 0.001}}


def test_input_plan_counts_temporal_guard_and_negative_payload():
    pytest.importorskip("torch")
    table = pd.DataFrame({"frame": [8, 9, 10, 11], "x_px": [10.0]*4, "y_px": [8.0]*4,
                          "crowded": [False]*4, "split": ["train", "train", "val", "val"]})
    report = input_plan(table, small_config(), fps=10, block_s=1)
    assert report["eligible_b1_centres"] == {"train": 1, "val": 1}
    assert report["crop_samples"] == {"train": 2, "val": 2}
    assert report["raw_crop_payload_bytes"] == 4*3*3*16*20


def test_synthetic_training_smoke_repeats_without_writing_checkpoints(tmp_path):
    pytest.importorskip("torch")
    report = training_smoke(small_config(), steps=2)
    assert report["exact_repeatability"]
    assert report["state_sha256"] == report["repeat_state_sha256"]
    assert report["peak_cuda_allocated_bytes"] is None
    assert not list(tmp_path.iterdir())

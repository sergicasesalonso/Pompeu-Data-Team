import pytest

from pompeu.vision.ball_det.provenance import checkpoint_pin, configuration_hash, file_sha256, verify_sources


def test_artifact_pin_does_not_move_experiment_and_is_required():
    cfg = {"seed": 0, "model": {"width": 4}, "artifact": {"checkpoint_sha256": None}}
    original = configuration_hash(cfg)
    with pytest.raises(ValueError, match="pin"):
        checkpoint_pin(cfg)
    cfg["artifact"]["checkpoint_sha256"] = "a"*64
    assert checkpoint_pin(cfg) == "a"*64 and configuration_hash(cfg) == original
    cfg["model"]["width"] = 8
    assert configuration_hash(cfg) != original


def test_stale_b1_and_different_video_are_rejected(tmp_path):
    b1, video = tmp_path/"synthetic_b1", tmp_path/"synthetic_video"
    b1.write_bytes(b"synthetic label version one")
    video.write_bytes(b"synthetic video bytes")
    metadata = {"config_sha256": "cfg", "b1_sha256": file_sha256(b1), "video_sha256": file_sha256(video)}
    verify_sources(metadata, "cfg", b1, video)
    b1.write_bytes(b"synthetic label version two")
    with pytest.raises(ValueError, match="B1"):
        verify_sources(metadata, "cfg", b1)
    b1.write_bytes(b"synthetic label version one")
    video.write_bytes(b"different synthetic video bytes")
    with pytest.raises(ValueError, match="video"):
        verify_sources(metadata, "cfg", b1, video)

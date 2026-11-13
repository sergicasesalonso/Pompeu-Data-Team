import importlib.util
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
from pompeu import paths  # noqa: E402
from pompeu.vision.ball_det.checkpoint import save_checkpoint  # noqa: E402
from pompeu.vision.ball_det.model import TemporalHeatmapNet  # noqa: E402
from pompeu.vision.ball_det.provenance import configuration_hash, file_sha256  # noqa: E402


def test_pin_records_local_checkpoint_without_changing_experiment_identity(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("b2_runner", Path(paths.ROOT)/"scripts/ball_detector.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = {"seed": 0, "model": {"n_frames": 3, "width": 4}, "execution": {"threads": 1},
              "artifact": {"checkpoint_sha256": None}}
    config_path = tmp_path/"pilot.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    identity = configuration_hash(config)
    monkeypatch.setattr(paths, "CACHE_DIR", str(tmp_path/"private-cache"))
    checkpoint = Path(paths.CACHE_DIR)/"ball_det"/identity[:16]/"best.pt"
    save_checkpoint(TemporalHeatmapNet(width=4), checkpoint, {"config_sha256": identity})
    runner.main(["pin", "--config", str(config_path)])
    pinned = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert pinned["artifact"]["checkpoint_sha256"] == file_sha256(checkpoint)
    assert configuration_hash(pinned) == identity

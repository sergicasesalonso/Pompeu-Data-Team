"""Private checkpoints; deterministic bytes for identical CPU states and metadata."""

import hashlib
import io
from pathlib import Path

import torch

from .model import TemporalHeatmapNet


def save_checkpoint(model, path, metadata):
    payload = {"version": 1, "model_config": model.config, "metadata": metadata,
               "torch_version": str(torch.__version__),
               "state_dict": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def load_model(path, expected_sha256=None):
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("checkpoint sha256 mismatch")
    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if payload.get("version") != 1:
        raise ValueError("unsupported B2 checkpoint format")
    with torch.random.fork_rng(devices=[]):
        model = TemporalHeatmapNet(**payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), payload["metadata"]

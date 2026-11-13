"""Opaque local input hashes; never print rows or retain input contents."""

import hashlib
import json
from pathlib import Path
import re


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration_hash(config):
    """Exclude the output artifact pin to avoid a self-referential config hash."""
    experiment = {k: v for k, v in config.items() if k != "artifact"}
    encoded = json.dumps(experiment, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_pin(config):
    value = config.get("artifact", {}).get("checkpoint_sha256")
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("pin the trained checkpoint SHA-256 in config artifact.checkpoint_sha256")
    return value


def verify_sources(metadata, config_hash, b1_path, video_path=None):
    if metadata.get("config_sha256") != config_hash:
        raise ValueError("cache configuration hash differs")
    if metadata.get("b1_sha256") != file_sha256(b1_path):
        raise ValueError("B1 hash differs; regenerate the crop cache")
    if video_path is not None and metadata.get("video_sha256") != file_sha256(video_path):
        raise ValueError("video hash differs from preparation input")

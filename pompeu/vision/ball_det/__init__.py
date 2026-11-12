"""B2 heatmap utilities; no model weights or match data are bundled."""

import numpy as np
import pandas as pd


def detect_ball(frames, *, checkpoint=None, expected_sha256=None, **options):
    """Load a private trained checkpoint and detect on CPU unless explicitly overridden.

    PyTorch remains an optional dependency: importing heatmap utilities does not
    import it. Input is a video array or (absolute_frame, RGB_uint8) pairs.
    """
    from pathlib import Path
    import hashlib
    from pompeu import paths
    from .checkpoint import load_model
    from .predict import BallDetector
    if checkpoint is None:
        config = Path(paths.ROOT) / "configs" / "b2_temporal.yaml"
        config_id = hashlib.sha256(config.read_bytes()).hexdigest()[:16]
        checkpoint = Path(paths.CACHE_DIR) / "ball_det" / config_id / "best.pt"
    path = checkpoint
    model, metadata = load_model(path, expected_sha256)
    if "crop_shape" in metadata:
        options.setdefault("crop_shape", tuple(metadata["crop_shape"]))
    return BallDetector(model, **options).detect_ball(frames)


def make_heatmap(height, width, x_px, y_px, sigma=2.0):
    """Gaussian target in crop pixels, with its peak at the labelled centre."""
    if height <= 0 or width <= 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("positive dimensions and sigma required")
    if not np.isfinite([x_px, y_px]).all() or not (0 <= x_px < width and 0 <= y_px < height):
        raise ValueError("target must be inside the crop")
    y, x = np.ogrid[:height, :width]
    return np.exp(-((x - x_px)**2 + (y - y_px)**2) / (2 * sigma**2)).astype(np.float32)


def decode_heatmaps(heatmaps, frame_ids, top_k=3, threshold=0.5, radius=4):
    """Decode crop-resolution probabilities; deterministic ties use row-major order.

    Coordinates refer to the input heatmap, with no implicit resize or crop offset.
    The inference caller must restore full-image coordinates before passing to B3.
    """
    maps = np.asarray(heatmaps, dtype=np.float32)
    ids = np.asarray(frame_ids)
    if maps.ndim != 3 or min(maps.shape[1:]) <= 0 or ids.shape != (len(maps),):
        raise ValueError("expected (N,H,W) heatmaps and N frame IDs")
    if not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any():
        raise ValueError("frame IDs must be nonnegative integers")
    if not np.isfinite(maps).all() or (maps < 0).any() or (maps > 1).any():
        raise ValueError("heatmaps must contain finite probabilities")
    if not isinstance(top_k, int) or top_k < 1 or not isinstance(radius, int) or radius < 0:
        raise ValueError("positive top_k and nonnegative integer radius required")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0,1]")
    rows = []
    for frame, original in zip(ids, maps):
        scores = original.copy()
        for _ in range(top_k):
            y, x = np.unravel_index(np.argmax(scores), scores.shape)
            conf = float(scores[y, x])
            if conf < threshold:
                break
            rows.append((int(frame), float(x), float(y), round(conf, 3)))
            scores[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = -1
    return pd.DataFrame(rows, columns=["frame", "x_px", "y_px", "conf"]).astype(
        {"frame": "int64", "x_px": "float64", "y_px": "float64", "conf": "float64"})

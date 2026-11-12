"""Bounded-batch CPU-default inference; returns pixels only to the local caller."""

from collections import deque

import numpy as np
import torch

from . import decode_heatmaps
from .model import set_deterministic


class BallDetector:
    """Pass consecutive (absolute_frame, RGB_uint8) pairs, or an RGB video array.

    The first/last context-radius frames and contexts spanning gaps are omitted.
    Tiles stay at native resolution; overlaps are suppressed before top-k output.
    No tracking, interpolation or A1 cleaning is performed here.
    """

    def __init__(self, model, *, crop_shape=(256, 448), batch_size=8, overlap=32,
                 top_k=3, threshold=0.5, radius=4, device="cpu"):
        if (len(crop_shape) != 2 or any(not isinstance(v, int) or v < 4 for v in crop_shape)
                or not isinstance(overlap, int) or not 0 <= overlap < min(crop_shape)
                or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("invalid crop, overlap or batch size")
        # Reuse the decoder's validation even for an empty input stream.
        decode_heatmaps(np.empty((0, 1, 1)), np.array([], dtype=np.int64), top_k, threshold, radius)
        set_deterministic()
        self.model, self.device = model.to(device).eval(), device
        self.n_frames = int(model.config["n_frames"])
        if self.n_frames not in (3, 5):
            raise ValueError("expected a 3- or 5-frame model")
        self.crop_shape, self.batch_size, self.overlap = tuple(crop_shape), batch_size, overlap
        self.top_k, self.threshold, self.radius = top_k, threshold, radius

    def _positions(self, length, crop):
        last = max(0, length-crop)
        return sorted(set([*range(0, last+1, crop-self.overlap), last]))

    @torch.inference_mode()
    def _predict_frame(self, clip, frame):
        _, h, w, _ = clip.shape
        ch, cw = self.crop_shape
        locations = [(x, y) for y in self._positions(h, ch) for x in self._positions(w, cw)]
        candidates = []
        for start in range(0, len(locations), self.batch_size):
            batch_locations = locations[start:start+self.batch_size]
            crops = np.zeros((len(batch_locations), self.n_frames, ch, cw, 3), dtype=np.uint8)
            for i, (x, y) in enumerate(batch_locations):
                cut = clip[:, y:y+ch, x:x+cw]
                crops[i, :, :cut.shape[1], :cut.shape[2]] = cut
            inputs = torch.from_numpy(crops.transpose(0, 1, 4, 2, 3).copy()).to(self.device).float()/255
            logits = self.model(inputs)
            if logits.shape != (len(crops), 1, ch, cw):
                raise ValueError("model output has an invalid heatmap shape")
            maps = logits.sigmoid()[:, 0].cpu().numpy()
            table = decode_heatmaps(maps, np.arange(len(crops)), self.top_k, self.threshold, self.radius)
            for tile, px, py, conf in table.itertuples(index=False, name=None):
                x, y = batch_locations[tile]
                if x+px < w and y+py < h:  # never accept zero-padding detections
                    candidates.append((x+px, y+py, conf))
        chosen = []
        for x, y, conf in sorted(candidates, key=lambda p: (-p[2], p[1], p[0])):
            if all((x-a)**2 + (y-b)**2 > self.radius**2 for a, b, _ in chosen):
                chosen.append((x, y, conf))
                if len(chosen) == self.top_k:
                    break
        return [(frame, round(x, 2), round(y, 2), conf) for x, y, conf in chosen]

    def detect_ball(self, frames):
        source = enumerate(frames) if isinstance(frames, np.ndarray) else frames
        buffer, rows = deque(maxlen=self.n_frames), []
        previous, shape = None, None
        for frame, image in source:
            if not isinstance(frame, (int, np.integer)) or frame < 0 or (previous is not None and frame <= previous):
                raise ValueError("frame IDs must be strictly increasing nonnegative integers")
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1 or image.dtype != np.uint8:
                raise ValueError("expected RGB uint8 images")
            if shape is not None and image.shape != shape:
                raise ValueError("image dimensions must stay constant")
            if previous is not None and frame != previous+1:
                buffer.clear()
            # Decoder callers may reuse a single output buffer.
            buffer.append((int(frame), image.copy()))
            previous, shape = frame, image.shape
            if len(buffer) == self.n_frames:
                clip = np.stack([item[1] for item in buffer])
                rows.extend(self._predict_frame(clip, buffer[self.n_frames//2][0]))
        import pandas as pd
        return pd.DataFrame(rows, columns=["frame", "x_px", "y_px", "conf"]).astype(
            {"frame": "int64", "x_px": "float64", "y_px": "float64", "conf": "float64"})

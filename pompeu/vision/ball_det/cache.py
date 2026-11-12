"""One-pass private crop cache; never display samples, labels or coordinates."""

from collections import deque
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import B1ClipDataset, augment_clip
from .model import heatmap_targets


def prepare_cache(table, frames, directory, *, stride=5, metadata=None, **settings):
    """Cache fixed native-resolution crops from a single sequential decode.

    `frames` yields (absolute frame, RGB uint8). stride subsamples B1 rows by
    frame modulo, independent of labels. Directory must be new: an incomplete
    build cannot silently be reused. Per-frame manifest and clips remain local.
    """
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("positive integer stride required")
    settings = {"crop_shape": (256, 448), "n_frames": 3, "block_frames": 1500, "seed": 0, **settings}
    selected = table.loc[table.frame % stride == 0]
    datasets = [B1ClipDataset(selected, None, split, **settings) for split in ("train", "val")]
    if any(not len(dataset) for dataset in datasets):
        raise ValueError("both B1 splits need eligible contexts")
    lookup = {int(row.frame): (dataset, index) for dataset in datasets
              for index, row in dataset.table.iterrows()}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    n_frames = settings["n_frames"]
    buffer, seen = deque(maxlen=n_frames), set()
    previous, shape = None, None
    for frame, image in frames:
        image = np.asarray(image)
        if (not isinstance(frame, (int, np.integer)) or frame < 0
                or (previous is not None and frame <= previous)
                or image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8):
            raise ValueError("invalid ordered RGB video stream")
        if shape is not None and image.shape != shape:
            raise ValueError("video dimensions changed")
        if previous is not None and frame != previous+1:
            buffer.clear()
        buffer.append((int(frame), image.copy()))
        previous, shape = frame, image.shape
        if len(buffer) != n_frames:
            continue
        centre = buffer[n_frames//2][0]
        if centre not in lookup:
            continue
        dataset, index = lookup[centre]
        clip = np.stack([item[1] for item in buffer])
        dataset.read_clip = lambda frame, count: clip
        sample = dataset[index]
        np.savez_compressed(directory/f"{centre:010d}.npz",
                            clip=np.rint(sample["clip"].numpy()*255).astype(np.uint8),
                            point=sample["point"].numpy())
        seen.add(centre)
    if seen != set(lookup):
        raise ValueError("not all eligible B1 contexts were decoded; cache is incomplete")
    manifest = pd.concat([dataset.table for dataset in datasets], ignore_index=True).sort_values("frame")
    manifest.to_parquet(directory/"manifest.parquet", index=False)
    info = {"version": 1, "settings": settings, "stride": stride, "metadata": metadata or {},
            "counts": {dataset.split: len(dataset) for dataset in datasets}}
    (directory/"ready.json").write_text(json.dumps(info, sort_keys=True), encoding="utf-8")
    return info["counts"]


class CachedB1Dataset(B1ClipDataset):
    """Reuse the exact B1 split and fixed cached crops; augmentation remains train-only."""

    def __init__(self, directory, split, *, augment=False):
        self.directory = Path(directory)
        info = json.loads((self.directory/"ready.json").read_text(encoding="utf-8"))
        if info.get("version") != 1:
            raise ValueError("unsupported B2 crop cache")
        table = pd.read_parquet(self.directory/"manifest.parquet")
        super().__init__(table, None, split, augment=augment, **info["settings"])
        self.metadata = info["metadata"]

    def __getitem__(self, index):
        row = self.table.iloc[index]
        frame = int(row.frame)
        with np.load(self.directory/f"{frame:010d}.npz", allow_pickle=False) as stored:
            pixels, point = stored["clip"], stored["point"].tolist()
        if pixels.shape != (self.n_frames, 3, *self.crop_shape) or pixels.dtype != np.uint8:
            raise ValueError("invalid cached clip")
        tensor = torch.from_numpy(pixels.copy()).float()/255
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, frame]))
            tensor, point = augment_clip(tensor, point, rng)
        return {"clip": tensor, "target": heatmap_targets([point], *self.crop_shape)[0],
                "point": torch.tensor(point), "weight": 0.5 if row.crowded else 1.0, "frame": frame}

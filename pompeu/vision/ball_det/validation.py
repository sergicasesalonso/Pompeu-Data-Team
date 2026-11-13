"""Full-image held-out B1 agreement, counting all candidates and zero detections.

This is agreement with projected pseudo-labels, not an estimate against human GT
or of accuracy on the ATD-missing distribution. No per-frame rows are printed.
"""

import numpy as np

from .dataset import B1ClipDataset


def evaluate_frames(detector, frames, table, *, block_frames=1500):
    val = B1ClipDataset(table, None, "val", crop_shape=detector.crop_shape,
                        n_frames=detector.n_frames, block_frames=block_frames).table
    val = val.loc[val.frame % detector.frame_stride == 0]
    targets = {int(row.frame): np.array([row.x_px, row.y_px]) for row in val.itertuples()}
    if not targets:
        raise ValueError("no eligible validation frames")
    seen, candidates, hits = set(), 0, {5: 0, 10: 0}
    for frame, rows in detector.iter_detections(frames, frame_ids=targets):
        seen.add(frame)
        candidates += len(rows)
        xy = np.asarray([(row[1], row[2]) for row in rows], dtype=float).reshape(-1, 2)
        distances = np.linalg.norm(xy-targets[frame], axis=1)
        for radius in hits:
            hits[radius] += int((distances <= radius).any())
    if seen != set(targets):
        raise ValueError("validation video lacks some eligible temporal contexts")
    report = {"full_frame_val_frames": len(seen), "full_frame_candidates": candidates}
    for radius, count in hits.items():
        report[f"full_frame_pseudo_precision_{radius}px"] = count/candidates if candidates else 0.0
        report[f"full_frame_pseudo_recall_{radius}px"] = count/len(seen)
    return report

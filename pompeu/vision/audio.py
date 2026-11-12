"""B7 spectral candidate extraction. No claims of referee-whistle accuracy.

The caller supplies decoded mono PCM and the measured audio/video timestamp offset.
Video decoding and the public detect_whistles(video_path) adapter are pending.
"""

import numpy as np
import pandas as pd


def whistle_candidates(samples, sample_rate, *, fps, audio_offset_s,
                       min_duration_s=0.12, threshold=0.55, merge_gap_s=0.10):
    """Return frame/conf candidates using sustained narrow-band energy.

    audio_offset_s = first decoded audio timestamp minus first video timestamp.
    Confidence is a spectral heuristic, not a calibrated probability. Silence and
    broadband noise are rejected by band energy and spectral concentration.
    """
    pcm = np.asarray(samples, dtype=np.float64)
    if pcm.ndim != 1 or not np.isfinite(pcm).all():
        raise ValueError("expected finite mono samples")
    if (not np.isfinite([sample_rate, fps, audio_offset_s, min_duration_s,
                         threshold, merge_gap_s]).all()
            or sample_rate <= 9000 or fps <= 0 or min_duration_s <= 0
            or not 0 < threshold <= 1 or merge_gap_s < 0):
        raise ValueError("invalid sampling, timing or detection settings")
    size = int(round(sample_rate * 0.032))
    hop = max(1, int(round(sample_rate * 0.008)))
    freq = np.fft.rfftfreq(size, 1 / sample_rate)
    band = (freq >= 2000) & (freq <= 4500)
    window = np.hanning(size)
    active = []
    for start in range(0, len(pcm) - size + 1, hop):
        chunk = pcm[start:start+size]
        power = abs(np.fft.rfft((chunk - chunk.mean()) * window))**2
        total = float(power.sum())
        selected = power[band]
        energy = float(selected.sum())
        concentration = float(np.sort(selected)[-3:].sum()) / max(energy, 1e-30)
        score = energy / max(total, 1e-30) * concentration
        if total > 1e-20 and score >= threshold:
            active.append((start, score))
    rows = []
    i = 0
    while i < len(active):
        j = i + 1
        while j < len(active) and active[j][0] - active[j-1][0] <= hop:
            j += 1
        # Duration must be supported continuously, before merging event candidates.
        duration = (active[j-1][0] - active[i][0] + size) / sample_rate
        if duration >= min_duration_s:
            time = (active[i][0] + size / 2) / sample_rate + audio_offset_s
            if time >= 0:
                rows.append([time, max(score for _, score in active[i:j])])
        i = j
    merged = []
    for time, score in rows:
        if merged and time - merged[-1][0] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], score)
        else:
            merged.append([time, score])
    events = [(int(np.floor(time * fps + 0.5)), round(score, 3)) for time, score in merged]
    table = pd.DataFrame(events, columns=["frame", "conf"]).astype({"frame": "int64", "conf": "float64"})
    return table.groupby("frame", as_index=False, sort=True).conf.max()

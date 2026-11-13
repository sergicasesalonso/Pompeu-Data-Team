"""B7: local streaming whistle candidates; confidence is an uncalibrated score.

No audio or decoder stderr is displayed or saved. Only frame/conf events leave
this module. Thresholds are initial defaults tested with synthetic signals.
"""

from fractions import Fraction
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd


def _table(events=()):
    table = pd.DataFrame(events, columns=["frame", "conf"]).astype({"frame": "int64", "conf": "float64"})
    return table.groupby("frame", as_index=False, sort=True).conf.max()


class _WhistleStream:
    def __init__(self, sample_rate, *, fps, audio_offset_s, min_duration_s=0.12,
                 threshold=0.55, merge_gap_s=0.10):
        if (not np.isfinite([sample_rate, fps, audio_offset_s, min_duration_s, threshold, merge_gap_s]).all()
                or sample_rate <= 9000 or fps <= 0 or min_duration_s <= 0
                or not 0 < threshold <= 1 or merge_gap_s < 0):
            raise ValueError("invalid sampling, timing or detection settings")
        self.rate, self.fps, self.offset = sample_rate, fps, audio_offset_s
        self.duration, self.threshold, self.gap = min_duration_s, threshold, merge_gap_s
        self.size, self.hop = int(round(sample_rate*0.032)), max(1, int(round(sample_rate*0.008)))
        freq = np.fft.rfftfreq(self.size, 1/sample_rate)
        self.band, self.window = (freq >= 2000) & (freq <= 4500), np.hanning(self.size)
        self.buffer, self.position = np.empty(0), 0
        self.group, self.pending, self.events = None, None, []

    def _close_group(self):
        if self.group is None:
            return
        first, last, score = self.group
        self.group = None
        start, end = first/self.rate+self.offset, (last+self.size)/self.rate+self.offset
        if (last-first+self.size)/self.rate < self.duration or end < 0:
            return
        time = max(0.0, start+self.size/(2*self.rate))
        if self.pending is not None and start-self.pending[1] <= self.gap:
            self.pending[1], self.pending[2] = end, max(score, self.pending[2])
        else:
            if self.pending is not None:
                self.events.append(self.pending)
            self.pending = [time, end, score]

    def feed(self, samples):
        pcm = np.asarray(samples, dtype=np.float64)
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError("expected finite mono samples")
        self.buffer = np.concatenate((self.buffer, pcm))
        consumed = 0
        while consumed+self.size <= len(self.buffer):
            chunk = self.buffer[consumed:consumed+self.size]
            power = abs(np.fft.rfft((chunk-chunk.mean())*self.window))**2
            energy = float(power[self.band].sum())
            total = float(power.sum())
            concentration = float(np.sort(power[self.band])[-3:].sum()) / max(energy, 1e-30)
            score = energy/max(total, 1e-30)*concentration
            start = self.position+consumed
            if total > 1e-20 and score >= self.threshold:
                if self.group is None:
                    self.group = [start, start, score]
                else:
                    self.group[1], self.group[2] = start, max(score, self.group[2])
            else:
                self._close_group()
            consumed += self.hop
        self.buffer = self.buffer[consumed:].copy()
        self.position += consumed

    def finish(self):
        self._close_group()
        if self.pending is not None:
            self.events.append(self.pending)
            self.pending = None
        return _table((int(np.floor(time*self.fps+0.5)), round(float(np.clip(score, 0, 1)), 3))
                      for time, _, score in self.events)


def whistle_candidates(samples, sample_rate, *, fps, audio_offset_s, **settings):
    """Mono PCM to candidates; time is the first qualifying STFT window centre.

    Offset = first decoded PCM timestamp minus first video timestamp. Spectral
    confidence is not a probability. No public possession GT is used.
    """
    detector = _WhistleStream(sample_rate, fps=fps, audio_offset_s=audio_offset_s, **settings)
    pcm = np.asarray(samples)
    if pcm.ndim != 1:
        raise ValueError("expected mono samples")
    for start in range(0, len(pcm), 16384):
        detector.feed(pcm[start:start+16384])
    return detector.finish()


def _audio_timing(video_path):
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("ffprobe is required for local B7 timing")
    probe = subprocess.run([executable, "-v", "error", "-show_entries",
                            "stream=codec_type,start_time,r_frame_rate,avg_frame_rate,nb_frames,duration",
                            "-of", "json", str(video_path)], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, text=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if probe.returncode:
        raise RuntimeError("audio/video metadata probe failed")
    streams = json.loads(probe.stdout).get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError("no video stream")
    if audio is None:
        return None
    try:
        fps = Fraction(video["r_frame_rate"])
        average = Fraction(video["avg_frame_rate"])
        audio_start, video_start = float(audio["start_time"]), float(video["start_time"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise ValueError("audio/video timestamps and constant frame rate are required") from None
    if (fps <= 0 or average <= 0 or not np.isclose(float(fps), float(average), rtol=1e-4, atol=1e-6)
            or not np.isfinite([audio_start, video_start]).all()):
        raise ValueError("unsupported frame rate or missing audio/video timestamps")
    try:
        count = int(video.get("nb_frames", "N/A"))
    except (ValueError, TypeError):
        try:
            duration = float(video["duration"])
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError("invalid duration")
            count = int(np.floor(duration*float(fps)+0.5))
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError("video frame count or duration is required") from None
    if count < 1:
        raise ValueError("positive video frame count required")
    return float(fps), audio_start, video_start, count


def detect_whistles(video_path, *, sample_rate=16000, block_samples=16384, **settings):
    """Local block decoder -> DataFrame(frame, conf); CPU-only, bounded PCM.

    copyts preserves stream origins. aresample fills timestamp gaps and anchors
    PCM to a measured PTS, including codec-delay trimming. Missing timing raises
    an error; no audio returns a typed empty table. No temporary clips are made.
    """
    if not isinstance(sample_rate, int) or sample_rate <= 9000 or not isinstance(block_samples, int) or block_samples < 1:
        raise ValueError("invalid PCM sampling or block size")
    timing = _audio_timing(video_path)
    if timing is None:
        return _table()
    fps, audio_start, video_start, n_frames = timing
    first_pts = int(np.floor(audio_start*sample_rate+0.5))
    detector = _WhistleStream(sample_rate, fps=fps, audio_offset_s=first_pts/sample_rate-video_start, **settings)
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("FFmpeg is required for local B7 audio decoding")
    command = [executable, "-nostdin", "-v", "error", "-threads", "1", "-copyts", "-i", str(video_path),
               "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-af",
               f"aresample={sample_rate}:async=1:first_pts={first_pts}", "-f", "f32le", "pipe:1"]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    remainder = b""
    try:
        while True:
            raw = process.stdout.read(block_samples*4)
            if not raw:
                break
            raw = remainder+raw
            complete = len(raw)//4*4
            detector.feed(np.frombuffer(raw[:complete], dtype="<f4"))
            remainder = raw[complete:]
        if remainder or process.wait() != 0:
            raise RuntimeError("audio decoding failed or returned incomplete PCM")
        events = detector.finish()
        return events.loc[events.frame.between(0, n_frames-1)].reset_index(drop=True)
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()

import io
import json
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from pompeu.vision.audio import _WhistleStream, detect_whistles, whistle_candidates


def tone(rate=16000, intervals=((0.2, 0.6),), length=1.2):
    t = np.arange(int(rate*length))/rate
    mask = np.zeros(len(t), bool)
    for start, end in intervals:
        mask |= (t >= start) & (t < end)
    return (np.sin(2*np.pi*3000*t)*mask).astype(np.float32)


@pytest.mark.parametrize("block", [17, 999, 8192])
def test_chunk_boundaries_do_not_change_events(block):
    pcm = tone()
    stream = _WhistleStream(16000, fps=25, audio_offset_s=0.24)
    for start in range(0, len(pcm), block):
        stream.feed(pcm[start:start+block])
        assert len(stream.buffer) < stream.size
    expected = whistle_candidates(pcm, 16000, fps=25, audio_offset_s=0.24)
    assert stream.finish().equals(expected)
    assert len(expected) == 1


def test_event_grouping_short_tones_and_overlap():
    args = dict(sample_rate=16000, fps=25, audio_offset_s=0)
    assert len(whistle_candidates(tone(intervals=((0.2, 0.4), (0.45, 0.65))), **args)) == 1
    assert len(whistle_candidates(tone(intervals=((0.2, 0.4), (0.9, 1.1))), **args)) == 2
    assert whistle_candidates(tone(intervals=((0.2, 0.22),)), **args).empty
    assert whistle_candidates(tone(), 16000, fps=25, audio_offset_s=-2).empty
    noise = np.random.default_rng(8).normal(0, 0.1, size=len(tone()))
    assert len(whistle_candidates(tone()+noise, **args)) == 1


def probe(audio_start="10.24"):
    streams = [{"codec_type": "video", "start_time": "10.0", "r_frame_rate": "25/1",
                "avg_frame_rate": "25/1"}]
    if audio_start is not None:
        streams.append({"codec_type": "audio", "start_time": audio_start})
    return subprocess.CompletedProcess([], 0, stdout=json.dumps({"streams": streams}))


class PartialPCM(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 101))


class Decoder:
    def __init__(self, pcm, code=0):
        self.stdout, self.returncode = PartialPCM(pcm), code

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


def test_video_adapter_uses_nonzero_offset_and_partial_pcm_reads():
    pcm = tone()
    with patch("shutil.which", side_effect=lambda name: name), patch("subprocess.run", return_value=probe()), \
            patch("subprocess.Popen", return_value=Decoder(pcm.tobytes())) as decoder:
        result = detect_whistles("synthetic.mp4", block_samples=257)
    expected = whistle_candidates(pcm, 16000, fps=25, audio_offset_s=0.24)
    assert result.equals(expected)
    command = decoder.call_args.args[0]
    assert "-copyts" in command
    assert "aresample=16000:async=1:first_pts=163840" in command
    assert decoder.call_args.kwargs["stderr"] == subprocess.DEVNULL


def test_missing_audio_or_timestamps_and_decode_failure():
    with patch("shutil.which", return_value="ffprobe"), patch("subprocess.run", return_value=probe(None)), \
            patch("subprocess.Popen") as decoder:
        result = detect_whistles("synthetic.mp4")
        assert result.empty and list(result.columns) == ["frame", "conf"]
        decoder.assert_not_called()
    with patch("shutil.which", return_value="ffprobe"), patch("subprocess.run", return_value=probe("N/A")):
        with pytest.raises(ValueError, match="timestamps"):
            detect_whistles("synthetic.mp4")
    with patch("shutil.which", side_effect=lambda name: name), patch("subprocess.run", return_value=probe()), \
            patch("subprocess.Popen", return_value=Decoder(b"x")):
        with pytest.raises(RuntimeError, match="PCM"):
            detect_whistles("synthetic.mp4")

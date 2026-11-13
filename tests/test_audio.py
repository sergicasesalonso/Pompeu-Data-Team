import numpy as np
import pytest

from pompeu.vision.audio import whistle_candidates


@pytest.mark.parametrize("rate", [16000, 48000])
def test_sustained_tone_and_offset(rate):
    t = np.arange(rate) / rate
    pcm = np.sin(2 * np.pi * 3000 * t) * ((t >= 0.2) & (t < 0.5))
    args = dict(fps=25, audio_offset_s=0)
    base = whistle_candidates(pcm, rate, **args)
    shifted = whistle_candidates(pcm, rate, fps=25, audio_offset_s=1)
    assert len(base) == 1
    assert shifted.frame.iloc[0] - base.frame.iloc[0] == 25
    assert base.conf.between(0, 1).all()


def test_silence_noise_and_short_tone():
    args = dict(sample_rate=16000, fps=25, audio_offset_s=0)
    assert whistle_candidates(np.zeros(16000), **args).empty
    assert whistle_candidates(np.random.default_rng(0).normal(size=16000), **args).empty
    assert whistle_candidates(np.zeros(0), **args).empty


def test_invalid_sampling_rate():
    with pytest.raises(ValueError):
        whistle_candidates(np.zeros(100), 8000, fps=25, audio_offset_s=0)

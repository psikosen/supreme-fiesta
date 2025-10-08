from pathlib import Path

import numpy as np
import pytest

from voice_agent.vad import SileroVadConfig, SileroVadStream, VadState


class FakeModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.index = 0
        self.reset_called = 0

    def reset_states(self) -> None:
        self.reset_called += 1
        self.index = 0

    def __call__(self, chunk: np.ndarray, sample_rate: int) -> float:
        if self.index >= len(self.probabilities):
            return 0.0
        value = self.probabilities[self.index]
        self.index += 1
        return value


class CountingModel:
    def __init__(self) -> None:
        self.calls = 0
        self.reset_called = 0

    def reset_states(self) -> None:
        self.reset_called += 1

    def __call__(self, chunk: np.ndarray, sample_rate: int) -> float:
        self.calls += 1
        # return low confidence so no events fire
        return 0.0


def _build_stream(probabilities: list[float]) -> SileroVadStream:
    config = SileroVadConfig(
        model_path=Path("silero_vad.onnx"),
        sample_rate=16000,
        trigger_level=0.6,
        release_level=0.3,
        sensitivity=1.0,
        frame_size=512,
    )
    audio_chunks = [np.ones(config.frame_size, dtype=np.float32) for _ in probabilities]
    model = FakeModel(probabilities)
    stream = SileroVadStream(audio_chunks, config, model=model)
    return stream


def test_silero_vad_stream_transitions():
    stream = _build_stream([0.05, 0.7, 0.8, 0.9, 0.85, 0.2, 0.1, 0.05, 0.05])
    events = list(stream)
    assert [event.state for event in events] == [VadState.SPEECH, VadState.SILENCE]
    assert events[0].timestamp == pytest.approx(0.064, rel=1e-2)
    assert events[1].timestamp == pytest.approx(0.288, rel=1e-2)
    assert events[0].confidence == pytest.approx(0.61, rel=1e-2)
    assert events[1].confidence == pytest.approx(0.052, rel=1e-2)


def test_from_numpy_yields_expected_frames():
    audio = np.ones(1200, dtype=np.float32)
    model = CountingModel()
    stream = SileroVadStream.from_numpy(
        audio,
        model_path=Path("silero_vad.onnx"),
        sample_rate=16000,
        trigger_level=0.6,
        release_level=0.3,
        sensitivity=0.5,
        frame_size=512,
        model=model,
    )
    events = list(stream)
    assert events == []
    assert model.calls == 3
    assert model.reset_called == 1


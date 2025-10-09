from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from voice_agent.asr import create_asr_engine
from voice_agent.asr.faster_whisper import FasterWhisperEngine
from voice_agent.asr.mlx_whisper import MlxWhisperEngine
from voice_agent.config.models import AsrConfig


def _audio_chunk() -> bytes:
    return np.ones(1600, dtype=np.float32).tobytes()


def test_faster_whisper_engine_transcribes(monkeypatch: pytest.MonkeyPatch) -> None:
    segments = [
        types.SimpleNamespace(text=" hello", end=0.5, avg_log_prob=np.log(0.9)),
        types.SimpleNamespace(text="world", end=1.0, avg_log_prob=np.log(0.8)),
    ]

    class DummyModel:
        def __init__(self, model_path: str, device: str, compute_type: str) -> None:  # noqa: ARG002
            self.calls = [(model_path, device, compute_type)]

        def transcribe(self, audio: np.ndarray, language: str, beam_size: int):  # noqa: ARG002
            assert isinstance(audio, np.ndarray)
            info = types.SimpleNamespace(language_probability=0.75)
            return iter(segments), info

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=DummyModel))

    config = AsrConfig.model_validate(
        {"asr_backend": "faster-whisper", "asr_model": "model.bin", "asr_device": "cpu"}
    )
    engine = create_asr_engine(config)
    assert isinstance(engine, FasterWhisperEngine)

    partials: list[str] = []
    finals: list[str] = []
    confidences: list[float] = []
    engine.on_partial(lambda event: partials.append(event.text))
    engine.on_final(lambda event: finals.append(event.text))
    engine.on_confidence(lambda value: confidences.append(value))

    engine.transcribe([_audio_chunk()])

    assert partials == ["hello", "world"]
    assert finals == ["world"]
    assert confidences == [0.75]


def test_mlx_whisper_engine_transcribes(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyModel:
        def __init__(self, model_path: str, device: str) -> None:  # noqa: ARG002
            self.created = (model_path, device)

        def transcribe(self, audio: np.ndarray, language: str):  # noqa: ARG002
            return [
                {"text": "foo", "end": 0.4, "confidence": 0.42},
                types.SimpleNamespace(text="bar", end=1.0, avg_log_prob=np.log(0.5)),
            ]

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(WhisperModel=DummyModel))

    config = AsrConfig.model_validate(
        {"asr_backend": "mlx-whisper", "asr_model": "model.bin", "asr_device": "mps"}
    )
    engine = create_asr_engine(config)
    assert isinstance(engine, MlxWhisperEngine)

    partials: list[str] = []
    finals: list[str] = []
    engine.on_partial(lambda event: partials.append(event.text))
    engine.on_final(lambda event: finals.append(event.text))

    engine.transcribe([_audio_chunk()])

    assert partials == ["foo", "bar"]
    assert finals == ["bar"]


def test_create_asr_engine_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        create_asr_engine(
            AsrConfig.model_validate(
                {"asr_backend": "unknown", "asr_model": "model.bin", "asr_device": "cpu"}
            )
        )


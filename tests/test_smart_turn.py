from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_agent.asr import AsrEngine, AsrEvent
from voice_agent.config.models import TurnConfig, TtsConfig
from voice_agent.llm import LlmEngine
from voice_agent.runtime import SmartTurnError, SmartTurnOrchestrator
from voice_agent.telemetry import TelemetryClient
from voice_agent.vad import VadEvent, VadState


class FakeAsr(AsrEngine):
    def transcribe(self, audio_stream):  # type: ignore[override]
        self.emit_partial(AsrEvent(text="hello", timestamp=0.05))
        self.emit_final(AsrEvent(text="hello world", timestamp=0.2, confidence=0.92))


class FakeLlm(LlmEngine):
    def stream(self, prompt: str, *, max_tokens: int | None = None):  # type: ignore[override]
        yield f"Response to {prompt}"


class FakeTts:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, int]]] = []

    def speak(self, text: str, chunk_config: dict[str, int]) -> None:
        self.calls.append((text, chunk_config))


class FakeClock:
    def __init__(self) -> None:
        self._perf = 0.0
        self._wall = 1000.0

    def perf(self) -> float:
        self._perf += 0.05
        return self._perf

    def wall(self) -> float:
        self._wall += 1.0
        return self._wall


def _turn_config() -> TurnConfig:
    return TurnConfig.model_validate(
        {
            "turn_min_utterance_ms": 50,
            "turn_max_silence_ms": 600,
            "turn_max_turn_ms": 2_000,
            "turn_barge_in": True,
        }
    )


def _tts_config(tmp_path: Path) -> TtsConfig:
    return TtsConfig.model_validate(
        {
            "backend": "kitten",
            "voice_id": "test",
            "model_dir": str(tmp_path),
            "chunk_min_chars": 10,
            "chunk_min_ms": 100,
            "chunk_max_gap_ms": 200,
        }
    )


def test_smart_turn_success(tmp_path: Path) -> None:
    frames = [
        np.ones(1600, dtype=np.float32),
        np.ones(1600, dtype=np.float32) * 0.5,
    ]
    vad_events = [
        VadEvent(state=VadState.SPEECH, confidence=0.9, timestamp=0.05),
        VadEvent(state=VadState.SILENCE, confidence=0.2, timestamp=0.2),
    ]

    fake_clock = FakeClock()
    telemetry = TelemetryClient(clock=fake_clock.perf, time_source=fake_clock.wall)
    tts = FakeTts()
    orchestrator = SmartTurnOrchestrator(
        asr_engine=FakeAsr(),
        llm_engine=FakeLlm(),
        tts_engine=tts,
        turn_config=_turn_config(),
        tts_config=_tts_config(tmp_path),
        sample_rate=16000,
        telemetry=telemetry,
    )

    result = orchestrator.run(frames, vad_events)

    assert result.transcript == "hello world"
    assert result.response == "Response to hello world"
    assert result.utterance_ms == pytest.approx(150.0)
    assert "asr.latency" in result.metrics
    assert "llm.latency" in result.metrics
    assert "tts.latency" in result.metrics
    assert tts.calls and tts.calls[0][0] == "Response to hello world"


def test_short_utterance_raises(tmp_path: Path) -> None:
    frames = [np.ones(1600, dtype=np.float32)]
    vad_events = [
        VadEvent(state=VadState.SPEECH, confidence=0.9, timestamp=0.0),
        VadEvent(state=VadState.SILENCE, confidence=0.1, timestamp=0.01),
    ]

    fake_clock = FakeClock()
    telemetry = TelemetryClient(clock=fake_clock.perf, time_source=fake_clock.wall)
    orchestrator = SmartTurnOrchestrator(
        asr_engine=FakeAsr(),
        llm_engine=FakeLlm(),
        tts_engine=FakeTts(),
        turn_config=_turn_config(),
        tts_config=_tts_config(tmp_path),
        sample_rate=16000,
        telemetry=telemetry,
    )

    with pytest.raises(SmartTurnError):
        orchestrator.run(frames, vad_events)

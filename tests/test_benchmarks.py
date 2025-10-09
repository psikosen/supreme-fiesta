from __future__ import annotations

import numpy as np

from voice_agent.asr import AsrEngine
from voice_agent.benchmarks import (
    make_silence,
    measure_asr_latency,
    measure_llm_latency,
    measure_tts_latency,
)
from voice_agent.llm import LlmEngine
from voice_agent.telemetry import TelemetryClient


class FakeClock:
    def __init__(self) -> None:
        self._value = 0.0

    def tick(self, step: float = 0.02) -> float:
        self._value += step
        return self._value


class DummyAsr(AsrEngine):
    def transcribe(self, audio_stream):  # type: ignore[override]
        for _ in audio_stream:
            pass


class DummyLlm(LlmEngine):
    def stream(self, prompt: str, *, max_tokens: int | None = None):  # type: ignore[override]
        yield prompt
        yield " done"


class DummyTts:
    def synthesize(self, text: str, chunk_config: dict[str, int]):
        yield np.ones(10, dtype=np.float32)
        yield np.ones(5, dtype=np.float32) * 0.5


def test_measure_functions_emit_metrics() -> None:
    clock = FakeClock()
    telemetry = TelemetryClient(clock=lambda: clock.tick(0.01), time_source=lambda: clock.tick(0.001))

    asr_result = measure_asr_latency(DummyAsr(), make_silence(0.1, 16000), clock=clock.tick, telemetry=telemetry)
    assert asr_result.component == "asr"
    assert asr_result.metrics["latency_ms"] > 0

    llm_result = measure_llm_latency(DummyLlm(), "hi", max_tokens=4, clock=clock.tick, telemetry=telemetry)
    assert llm_result.component == "llm"
    assert llm_result.metrics["chunks"] == 2

    tts_result = measure_tts_latency(DummyTts(), "hello", {}, clock=clock.tick, telemetry=telemetry)
    assert tts_result.component == "tts"
    assert tts_result.metrics["chunks"] == 2


def test_make_silence_generates_bytes() -> None:
    silence = list(make_silence(0.05, 16000))
    assert len(silence) == 1
    assert isinstance(silence[0], bytes)

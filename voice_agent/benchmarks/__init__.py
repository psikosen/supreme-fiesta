"""Latency benchmarking helpers for ASR, LLM, and TTS."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Protocol

import numpy as np

from voice_agent.asr import AsrEngine
from voice_agent.llm import LlmEngine
from voice_agent.logging import get_logger
from voice_agent.telemetry import TelemetryClient

_LOG = get_logger(__name__)


class TtsSynthesiser(Protocol):
    def synthesize(self, text: str, chunk_config: Dict[str, int]) -> Iterable[np.ndarray]:
        ...


@dataclass
class BenchmarkResult:
    component: str
    metrics: Dict[str, Any]


def _emit_metrics(name: str, metrics: Dict[str, Any]) -> None:
    _LOG.info(
        "Benchmark metrics",
        extra={
            "classname": "Benchmark",
            "function": name,
            "system_section": "benchmark",
            "structured_message": json.dumps(metrics, ensure_ascii=False),
        },
    )


def measure_asr_latency(
    engine: AsrEngine,
    audio_stream: Iterable[bytes],
    *,
    clock: Callable[[], float] | None = None,
    telemetry: TelemetryClient | None = None,
) -> BenchmarkResult:
    timer = clock or time.perf_counter
    start = timer()
    engine.transcribe(audio_stream)
    end = timer()
    latency_ms = (end - start) * 1000.0
    metrics = {"latency_ms": latency_ms}
    if telemetry:
        telemetry.emit("asr.benchmark", **metrics)
    _emit_metrics("measure_asr_latency", metrics)
    return BenchmarkResult(component="asr", metrics=metrics)


def measure_llm_latency(
    engine: LlmEngine,
    prompt: str,
    *,
    max_tokens: int = 128,
    clock: Callable[[], float] | None = None,
    telemetry: TelemetryClient | None = None,
) -> BenchmarkResult:
    timer = clock or time.perf_counter
    start = timer()
    first_token_ms: float | None = None
    chunk_count = 0
    char_count = 0
    for chunk in engine.stream(prompt=prompt, max_tokens=max_tokens):
        now = timer()
        if first_token_ms is None:
            first_token_ms = (now - start) * 1000.0
        chunk_count += 1
        char_count += len(chunk)
    total_ms = (timer() - start) * 1000.0
    metrics = {
        "total_ms": total_ms,
        "time_to_first_chunk_ms": first_token_ms or total_ms,
        "chunks": chunk_count,
        "characters": char_count,
    }
    if telemetry:
        telemetry.emit("llm.benchmark", **metrics)
    _emit_metrics("measure_llm_latency", metrics)
    return BenchmarkResult(component="llm", metrics=metrics)


def measure_tts_latency(
    synthesiser: TtsSynthesiser,
    text: str,
    chunk_config: Dict[str, int],
    *,
    clock: Callable[[], float] | None = None,
    telemetry: TelemetryClient | None = None,
) -> BenchmarkResult:
    timer = clock or time.perf_counter
    start = timer()
    first_chunk_ms: float | None = None
    chunk_count = 0
    sample_count = 0
    for chunk in synthesiser.synthesize(text, chunk_config):
        now = timer()
        if first_chunk_ms is None:
            first_chunk_ms = (now - start) * 1000.0
        chunk_count += 1
        if isinstance(chunk, np.ndarray):
            sample_count += int(chunk.size)
    total_ms = (timer() - start) * 1000.0
    metrics = {
        "total_ms": total_ms,
        "time_to_first_chunk_ms": first_chunk_ms or total_ms,
        "chunks": chunk_count,
        "samples": sample_count,
    }
    if telemetry:
        telemetry.emit("tts.benchmark", **metrics)
    _emit_metrics("measure_tts_latency", metrics)
    return BenchmarkResult(component="tts", metrics=metrics)


def make_silence(duration_s: float, sample_rate: int) -> Iterator[bytes]:
    frames = int(duration_s * sample_rate)
    chunk = np.zeros(frames, dtype=np.float32)
    yield chunk.tobytes()


__all__ = [
    "BenchmarkResult",
    "measure_asr_latency",
    "measure_llm_latency",
    "measure_tts_latency",
    "make_silence",
]


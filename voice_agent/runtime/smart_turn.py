"""Smart-Turn v2 orchestration for coordinating ASR, LLM, and TTS."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Protocol, Sequence

import numpy as np

from voice_agent.asr import AsrEngine, AsrEvent
from voice_agent.config.models import TurnConfig, TtsConfig
from voice_agent.llm import LlmEngine
from voice_agent.logging import get_logger
from voice_agent.telemetry import TelemetryClient
from voice_agent.vad import VadEvent, VadState


class TtsEngine(Protocol):
    """Protocol for the subset of TTS behaviour required by the orchestrator."""

    def speak(self, text: str, chunk_config: Dict[str, int]) -> None:  # pragma: no cover - interface
        ...


@dataclass(slots=True)
class SmartTurnResult:
    """Captures the results of a Smart-Turn conversation cycle."""

    transcript: str
    response: str
    partials: Sequence[str]
    confidences: Sequence[float]
    utterance_ms: float
    metrics: Dict[str, float] = field(default_factory=dict)


class SmartTurnError(RuntimeError):
    """Raised when the Smart-Turn orchestration fails."""


class SmartTurnOrchestrator:
    """Coordinate audio VAD, ASR transcription, LLM completion, and TTS playback."""

    def __init__(
        self,
        *,
        asr_engine: AsrEngine,
        llm_engine: LlmEngine,
        tts_engine: TtsEngine,
        turn_config: TurnConfig,
        tts_config: TtsConfig,
        sample_rate: int,
        telemetry: TelemetryClient | None = None,
    ) -> None:
        self._asr = asr_engine
        self._llm = llm_engine
        self._tts = tts_engine
        self._turn = turn_config
        self._tts_config = tts_config
        self._sample_rate = sample_rate
        self._telemetry = telemetry or TelemetryClient()
        self._logger = get_logger(f"{__name__}.SmartTurnOrchestrator")
        self._tts_active = False

    def run(
        self,
        audio_frames: Iterable[np.ndarray],
        vad_stream: Iterable[VadEvent],
    ) -> SmartTurnResult:
        """Execute a single Smart-Turn conversation cycle."""

        if self._tts_active and not self._turn.barge_in:
            raise SmartTurnError("Barge-in disabled while synthesiser is active")

        start_index = len(self._telemetry.events)
        captured: List[np.ndarray] = []
        partials: List[str] = []
        finals: List[AsrEvent] = []
        confidences: List[float] = []

        speech_started = False
        speech_start_ts: float | None = None
        speech_end_ts: float | None = None
        processed_seconds = 0.0
        vad_iter = iter(vad_stream)
        next_event = self._advance_event(vad_iter)

        for chunk in audio_frames:
            frame = np.ascontiguousarray(chunk, dtype=np.float32).reshape(-1)
            chunk_seconds = frame.size / float(self._sample_rate)
            chunk_end = processed_seconds + chunk_seconds
            started_in_chunk = False

            while next_event and next_event.timestamp <= chunk_end + 1e-6:
                if next_event.state is VadState.SPEECH:
                    if not speech_started:
                        speech_start_ts = next_event.timestamp
                        speech_started = True
                        started_in_chunk = True
                        self._logger.info(
                            "Detected speech onset",
                            extra={
                                "classname": self.__class__.__name__,
                                "function": "run",
                                "system_section": "smart_turn",
                                "structured_message": json.dumps({"timestamp": speech_start_ts}),
                            },
                        )
                elif next_event.state is VadState.SILENCE and speech_started:
                    speech_end_ts = next_event.timestamp
                    speech_started = False
                    self._logger.info(
                        "Detected speech offset",
                        extra={
                            "classname": self.__class__.__name__,
                            "function": "run",
                            "system_section": "smart_turn",
                            "structured_message": json.dumps({"timestamp": speech_end_ts}),
                        },
                    )
                    break
                next_event = self._advance_event(vad_iter)

            if speech_started or started_in_chunk:
                captured.append(frame.copy())

            processed_seconds = chunk_end
            if speech_end_ts is not None and not speech_started:
                break

            if speech_start_ts is not None:
                elapsed_ms = (processed_seconds - speech_start_ts) * 1000.0
                if elapsed_ms >= self._turn.max_turn_ms:
                    self._logger.warning(
                        "Turn exceeded maximum duration",
                        extra={
                            "classname": self.__class__.__name__,
                            "function": "run",
                            "system_section": "smart_turn",
                            "structured_message": json.dumps({"elapsed_ms": elapsed_ms}),
                        },
                    )
                    speech_end_ts = processed_seconds
                    break

        if not captured or speech_start_ts is None:
            raise SmartTurnError("No utterance detected")

        if speech_end_ts is None:
            speech_end_ts = processed_seconds

        utterance_ms = max(0.0, (speech_end_ts - speech_start_ts) * 1000.0)
        if utterance_ms < float(self._turn.min_utterance_ms):
            raise SmartTurnError("Utterance shorter than minimum duration")

        audio_payload = [frame.tobytes() for frame in captured]
        metrics: Dict[str, float] = {"utterance_ms": utterance_ms}

        def _on_partial(event: AsrEvent) -> None:
            partials.append(event.text)

        def _on_final(event: AsrEvent) -> None:
            finals.append(event)

        def _on_confidence(value: float) -> None:
            confidences.append(value)

        self._asr.on_partial(_on_partial)
        self._asr.on_final(_on_final)
        self._asr.on_confidence(_on_confidence)

        with self._telemetry.span("asr.latency"):
            self._asr.transcribe(audio_payload)

        if not finals:
            raise SmartTurnError("ASR produced no final transcript")

        transcript = finals[-1].text

        with self._telemetry.span("llm.latency"):
            response = "".join(self._llm.stream(prompt=transcript))

        self._tts_active = True
        try:
            with self._telemetry.span("tts.latency"):
                self._tts.speak(response, {
                    "chunk_min_chars": self._tts_config.chunk_min_chars,
                    "chunk_min_ms": self._tts_config.chunk_min_ms,
                    "chunk_max_gap_ms": self._tts_config.chunk_max_gap_ms,
                })
        finally:
            self._tts_active = False

        for event in self._telemetry.events[start_index:]:
            if event.name.endswith("latency") and "duration_ms" in event.fields:
                metrics[event.name] = float(event.fields["duration_ms"])

        return SmartTurnResult(
            transcript=transcript,
            response=response,
            partials=tuple(partials),
            confidences=tuple(confidences),
            utterance_ms=utterance_ms,
            metrics=metrics,
        )

    @staticmethod
    def _advance_event(events: Iterator[VadEvent]) -> VadEvent | None:
        try:
            return next(events)
        except StopIteration:
            return None


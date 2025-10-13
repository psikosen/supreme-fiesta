"""Faster-Whisper backend implementation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from voice_agent.logging import get_logger

from .base import AsrEngine, AsrEvent
from .utils import stream_to_numpy

_LOG = get_logger(__name__)


class FasterWhisperEngine(AsrEngine):
    """Stream transcription through the faster-whisper package."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        compute_type: str | None = None,
        language: str = "en",
        beam_size: int = 1,
    ) -> None:
        super().__init__()
        module = sys.modules.get("faster_whisper")
        if module is None:
            spec = importlib.util.find_spec("faster_whisper")
            if spec is None:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "faster-whisper is not installed; install with poetry install --extras asr"
                )
            module = importlib.import_module("faster_whisper")
        WhisperModel = getattr(module, "WhisperModel")

        resolved_path = Path(model_path).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = Path.cwd() / resolved_path
        resolved_path = resolved_path.resolve()

        if not resolved_path.exists():
            error_message = f"Model missing at {resolved_path}"
            _LOG.error(
                "[Continuous skepticism (Sherlock Protocol)] Failed to load Faster-Whisper model",
                extra={
                    "classname": self.__class__.__name__,
                    "function": "__init__",
                    "system_section": "asr",
                    "error": error_message,
                    "structured_message": "Run voice-agent models pull asr/faster-whisper-base",
                },
            )
            raise RuntimeError(f"{error_message}. Run voice-agent models pull asr/faster-whisper-base")

        self.language = language
        self.beam_size = beam_size
        compute = compute_type or ("int8" if device == "cpu" else "float16")
        self._model = WhisperModel(str(resolved_path), device=device, compute_type=compute)

    def transcribe(self, audio_stream: Iterable[bytes]) -> None:
        audio = stream_to_numpy(audio_stream)
        if audio.size == 0:
            _LOG.warning(
                "[Continuous skepticism (Sherlock Protocol)] No frames to process",
                extra={
                    "classname": self.__class__.__name__,
                    "function": "transcribe",
                    "system_section": "asr",
                    "structured_message": "Empty audio stream received",
                },
            )
            return

        _LOG.info(
            "[Continuous skepticism (Sherlock Protocol)] Running inference",
            extra={
                "classname": self.__class__.__name__,
                "function": "transcribe",
                "system_section": "asr",
                "structured_message": "Starting Faster-Whisper transcription",
            },
        )

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
        )

        probability = getattr(info, "language_probability", None)
        if probability is not None:
            self.emit_confidence(float(probability))

        last_event: AsrEvent | None = None
        for segment in segments:
            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue
            end_time = float(getattr(segment, "end", 0.0))
            confidence = getattr(segment, "avg_log_prob", None)
            if confidence is not None:
                # Convert from log probability to linear scale when possible.
                confidence = float(np.exp(confidence))
            event = AsrEvent(text=text, timestamp=end_time, confidence=confidence)
            self.emit_partial(event)
            last_event = event

        if last_event is not None:
            self.emit_final(last_event)


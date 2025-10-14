"""MLX Whisper backend implementation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from voice_agent.logging import get_logger

from .base import AsrEngine, AsrEvent
from .utils import stream_to_numpy

_LOG = get_logger(__name__)


class MlxWhisperEngine(AsrEngine):
    """Thin adapter around the mlx-whisper project."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "mps",
        language: str = "en",
    ) -> None:
        super().__init__()
        module = sys.modules.get("mlx_whisper")
        if module is None:
            spec = importlib.util.find_spec("mlx_whisper")
            if spec is None:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "mlx-whisper is not installed; install with poetry install --extras asr"
                )
            module = importlib.import_module("mlx_whisper")  # type: ignore[import-not-found]
        mlx_whisper = module

        self.language = language
        self.device = device
        self._transcribe: Callable[[np.ndarray], Sequence[object]]

        if hasattr(mlx_whisper, "WhisperModel"):
            self._model = mlx_whisper.WhisperModel(str(model_path), device=device)

            def _call(audio: np.ndarray) -> Sequence[object]:
                return self._model.transcribe(audio, language=self.language)

            self._transcribe = _call
        else:
            load_model = getattr(mlx_whisper, "load_model", None)
            transcribe_fn = getattr(mlx_whisper, "transcribe", None)
            if load_model is None or transcribe_fn is None:
                raise RuntimeError("Unsupported mlx_whisper version")

            self._model = load_model(str(model_path), device=device)

            def _call(audio: np.ndarray) -> Sequence[object]:
                result = transcribe_fn(
                    self._model,
                    audio,
                    language=self.language,
                    device=device,
                    verbose=False,
                )
                if isinstance(result, dict) and "segments" in result:
                    return result["segments"]
                return result

            self._transcribe = _call

    def transcribe(self, audio_stream: Iterable[bytes]) -> None:
        audio = stream_to_numpy(audio_stream)
        if audio.size == 0:
            _LOG.warning(
                "No audio frames available for MLX Whisper",
                extra={
                    "classname": self.__class__.__name__,
                    "function": "transcribe",
                    "system_section": "asr",
                    "structured_message": "Empty audio stream received",
                    "derived_message": "Check microphone input and retry",
                },
            )
            return

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        _LOG.info(
            "Starting MLX Whisper transcription",
            extra={
                "classname": self.__class__.__name__,
                "function": "transcribe",
                "system_section": "asr",
                "structured_message": "Starting MLX Whisper transcription",
            },
        )

        segments = self._transcribe(audio)

        last_event: AsrEvent | None = None
        for segment in segments:
            if isinstance(segment, dict):
                text = str(segment.get("text", "")).strip()
                end = float(segment.get("end", segment.get("timestamp", 0.0)))
                confidence = segment.get("confidence") or segment.get("avg_log_prob")
            else:
                text = str(getattr(segment, "text", "")).strip()
                end = float(getattr(segment, "end", getattr(segment, "timestamp", 0.0)))
                confidence = getattr(segment, "confidence", None)
                if confidence is None:
                    confidence = getattr(segment, "avg_log_prob", None)
            if not text:
                continue
            if confidence is not None:
                confidence = float(confidence)
                if confidence <= 0:
                    confidence = float(np.exp(confidence))
            event = AsrEvent(text=text, timestamp=end, confidence=confidence)
            self.emit_partial(event)
            last_event = event

        if last_event is not None:
            self.emit_final(last_event)


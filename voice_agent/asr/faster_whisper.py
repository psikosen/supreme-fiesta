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


MODEL_ALIAS_PREFIX = "faster-whisper-"

MODEL_SIZE_ALIASES = {
    "tiny": "tiny",
    "tiny.en": "tiny.en",
    "base": "base",
    "base.en": "base.en",
    "small": "small",
    "small.en": "small.en",
    "medium": "medium",
    "medium.en": "medium.en",
    "large": "large-v3",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
}


def _resolve_pretrained_name(path: Path) -> str | None:
    """Return the Faster-Whisper model identifier for a missing asset path."""

    name = path.name.lower().replace("_", "-")
    if name.startswith(MODEL_ALIAS_PREFIX):
        candidate = name[len(MODEL_ALIAS_PREFIX) :]
    else:
        candidate = name
    return MODEL_SIZE_ALIASES.get(candidate)


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

        compute = compute_type or ("int8" if device == "cpu" else "float16")
        download_root: str | None = None
        model_location: str | Path = resolved_path

        pretrained_name = _resolve_pretrained_name(resolved_path)

        if not resolved_path.exists():
            if pretrained_name:
                resolved_path.mkdir(parents=True, exist_ok=True)
                download_root = str(resolved_path)
                model_location = pretrained_name
                _LOG.info(
                    "Downloading Faster-Whisper model",
                    extra={
                        "classname": self.__class__.__name__,
                        "function": "__init__",
                        "system_section": "asr",
                        "structured_message": str(
                            {
                                "pretrained": pretrained_name,
                                "destination": str(resolved_path),
                                "device": device,
                                "compute_type": compute,
                            }
                        ),
                        "derived_message": "Quality review: Missing ASR assets detected; downloading pretrained model",
                    },
                )
            else:
                error_message = f"Model missing at {resolved_path}"
                _LOG.error(
                    "Failed to load Faster-Whisper model",
                    extra={
                        "classname": self.__class__.__name__,
                        "function": "__init__",
                        "system_section": "asr",
                        "error": error_message,
                        "structured_message": "Provide a valid Faster-Whisper model path or alias",
                        "derived_message": "Quality review: ASR backend cannot locate requested model assets",
                    },
                )
                raise RuntimeError(
                    f"{error_message}. Provide a valid Faster-Whisper model path or use a supported alias"
                )
        elif resolved_path.is_dir():
            has_model_binary = any(resolved_path.glob("*.bin"))
            if not has_model_binary:
                if pretrained_name:
                    download_root = str(resolved_path)
                    model_location = pretrained_name
                    _LOG.warning(
                        "Re-downloading incomplete Faster-Whisper assets",
                        extra={
                            "classname": self.__class__.__name__,
                            "function": "__init__",
                            "system_section": "asr",
                            "structured_message": str(
                                {
                                    "pretrained": pretrained_name,
                                    "destination": str(resolved_path),
                                    "reason": "missing model binaries",
                                }
                            ),
                            "derived_message": "Quality review: Faster-Whisper asset directory incomplete; re-downloading pretrained model",
                        },
                    )
                else:
                    error_message = (
                        f"Model assets incomplete at {resolved_path}; expected model binary files"
                    )
                    _LOG.error(
                        "Failed to load Faster-Whisper model",
                        extra={
                            "classname": self.__class__.__name__,
                            "function": "__init__",
                            "system_section": "asr",
                            "error": error_message,
                            "structured_message": "Remove the directory or provide a valid Faster-Whisper model path",
                            "derived_message": "Quality review: Incomplete ASR assets detected without a known alias",
                        },
                    )
                    raise RuntimeError(
                        f"{error_message}. Provide a valid Faster-Whisper model path or use a supported alias"
                    )

        try:
            if download_root is not None:
                self._model = WhisperModel(
                    model_location,
                    device=device,
                    compute_type=compute,
                    download_root=download_root,
                )
            else:
                self._model = WhisperModel(str(model_location), device=device, compute_type=compute)
        except Exception as exc:  # pragma: no cover - backend import failures
            _LOG.error(
                "Failed to initialise Faster-Whisper model",
                extra={
                    "classname": self.__class__.__name__,
                    "function": "__init__",
                    "system_section": "asr",
                    "error": str(exc),
                    "structured_message": str({"model": str(model_location), "device": device}),
                    "derived_message": "Quality review: Faster-Whisper initialisation raised an exception",
                },
            )
            raise RuntimeError("Unable to initialise Faster-Whisper model") from exc

        self.language = language
        self.beam_size = beam_size

    def transcribe(self, audio_stream: Iterable[bytes]) -> None:
        audio = stream_to_numpy(audio_stream)
        if audio.size == 0:
            _LOG.warning(
                "No audio frames to process",
                extra={
                    "classname": self.__class__.__name__,
                    "function": "transcribe",
                    "system_section": "asr",
                    "structured_message": "Empty audio stream received",
                },
            )
            return

        _LOG.info(
            "Running Faster-Whisper inference",
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


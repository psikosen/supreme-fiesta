"""Silero VAD streaming integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from voice_agent.logging import get_logger
from voice_agent.vad.base import VadEvent, VadState, VadStream

try:  # pragma: no cover - optional dependency at runtime
    import onnxruntime as _ort
except Exception:  # pragma: no cover - we raise a structured error during initialisation
    _ort = None  # type: ignore[assignment]


class SileroVadError(RuntimeError):
    """Raised when Silero VAD cannot be initialised."""


class _SileroOnnxModel:
    """Minimal ONNX wrapper replicating Silero streaming behaviour."""

    def __init__(self, model_path: Path, sample_rate: int) -> None:
        if _ort is None:
            raise SileroVadError("onnxruntime is required for Silero VAD; install extras with poetry install")
        if not model_path.exists():
            raise FileNotFoundError(f"Silero VAD model missing: {model_path}")

        opts = _ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in _ort.get_available_providers() else None
        self._session = _ort.InferenceSession(str(model_path), providers=providers, sess_options=opts)
        self._sr = sample_rate
        if self._sr not in (8000, 16000):
            raise SileroVadError("Silero VAD supports 8 kHz or 16 kHz audio")
        self._context_width = 64 if self._sr == 16000 else 32
        self._frame_samples = 512 if self._sr == 16000 else 256
        self.reset_states()

    def reset_states(self, batch_size: int = 1) -> None:
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros((batch_size, self._context_width), dtype=np.float32)

    def __call__(self, chunk: np.ndarray, sample_rate: int) -> float:
        if sample_rate != self._sr:
            raise SileroVadError(f"Silero model initialised for {self._sr} Hz, received {sample_rate} Hz")

        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]
        if chunk.ndim != 2:
            raise SileroVadError(f"Unexpected chunk dimensions: {chunk.shape}")

        num_samples = chunk.shape[1]
        if num_samples < self._frame_samples:
            padded = np.zeros((chunk.shape[0], self._frame_samples), dtype=np.float32)
            padded[:, :num_samples] = chunk
            chunk = padded
        elif num_samples > self._frame_samples:
            raise SileroVadError(
                f"Silero VAD expects {self._frame_samples} samples per frame; received {num_samples}"
            )

        chunk = chunk.astype(np.float32, copy=False)
        model_input = np.concatenate([self._context, chunk], axis=1)
        ort_inputs = {
            "input": model_input,
            "state": self._state,
            "sr": np.array(self._sr, dtype=np.int64),
        }
        out, state = self._session.run(None, ort_inputs)
        self._state = state.astype(np.float32)
        self._context = model_input[:, -self._context_width :]
        prob = float(out.squeeze())
        return prob


@dataclass
class SileroVadConfig:
    model_path: Path
    sample_rate: int
    trigger_level: float
    release_level: float
    sensitivity: float
    frame_size: int = 512


class SileroVadStream(VadStream):
    """Stream VAD events by running the Silero ONNX model over audio frames."""

    def __init__(
        self,
        audio_source: Iterable[np.ndarray],
        config: SileroVadConfig,
        *,
        model: Optional[_SileroOnnxModel] = None,
    ) -> None:
        self._audio_source = audio_source
        self._config = config
        self._frame_size = config.frame_size
        self._frame_seconds = self._frame_size / float(config.sample_rate)
        self._model = model or _SileroOnnxModel(config.model_path, config.sample_rate)
        self._state = VadState.SILENCE
        self._timestamp = 0.0
        self._smoothed = config.release_level
        self._sensitivity = float(np.clip(config.sensitivity, 0.0, 1.0))
        self._trigger_hold_frames = max(1, int(round(self._compute_hold_seconds(0.04) / self._frame_seconds)))
        self._release_hold_frames = max(1, int(round(self._compute_hold_seconds(0.12) / self._frame_seconds)))
        self._active_frames = 0
        self._inactive_frames = 0
        self._logger = get_logger(f"{__name__}.SileroVadStream")
        self._model.reset_states()

    def _compute_hold_seconds(self, base: float) -> float:
        span = 0.18  # seconds of variability across sensitivity range
        return max(0.02, base + (1.0 - self._sensitivity) * span)

    def _smooth(self, prob: float) -> float:
        alpha = 0.25 + 0.6 * self._sensitivity
        self._smoothed = alpha * prob + (1.0 - alpha) * self._smoothed
        return self._smoothed

    def _frame_iter(self, chunk: np.ndarray) -> Iterator[np.ndarray]:
        data = np.asarray(chunk, dtype=np.float32).flatten()
        if data.size == 0:
            return
        for start in range(0, data.size, self._frame_size):
            frame = data[start : start + self._frame_size]
            if frame.size < self._frame_size:
                padded = np.zeros(self._frame_size, dtype=np.float32)
                padded[: frame.size] = frame
                frame = padded
            yield frame

    def __iter__(self) -> Iterator[VadEvent]:
        for chunk in self._audio_source:
            for frame in self._frame_iter(chunk):
                prob = self._model(frame, self._config.sample_rate)
                confidence = self._smooth(prob)
                self._timestamp += self._frame_seconds
                if confidence >= self._config.trigger_level:
                    self._active_frames += 1
                    self._inactive_frames = 0
                elif confidence <= self._config.release_level:
                    self._inactive_frames += 1
                    self._active_frames = 0
                else:
                    self._active_frames = 0
                    self._inactive_frames = 0

                if self._state is VadState.SILENCE and self._active_frames >= self._trigger_hold_frames:
                    self._state = VadState.SPEECH
                    self._active_frames = 0
                    self._logger.info(
                        "[Continuous skepticism (Sherlock Protocol)] Speech detected",
                        extra={
                            "classname": self.__class__.__name__,
                            "function": "__iter__",
                            "system_section": "vad",
                        },
                    )
                    yield VadEvent(state=VadState.SPEECH, confidence=confidence, timestamp=self._timestamp)
                elif self._state is VadState.SPEECH and self._inactive_frames >= self._release_hold_frames:
                    self._state = VadState.SILENCE
                    self._inactive_frames = 0
                    self._logger.info(
                        "[Continuous skepticism (Sherlock Protocol)] Speech ended",
                        extra={
                            "classname": self.__class__.__name__,
                            "function": "__iter__",
                            "system_section": "vad",
                        },
                    )
                    yield VadEvent(state=VadState.SILENCE, confidence=confidence, timestamp=self._timestamp)

    @classmethod
    def from_numpy(
        cls,
        audio: np.ndarray,
        *,
        model_path: Path,
        sample_rate: int,
        trigger_level: float,
        release_level: float,
        sensitivity: float,
        frame_size: int = 512,
        model: Optional[_SileroOnnxModel] = None,
    ) -> "SileroVadStream":
        def generator() -> Iterator[np.ndarray]:
            yield audio

        config = SileroVadConfig(
            model_path=model_path,
            sample_rate=sample_rate,
            trigger_level=trigger_level,
            release_level=release_level,
            sensitivity=sensitivity,
            frame_size=frame_size,
        )
        return cls(generator(), config, model=model)


__all__ = ["SileroVadConfig", "SileroVadStream", "SileroVadError"]

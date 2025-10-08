"""Cross-platform audio input/output helpers."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import Iterable, List, Optional

import numpy as np

from voice_agent.logging import get_logger

_LOG = get_logger(__name__)


try:  # pragma: no cover - fallback path
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - optional dependency guard
    sd = None  # type: ignore[assignment]
    _LOG.error("sounddevice import failed", extra={"error": str(exc), "system_section": "audio"})

try:  # pragma: no cover - optional fallback
    import pyaudio
except Exception:  # pragma: no cover
    pyaudio = None  # type: ignore[assignment]


class AudioBackendError(RuntimeError):
    """Raised when the audio backend is unavailable."""


@dataclasses.dataclass
class AudioDeviceInfo:
    """Represents an audio device discovered by sounddevice."""

    id: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


class AudioIO:
    """Audio loopback utilities using sounddevice with pyaudio fallback."""

    def __init__(self, samplerate: int = 16000, channels: int = 1) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self._logger = logging.getLogger(f"{__name__}.AudioIO")
        configure = getattr(self._logger, "configure", None)
        if configure:
            configure()

    @staticmethod
    def _require_sounddevice() -> sd:  # type: ignore[name-defined]
        if sd is None:
            raise AudioBackendError("sounddevice is not available; install with poetry install --extras audio")
        return sd

    def list_devices(self) -> List[AudioDeviceInfo]:
        backend = self._require_sounddevice()
        devices = backend.query_devices()
        results: List[AudioDeviceInfo] = []
        for idx, info in enumerate(devices):
            results.append(
                AudioDeviceInfo(
                    id=idx,
                    name=str(info["name"]),
                    max_input_channels=int(info["max_input_channels"]),
                    max_output_channels=int(info["max_output_channels"]),
                    default_samplerate=float(info.get("default_samplerate", self.samplerate)),
                ),
            )
        return results

    def _select_device(self, name_or_index: Optional[str], kind: str) -> Optional[int]:
        if name_or_index is None:
            return None
        backend = self._require_sounddevice()
        devices = backend.query_devices()
        if name_or_index.isdigit():
            idx = int(name_or_index)
            if 0 <= idx < len(devices):
                return idx
            raise AudioBackendError(f"Invalid {kind} device index: {name_or_index}")

        for idx, info in enumerate(devices):
            if name_or_index.lower() in str(info["name"]).lower():
                return idx
        raise AudioBackendError(f"Could not find {kind} device matching '{name_or_index}'")

    def record_loopback(
        self,
        seconds: float = 3.0,
        input_device: Optional[str] = None,
        output_device: Optional[str] = None,
    ) -> np.ndarray:
        backend = self._require_sounddevice()
        frames = int(seconds * self.samplerate)
        device_kwargs = {
            "samplerate": self.samplerate,
            "channels": self.channels,
            "dtype": "float32",
        }
        input_index = self._select_device(input_device, "input")
        if input_index is not None:
            device_kwargs["device"] = (input_index, None)

        _LOG.info(
            "Recording audio",
            extra={
                "system_section": "audio",
                "structured_message": "Starting capture",
                "classname": self.__class__.__name__,
                "function": "record_loopback",
            },
        )
        data = backend.rec(frames, **device_kwargs)
        backend.wait()

        playback_kwargs = {
            "samplerate": self.samplerate,
        }
        output_index = self._select_device(output_device, "output")
        if output_index is not None:
            playback_kwargs["device"] = output_index

        backend.play(data, **playback_kwargs)
        backend.wait()
        return data

    def playback(self, audio: np.ndarray, output_device: Optional[str] = None) -> None:
        backend = self._require_sounddevice()
        playback_kwargs = {
            "samplerate": self.samplerate,
        }
        output_index = self._select_device(output_device, "output")
        if output_index is not None:
            playback_kwargs["device"] = output_index
        backend.play(audio, **playback_kwargs)
        backend.wait()

    @contextlib.contextmanager
    def pyaudio_stream(self, frames_per_buffer: int = 1024) -> Iterable[bytes]:  # pragma: no cover - fallback
        if pyaudio is None:
            raise AudioBackendError("pyaudio is unavailable")
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.samplerate,
            input=True,
            output=True,
            frames_per_buffer=frames_per_buffer,
        )
        try:
            yield stream
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()


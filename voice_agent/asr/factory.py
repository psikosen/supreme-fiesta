"""Factory helpers for ASR engines."""

from __future__ import annotations

from voice_agent.config.models import AsrConfig

from .base import AsrEngine
from .faster_whisper import FasterWhisperEngine
from .mlx_whisper import MlxWhisperEngine


def create_asr_engine(config: AsrConfig) -> AsrEngine:
    """Instantiate the configured ASR backend."""

    backend = config.backend
    if backend == "faster-whisper":
        return FasterWhisperEngine(
            model_path=config.model_path,
            device=config.device,
        )
    if backend == "mlx-whisper":
        return MlxWhisperEngine(
            model_path=config.model_path,
            device=config.device,
        )
    raise ValueError(f"Unsupported ASR backend: {backend}")


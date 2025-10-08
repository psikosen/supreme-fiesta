"""ASR abstraction layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass
class AsrEvent:
    text: str
    timestamp: float
    confidence: float | None = None


class AsrEngine:
    """Base ASR engine."""

    def __init__(self) -> None:
        self._on_partial: Callable[[AsrEvent], None] | None = None
        self._on_final: Callable[[AsrEvent], None] | None = None
        self._on_confidence: Callable[[float], None] | None = None

    def on_partial(self, callback: Callable[[AsrEvent], None]) -> None:
        self._on_partial = callback

    def on_final(self, callback: Callable[[AsrEvent], None]) -> None:
        self._on_final = callback

    def on_confidence(self, callback: Callable[[float], None]) -> None:
        self._on_confidence = callback

    def emit_partial(self, event: AsrEvent) -> None:
        if self._on_partial:
            self._on_partial(event)

    def emit_final(self, event: AsrEvent) -> None:
        if self._on_final:
            self._on_final(event)

    def emit_confidence(self, value: float) -> None:
        if self._on_confidence:
            self._on_confidence(value)

    def transcribe(self, audio_stream: Iterable[bytes]) -> None:
        raise NotImplementedError

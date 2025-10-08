"""Base VAD event stream."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class VadState(str, Enum):
    SILENCE = "silence"
    SPEECH = "speech"


@dataclass
class VadEvent:
    state: VadState
    confidence: float
    timestamp: float


class VadStream:
    """Base class for streaming VAD implementations."""

    def __iter__(self) -> Iterable[VadEvent]:
        raise NotImplementedError

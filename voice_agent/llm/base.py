"""LLM runner interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class LlmConfig:
    model_path: str
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    context_window: int = 2048


class LlmEngine:
    """Base streaming LLM engine."""

    def __iter__(self) -> Iterable[str]:
        raise NotImplementedError

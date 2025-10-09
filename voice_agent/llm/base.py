"""LLM runner interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(slots=True)
class LlmConfig:
    """Runtime configuration for an LLM engine."""

    model_path: Path
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    context_window: int = 2048


class LlmEngine:
    """Base streaming LLM engine."""

    def stream(self, prompt: str, *, max_tokens: int | None = None) -> Iterable[str]:
        """Yield text chunks produced by the model for *prompt*."""

        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - compatibility shim
        raise NotImplementedError(
            "Direct iteration is not supported; call stream(prompt=...) instead."
        )

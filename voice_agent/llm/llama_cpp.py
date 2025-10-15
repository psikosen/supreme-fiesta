"""llama.cpp GGUF streaming inference runner."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, Iterable, Iterator

if TYPE_CHECKING:  # pragma: no cover - import used for typing only
    from llama_cpp import Llama

from voice_agent.llm.base import LlmConfig, LlmEngine
from voice_agent.llm.gguf_metadata import read_general_architecture
from voice_agent.llm.hf_assets import ensure_local_gguf
from voice_agent.logging import get_logger

_LOG = get_logger(__name__)

_ARCHITECTURE_GUIDANCE: dict[str, str] = {
    "lfm2": (
        "LiquidAI LFM2 GGUF models require a llama.cpp build with Liquid Fourier Mamba support. "
        "Install LiquidAI's patched llama.cpp distribution or choose a model converted from a supported architecture."
    ),
}


class LlamaCppEngine(LlmEngine):
    """Wrapper around :mod:`llama_cpp` providing streaming completions."""

    def __init__(
        self,
        config: LlmConfig,
        *,
        n_threads: int | None = None,
        n_gpu_layers: int | None = None,
    ) -> None:
        self._config = config
        self._n_threads = n_threads
        self._n_gpu_layers = n_gpu_layers
        self._model = self._initialise_model()

    def _initialise_model(self) -> "Llama":
        model_path = ensure_local_gguf(self._config.model_path.expanduser())
        if not model_path.exists():
            _LOG.error(
                "Failed to load llama.cpp model",
                extra={
                    "classname": self.__class__.__name__,
                    "function": "_initialise_model",
                    "system_section": "llm",
                    "error": f"Model missing at {model_path}",
                    "structured_message": "Verify GGUF asset path",
                },
            )
            raise FileNotFoundError(f"Model missing at {model_path}")

        metadata_architecture = read_general_architecture(model_path)

        init_kwargs: Dict[str, object] = {
            "model_path": str(model_path),
            "n_ctx": self._config.context_window,
        }
        if self._n_threads is not None:
            init_kwargs["n_threads"] = self._n_threads
        if self._n_gpu_layers is not None:
            init_kwargs["n_gpu_layers"] = self._n_gpu_layers

        structured_payload = {k: v for k, v in init_kwargs.items() if k != "model_path"}
        if metadata_architecture:
            structured_payload["architecture"] = metadata_architecture

        _LOG.info(
            "Initialising llama.cpp",
            extra={
                "classname": self.__class__.__name__,
                "function": "_initialise_model",
                "system_section": "llm",
                "structured_message": str(structured_payload),
            },
        )
        from llama_cpp import Llama

        try:
            return Llama(**init_kwargs)
        except ValueError as exc:
            error_message = str(exc)
            architecture = _extract_unknown_architecture(error_message)
            if architecture is None:
                architecture = metadata_architecture
            if architecture:
                guidance = _ARCHITECTURE_GUIDANCE.get(architecture)
                if guidance is None:
                    guidance = (
                        f"Unsupported GGUF architecture '{architecture}' detected. "
                        "Upgrade llama-cpp-python to a build that recognises this architecture "
                        "or choose a model converted with a supported base architecture."
                    )
            else:
                guidance = "llama.cpp failed to load the GGUF model. Verify compatibility."

            derived_message = (
                "[Continuous skepticism (Sherlock Protocol)] "
                "Investigate llama.cpp GGUF compatibility failure"
            )
            _LOG.error(
                "Failed to load llama.cpp model",
                extra={
                    "classname": self.__class__.__name__,
                    "function": "_initialise_model",
                    "system_section": "llm",
                    "error": error_message,
                    "structured_message": guidance,
                    "derived_message": derived_message,
                },
            )
            raise RuntimeError(guidance) from exc

    def stream(self, prompt: str, *, max_tokens: int | None = None) -> Iterable[str]:
        sampling = {
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "repeat_penalty": self._config.repeat_penalty,
        }
        if max_tokens is not None:
            sampling["max_tokens"] = max_tokens

        _LOG.info(
            "Starting llama.cpp stream",
            extra={
                "classname": self.__class__.__name__,
                "function": "stream",
                "system_section": "llm",
                "structured_message": str(sampling),
            },
        )

        request_kwargs = {
            "prompt": prompt,
            "stream": True,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "repeat_penalty": self._config.repeat_penalty,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens

        try:
            iterator: Iterator[Dict[str, object]] = self._model.create_completion(**request_kwargs)
        except Exception as exc:  # pragma: no cover - backend raises varying errors
            _LOG.error(
                "llama.cpp completion failed",
                extra={
                    "classname": self.__class__.__name__,
                    "function": "stream",
                    "system_section": "llm",
                    "error": str(exc),
                    "structured_message": "Inspect llama.cpp backend logs",
                },
            )
            raise RuntimeError("llama.cpp completion failed") from exc

        for chunk in iterator:
            choices = chunk.get("choices") if isinstance(chunk, dict) else None
            if not choices:
                continue
            text = choices[0].get("text") if isinstance(choices[0], dict) else None
            if text:
                yield str(text)


def create_llm_engine(config: LlmConfig) -> LlmEngine:
    """Instantiate an :class:`LlamaCppEngine` for the provided config."""

    return LlamaCppEngine(config)


def _extract_unknown_architecture(message: str) -> str | None:
    """Return the architecture referenced in a llama.cpp unknown architecture error."""

    match = re.search(r"unknown model architecture: '([^']+)'", message)
    if match:
        return match.group(1)
    return None

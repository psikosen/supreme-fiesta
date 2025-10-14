"""Static registry describing supported model assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    filename: str
    size_mb: float
    sha256: str
    description: str

    def path(self, root: Path) -> Path:
        return root / self.filename


MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "llm/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF": ModelEntry(
        model_id="llm/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        size_mb=637.9,
        sha256="placeholder-sha256-tinyllama-q4km",
        description="TinyLlama 1.1B Chat Q4_K_M quantized GGUF",
    ),
    "tts/kitten-nano-0.2/onnx": ModelEntry(
        model_id="tts/kitten-nano-0.2/onnx",
        filename="kitten_tts_nano_v0_2.onnx",
        size_mb=23.8,
        sha256="placeholder-sha256-kittten-onnx",
        description="Kitten TTS Nano 0.2 model",
    ),
    "tts/kitten-nano-0.2/voices": ModelEntry(
        model_id="tts/kitten-nano-0.2/voices",
        filename="voices.npz",
        size_mb=2.0,
        sha256="placeholder-sha256-kittten-voices",
        description="Kitten TTS voice inventory",
    ),
    "tts/kitten-nano-0.2/config": ModelEntry(
        model_id="tts/kitten-nano-0.2/config",
        filename="config.json",
        size_mb=0.1,
        sha256="placeholder-sha256-kittten-config",
        description="Kitten TTS runtime configuration",
    ),
}


def resolve_model(model_id: str) -> ModelEntry:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:  # pragma: no cover - simple mapping
        raise KeyError(f"Unknown model id: {model_id}") from exc

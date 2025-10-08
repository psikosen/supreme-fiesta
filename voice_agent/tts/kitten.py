"""Kitten TTS integration using ONNX Runtime or deterministic fallback."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional

import numpy as np

from voice_agent.audio import AudioBackendError, AudioIO
from voice_agent.logging import get_logger

try:  # pragma: no cover - optional heavy dependency
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None  # type: ignore[assignment]

_LOG = get_logger(__name__)

MODEL_FILE = "kitten_tts_nano_v0_2.onnx"
VOICES_FILE = "voices.npz"
CONFIG_FILE = "config.json"
DEFAULT_SAMPLE_RATE = 22050


@dataclass
class KittenVoice:
    voice_id: str
    tags: List[str]


@dataclass
class KittenVoiceInventory:
    voices: List[KittenVoice]

    def find(self, voice_id: str) -> KittenVoice:
        for voice in self.voices:
            if voice.voice_id == voice_id:
                return voice
        raise KeyError(f"Voice {voice_id} not found")

    def default(self) -> KittenVoice:
        if not self.voices:
            raise RuntimeError("No voices defined")
        return self.voices[0]


def load_voice_inventory(voices_path: Path) -> KittenVoiceInventory:
    data = np.load(voices_path, allow_pickle=True)
    voices: List[KittenVoice] = []
    if "voices" in data:
        entries = data["voices"]
        for entry in entries:
            if isinstance(entry, np.void) and hasattr(entry, "tolist"):
                entry = entry.tolist()
            if isinstance(entry, np.ndarray):
                entry = entry.tolist()
            if isinstance(entry, (list, tuple)) and entry:
                voice_id = str(entry[0])
                tags = [str(tag) for tag in entry[1:]]
            elif isinstance(entry, dict):
                voice_id = str(entry.get("voice_id"))
                tags_raw = entry.get("tags", [])
                tags = [str(tag) for tag in tags_raw] if isinstance(tags_raw, Iterable) else []
            else:
                voice_id = str(entry)
                tags = []
            voices.append(KittenVoice(voice_id=voice_id, tags=tags))
    else:
        for key in data.files:
            if key.startswith("voice_"):
                voices.append(KittenVoice(voice_id=key, tags=[str(x) for x in np.atleast_1d(data[key])]))
    if not voices:
        voices.append(KittenVoice("kitten/default", []))
    return KittenVoiceInventory(voices=voices)


class KittenTTS:
    """Synthesise speech via Kitten ONNX runtime or deterministic fallback."""

    def __init__(self, model_dir: Path, voice_id: str, audio: Optional[AudioIO] = None) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.voice_id = voice_id
        self.audio = audio or AudioIO(samplerate=DEFAULT_SAMPLE_RATE)
        self.assets = self._load_assets()
        self.inventory = load_voice_inventory(self.assets["voices"])
        try:
            self.voice = self.inventory.find(voice_id)
        except KeyError:
            _LOG.warning(
                "Unknown voice id requested",
                extra={
                    "system_section": "tts",
                    "structured_message": f"Voice {voice_id} missing, falling back",
                    "classname": self.__class__.__name__,
                    "function": "__init__",
                },
            )
            self.voice = self.inventory.default()
        self.session = self._load_session()
        self.sample_rate = self._load_sample_rate()

    def _asset_path(self, name: str) -> Path:
        path = self.model_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Kitten asset missing: {path}")
        return path

    def _load_assets(self) -> Dict[str, Path]:
        return {
            "model": self._asset_path(MODEL_FILE),
            "voices": self._asset_path(VOICES_FILE),
            "config": self._asset_path(CONFIG_FILE),
        }

    def _load_session(self):  # pragma: no cover - heavy dependency
        if ort is None:
            _LOG.warning(
                "onnxruntime not available, falling back to tone synth",
                extra={
                    "system_section": "tts",
                    "structured_message": "Install poetry install --extras tts",
                    "classname": self.__class__.__name__,
                    "function": "_load_session",
                },
            )
            return None
        return ort.InferenceSession(str(self.assets["model"]), providers=["CPUExecutionProvider"])

    def _load_sample_rate(self) -> int:
        with self.assets["config"].open("r", encoding="utf-8") as fh:
            config = json.load(fh)
        return int(config.get("sample_rate", DEFAULT_SAMPLE_RATE))

    def _chunk_text(
        self, text: str, min_chars: int, min_ms: int, max_gap_ms: int,
    ) -> Generator[str, None, None]:
        sentences = []
        current = []
        for char in text:
            current.append(char)
            if char in ".!?" and len(current) >= min_chars:
                sentences.append("".join(current).strip())
                current = []
        tail = "".join(current).strip()
        if tail:
            sentences.append(tail)
        if not sentences:
            yield text
            return
        for sentence in sentences:
            yield sentence

    def _sine_wave(self, duration_s: float, frequency: float = 220.0) -> np.ndarray:
        t = np.linspace(0, duration_s, int(duration_s * self.sample_rate), endpoint=False)
        waveform = 0.2 * np.sin(2 * math.pi * frequency * t)
        envelope = np.linspace(0.1, 1.0, waveform.size)
        return (waveform * envelope).astype(np.float32)

    def synthesize(self, text: str, chunk_config: Dict[str, int]) -> Iterable[np.ndarray]:
        min_chars = chunk_config.get("chunk_min_chars", 40)
        min_ms = chunk_config.get("chunk_min_ms", 600)
        max_gap = chunk_config.get("chunk_max_gap_ms", 250)
        for chunk in self._chunk_text(text, min_chars, min_ms, max_gap):
            if self.session is None:  # fallback tone generator
                duration_s = max(min_ms / 1000.0, len(chunk) / 10 * 0.1)
                frequency = 200 + (len(chunk) % 40) * 5
                yield self._sine_wave(duration_s, frequency)
            else:  # pragma: no cover - requires real assets
                inputs = self._prepare_inputs(chunk)
                outputs = self.session.run(None, inputs)
                audio = self._postprocess(outputs)
                yield audio

    def _prepare_inputs(self, text: str):  # pragma: no cover - requires real assets
        return {"text": text, "voice": self.voice.voice_id}

    def _postprocess(self, outputs):  # pragma: no cover - requires real assets
        if isinstance(outputs, list):
            outputs = outputs[0]
        audio = np.asarray(outputs, dtype=np.float32).flatten()
        return audio

    def speak(self, text: str, chunk_config: Dict[str, int]) -> None:
        for audio_chunk in self.synthesize(text, chunk_config):
            try:
                self.audio.playback(audio_chunk)
            except AudioBackendError as exc:  # pragma: no cover - hardware specific
                _LOG.error(
                    "Audio playback failed",
                    extra={
                        "system_section": "tts",
                        "structured_message": str(exc),
                        "classname": self.__class__.__name__,
                        "function": "speak",
                        "error": str(exc),
                    },
                )
                break

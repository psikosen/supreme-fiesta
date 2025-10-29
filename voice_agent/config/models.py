"""Configuration models and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal

from pydantic import BaseModel, Field, field_validator


class AsrConfig(BaseModel):
    backend: Literal["mlx-whisper", "faster-whisper"] = Field(alias="asr_backend")
    model_path: Path = Field(alias="asr_model")
    device: str = Field(alias="asr_device")

    @field_validator("model_path")
    def _check_model_path(cls, value: Path) -> Path:
        if not value:
            raise ValueError("ASR model path is required")
        return value


class LlmConfig(BaseModel):
    backend: Literal["llama-cpp"] = Field(alias="llm_backend")
    model_path: Path = Field(alias="llm_model")
    context_window: int = Field(alias="llm_context_window", ge=512)
    temperature: float = Field(alias="llm_temperature", ge=0.0)
    top_p: float = Field(alias="llm_top_p", ge=0.0, le=1.0)
    repeat_penalty: float = Field(alias="llm_repeat_penalty", ge=0.5, le=2.0)

    @field_validator("model_path")
    def _check_llm_path(cls, value: Path) -> Path:
        if not value:
            raise ValueError("LLM model path is required")
        return value


class VadConfig(BaseModel):
    model_path: Path = Field(alias="vad_model")
    trigger_level: float = Field(alias="vad_trigger_level", ge=0.0, le=1.0)
    release_level: float = Field(alias="vad_release_level", ge=0.0, le=1.0)
    sensitivity: float = Field(alias="vad_sensitivity", ge=0.0, le=1.0)


class TurnConfig(BaseModel):
    min_utterance_ms: int = Field(alias="turn_min_utterance_ms", ge=0)
    max_silence_ms: int = Field(alias="turn_max_silence_ms", ge=0)
    max_turn_ms: int = Field(alias="turn_max_turn_ms", ge=0)
    barge_in: bool = Field(alias="turn_barge_in")


class TtsConfig(BaseModel):
    backend: Literal["kitten"]
    voice_id: str
    model_dir: Path
    chunk_min_chars: int
    chunk_min_ms: int
    chunk_max_gap_ms: int

    @field_validator("model_dir")
    def _check_model_dir(cls, value: Path) -> Path:
        if not value:
            raise ValueError("TTS model_dir is required")
        return value


class ProfileConfig(BaseModel):
    asr: AsrConfig
    llm: LlmConfig
    vad: VadConfig
    turn: TurnConfig
    tts: TtsConfig

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> ProfileConfig:
        combined = {
            "asr_backend": data.get("asr_backend"),
            "asr_model": data.get("asr_model"),
            "asr_device": data.get("asr_device", "cpu"),
            "llm_backend": data.get("llm_backend"),
            "llm_model": data.get("llm_model"),
            "llm_context_window": data.get("llm_context_window", 2048),
            "llm_temperature": data.get("llm_temperature", 0.7),
            "llm_top_p": data.get("llm_top_p", 0.95),
            "llm_repeat_penalty": data.get("llm_repeat_penalty", 1.1),
            "vad_model": data.get("vad_model"),
            "vad_trigger_level": data.get("vad_trigger_level", 0.5),
            "vad_release_level": data.get("vad_release_level", 0.3),
            "vad_sensitivity": data.get("vad_sensitivity", 0.5),
            "turn_min_utterance_ms": data.get("turn_min_utterance_ms", 600),
            "turn_max_silence_ms": data.get("turn_max_silence_ms", 700),
            "turn_max_turn_ms": data.get("turn_max_turn_ms", 15000),
            "turn_barge_in": data.get("turn_barge_in", True),
            "tts": data.get("tts", {}),
        }
        tts_data = combined.pop("tts")
        if not isinstance(tts_data, dict):
            raise ValueError("tts configuration must be a table")
        return cls(
            asr=AsrConfig(**combined),
            llm=LlmConfig(**combined),
            vad=VadConfig(**combined),
            turn=TurnConfig(**combined),
            tts=TtsConfig(**tts_data),
        )


class VoiceAgentConfig(BaseModel):
    active_profile: str
    profiles: Dict[str, ProfileConfig]

    @classmethod
    def from_toml(cls, data: Dict[str, object]) -> VoiceAgentConfig:
        active = str(data.get("active_profile", "default"))
        raw_profiles = data.get("profiles")
        if not isinstance(raw_profiles, dict):
            raise ValueError("profiles table missing from config")
        parsed = {name: ProfileConfig.from_dict(value) for name, value in raw_profiles.items() if isinstance(value, dict)}
        if active not in parsed:
            raise ValueError(f"Active profile '{active}' missing from config")
        return cls(active_profile=active, profiles=parsed)

    def ensure_paths(self) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        profile = self.profiles[self.active_profile]
        paths["asr"] = profile.asr.model_path
        paths["llm"] = profile.llm.model_path
        paths["vad"] = profile.vad.model_path
        paths["tts_model_dir"] = profile.tts.model_dir
        return paths

    def switch_profile(self, name: str) -> None:
        if name not in self.profiles:
            raise KeyError(f"Profile '{name}' not found")
        self.active_profile = name

    def active(self) -> ProfileConfig:
        return self.profiles[self.active_profile]

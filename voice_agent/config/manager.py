"""Config file helpers."""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Dict, Iterable

import platformdirs
import tomllib

from voice_agent.logging import get_logger

from .models import VoiceAgentConfig

_LOG = get_logger(__name__)
CONFIG_FILE_NAME = "config.toml"


class ConfigManager:
    """Load and persist configuration profiles."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_dir = config_path.parent if config_path else Path(platformdirs.user_config_path("voice-agent"))
        self.config_path = config_path or (self.config_dir / CONFIG_FILE_NAME)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            self._write_default()

    def load(self) -> VoiceAgentConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file missing at {self.config_path}")
        with self.config_path.open("rb") as fh:
            data = tomllib.load(fh)
        config = VoiceAgentConfig.from_toml(data)
        self._validate_assets(config)
        return config

    def save(self, config: VoiceAgentConfig) -> None:
        import tomli_w

        serializable = self._to_serializable(config)
        with self.config_path.open("wb") as fh:
            tomli_w.dump(serializable, fh)

    def _write_default(self) -> None:
        default_path = importlib.resources.files("voice_agent.config").joinpath("default_config.toml")
        with default_path.open("rb") as source, self.config_path.open("wb") as target:
            target.write(source.read())

    def list_profiles(self) -> Iterable[str]:
        config = self.load()
        return sorted(config.profiles.keys())

    def use_profile(self, profile_name: str) -> VoiceAgentConfig:
        config = self.load()
        config.switch_profile(profile_name)
        self.save(config)
        _LOG.info(
            "Switched profile",
            extra={
                "system_section": "config",
                "structured_message": f"Active profile set to {profile_name}",
                "classname": self.__class__.__name__,
                "function": "use_profile",
            },
        )
        return config

    def _validate_assets(self, config: VoiceAgentConfig) -> None:
        missing: Dict[str, str] = {}
        for name, path in config.ensure_paths().items():
            if not path.expanduser().exists():
                missing[name] = str(path)
        if missing:
            _LOG.warning(
                "Missing asset paths",
                extra={
                    "system_section": "config",
                    "structured_message": json.dumps(missing),
                    "classname": self.__class__.__name__,
                    "function": "_validate_assets",
                },
            )

    def _to_serializable(self, config: VoiceAgentConfig) -> Dict[str, object]:
        profiles: Dict[str, Dict[str, object]] = {}
        for name, profile in config.profiles.items():
            profiles[name] = {
                "asr_backend": profile.asr.backend,
                "asr_model": str(profile.asr.model_path),
                "asr_device": profile.asr.device,
                "llm_backend": profile.llm.backend,
                "llm_model": str(profile.llm.model_path),
                "llm_context_window": profile.llm.context_window,
                "llm_temperature": profile.llm.temperature,
                "llm_top_p": profile.llm.top_p,
                "llm_repeat_penalty": profile.llm.repeat_penalty,
                "vad_model": str(profile.vad.model_path),
                "vad_trigger_level": profile.vad.trigger_level,
                "vad_release_level": profile.vad.release_level,
                "vad_sensitivity": profile.vad.sensitivity,
                "turn_min_utterance_ms": profile.turn.min_utterance_ms,
                "turn_max_silence_ms": profile.turn.max_silence_ms,
                "turn_max_turn_ms": profile.turn.max_turn_ms,
                "turn_barge_in": profile.turn.barge_in,
                "tts": {
                    "backend": profile.tts.backend,
                    "voice_id": profile.tts.voice_id,
                    "model_dir": str(profile.tts.model_dir),
                    "chunk_min_chars": profile.tts.chunk_min_chars,
                    "chunk_min_ms": profile.tts.chunk_min_ms,
                    "chunk_max_gap_ms": profile.tts.chunk_max_gap_ms,
                },
            }
        return {"active_profile": config.active_profile, "profiles": profiles}


def load_config(config_path: Path | None = None) -> VoiceAgentConfig:
    manager = ConfigManager(config_path=config_path)
    return manager.load()

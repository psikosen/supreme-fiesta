# Voice Agent Runtime

A cross-platform (Linux + macOS) voice agent runtime that orchestrates audio I/O, wake/VAD, ASR backends, GGUF LLM inference, and Kitten TTS playback. The CLI bundles developer tooling for configuration profiles, model downloads, benchmarking, and a dots-based visualizer for runtime insight.

## Features

- Python 3.11+ project managed with Poetry and structured `voice_agent` package layout.
- Structured logging that emits JSON plus the Sherlock Protocol prompts for human review.
- Config profiles stored in TOML with validation, asset path checking, and a registry of supported models.
- Audio loopback demo via PortAudio (`sounddevice`) with device selection flags.
- Terminal dots visualizer showing VAD states, ASR partials, and simulated LLM token flow.
- Kitten TTS integration with ONNXRuntime fallback, voice inventory tooling, and CLI preview playback.
- Model fetcher with resumable downloads, bootstrap script for Linux/macOS, and placeholder benchmarking commands.

## Getting Started

### Prerequisites

- Python 3.11 or 3.12
- `sounddevice` requirements (PortAudio installed on the system)
- Optional: `onnxruntime` (CPU) for Kitten TTS inference
- Optional: `faster-whisper`/`mlx-whisper` for ASR backends (`poetry install --extras asr`)

### Bootstrap

Use the provided script to set up dependencies and fetch baseline models (LFM2-350M + Kitten TTS assets):

```bash
./scripts/bootstrap.sh
```

The script installs Poetry if missing, configures the environment, and downloads the smallest GGUF along with Kitten TTS Nano 0.2.

### CLI Usage

```
poetry run voice-agent --help
```

Common commands:

- `poetry run voice-agent run --seconds 5` — record and playback a loopback sample with live dots visualizer.
- `poetry run voice-agent audio-devices` — list available input/output devices.
- `poetry run voice-agent profile list` — enumerate configuration profiles.
- `poetry run voice-agent profile use fast-mlx` — switch to the MLX Whisper profile.
- `poetry run voice-agent tts list` — list available Kitten voices (requires assets present).
- `poetry run voice-agent tts sample --voice kitten/en_female_01 "Hello there"` — play a sample clip.
- `poetry run voice-agent models pull tts/kitten-nano-0.2` — fetch Kitten TTS model assets.
- `poetry run voice-agent bench latency` — measure loopback latency.

### Configuration

Configuration lives at `~/.config/voice-agent/config.toml` (auto-seeded on first run). Profiles capture ASR backend, GGUF paths, TTS voices, and turn parameters. Update the file to point to your local assets.

### Tests & Linting

```
poetry run pytest
poetry run ruff check .
```

### Roadmap

See [task.md](task.md) for the tracked epics and outstanding work (Silero VAD, Whisper integrations, llama.cpp runner, Smart-Turn orchestration, etc.).

## Licensing

This project is distributed under the Apache-2.0 license. Third-party model licenses (LiquidAI LFM2 GGUF, Kitten TTS Nano 0.2) must be reviewed before redistribution.

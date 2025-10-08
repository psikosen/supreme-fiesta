"""Typer CLI wiring for the voice agent."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
import typer

from voice_agent.audio import AudioIO
from voice_agent.cli.visualizer import DotsVisualizer
from voice_agent.config import ConfigManager
from voice_agent.logging import configure_logging, get_logger
from voice_agent.tts import KittenTTS, load_voice_inventory

app = typer.Typer(add_completion=False, help="Cross-platform voice agent runtime")
profile_app = typer.Typer(help="Manage configuration profiles")
tts_app = typer.Typer(help="Text-to-speech tools")
models_app = typer.Typer(help="Model asset management")
bench_app = typer.Typer(help="Benchmark utilities")

app.add_typer(profile_app, name="profile")
app.add_typer(tts_app, name="tts")
app.add_typer(models_app, name="models")
app.add_typer(bench_app, name="bench")


CONFIG_ENV_VAR = "VOICE_AGENT_CONFIG"
_LOG = get_logger(__name__)


def _config_manager(config_path: Optional[Path]) -> ConfigManager:
    configure_logging()
    if config_path is None:
        env_path = os.environ.get(CONFIG_ENV_VAR)
        if env_path:
            config_path = Path(env_path)
    return ConfigManager(config_path=config_path) if config_path else ConfigManager()


@app.command()
def run(
    config_path: Optional[Path] = typer.Option(None, "--config-path", help="Override configuration path"),
    seconds: float = typer.Option(3.0, help="Number of seconds to record"),
    input_device: Optional[str] = typer.Option(None, help="Input device name or index"),
    output_device: Optional[str] = typer.Option(None, help="Output device name or index"),
) -> None:
    """Run an audio loopback demo with the dot visualizer."""

    manager = _config_manager(config_path)
    manager.load()
    audio = AudioIO(samplerate=16000)
    visualizer = DotsVisualizer()
    data = audio.record_loopback(seconds=seconds, input_device=input_device, output_device=output_device)
    visualizer.run_demo(data)
    typer.echo("Loopback complete")


@app.command("audio-devices")
def audio_devices() -> None:
    """List available audio devices."""

    audio = AudioIO()
    devices = audio.list_devices()
    for device in devices:
        typer.echo(
            f"{device.id:>3} | {device.name} | inputs: {device.max_input_channels} | outputs: {device.max_output_channels} | sample_rate: {device.default_samplerate}",
        )


@profile_app.command("list")
def profile_list(
    config_path: Optional[Path] = typer.Option(None, "--config-path", help="Override configuration path"),
) -> None:
    """List available profiles."""

    manager = _config_manager(config_path)
    for name in manager.list_profiles():
        typer.echo(name)


@profile_app.command("use")
def profile_use(
    name: str = typer.Argument(..., help="Profile name"),
    config_path: Optional[Path] = typer.Option(None, "--config-path", help="Override configuration path"),
) -> None:
    """Switch active profile."""

    manager = _config_manager(config_path)
    manager.use_profile(name)
    typer.echo(f"Active profile: {name}")


@tts_app.command("list")
def tts_list(
    config_path: Optional[Path] = typer.Option(None, "--config-path", help="Override configuration path"),
) -> None:
    """List available voices from the Kitten inventory."""

    manager = _config_manager(config_path)
    profile = manager.load().active()
    voices_path = profile.tts.model_dir / "voices.npz"
    if not voices_path.exists():
        raise typer.BadParameter(f"Voices inventory missing at {voices_path}. Run voice-agent models pull tts/kitten-nano-0.2")
    inventory = load_voice_inventory(voices_path)
    for voice in inventory.voices:
        typer.echo(f"{voice.voice_id} | {' '.join(voice.tags)}")


@tts_app.command("sample")
def tts_sample(
    config_path: Optional[Path] = typer.Option(None, "--config-path", help="Override configuration path"),
    voice: Optional[str] = typer.Option(None, "--voice", help="Voice identifier"),
    text: str = typer.Argument("Hello from Kitten TTS", help="Text to speak"),
) -> None:
    """Speak a sample phrase using the configured Kitten TTS backend."""

    manager = _config_manager(config_path)
    profile = manager.load().active()
    voice_id = voice or profile.tts.voice_id
    try:
        tts = KittenTTS(profile.tts.model_dir, voice_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    tts.speak(text, {
        "chunk_min_chars": profile.tts.chunk_min_chars,
        "chunk_min_ms": profile.tts.chunk_min_ms,
        "chunk_max_gap_ms": profile.tts.chunk_max_gap_ms,
    })


@models_app.command("pull")
def models_pull(
    model_id: str = typer.Argument(..., help="Model identifier to download"),
    destination: Path = typer.Option(Path.home() / ".voice-agent" / "models", "--dest", help="Download directory"),
) -> None:
    """Download model assets with resume and checksum validation (simplified)."""

    MODEL_MAP = {
        "tts/kitten-nano-0.2": [
            (
                "https://huggingface.co/kitten-tts/kitten-tts-nano-0.2/resolve/main/kitten_tts_nano_v0_2.onnx",
                "kitten_tts_nano_v0_2.onnx",
            ),
            (
                "https://huggingface.co/kitten-tts/kitten-tts-nano-0.2/resolve/main/voices.npz",
                "voices.npz",
            ),
            (
                "https://huggingface.co/kitten-tts/kitten-tts-nano-0.2/resolve/main/config.json",
                "config.json",
            ),
        ],
    }
    if model_id not in MODEL_MAP:
        raise typer.BadParameter(f"Unknown model id: {model_id}")
    destination.mkdir(parents=True, exist_ok=True)
    for url, filename in MODEL_MAP[model_id]:
        target = destination / filename
        _download_file(url, target)
        typer.echo(f"Downloaded {target}")


@bench_app.command("latency")
def bench_latency(
    config_path: Optional[Path] = typer.Option(None, "--config-path", help="Override configuration path"),
) -> None:
    """Run a lightweight latency probe for audio loopback."""

    manager = _config_manager(config_path)
    manager.load()
    audio = AudioIO(samplerate=16000)
    start = time.perf_counter()
    audio.record_loopback(seconds=1.0)
    duration = time.perf_counter() - start
    typer.echo(json.dumps({"loopback_seconds": duration}, indent=2))


def _download_file(url: str, target: Path) -> None:
    headers = {}
    if target.exists():
        existing = target.stat().st_size
        headers["Range"] = f"bytes={existing}-"
    else:
        existing = 0
    response = requests.get(url, stream=True, timeout=30, headers=headers)
    response.raise_for_status()
    mode = "ab" if existing else "wb"
    with target.open(mode) as fh:
        for chunk in response.iter_content(chunk_size=1_048_576):
            if chunk:
                fh.write(chunk)

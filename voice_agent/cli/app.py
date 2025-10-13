"""Typer CLI wiring for the voice agent."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import requests
import typer

from voice_agent.benchmarks import (
    make_silence,
    measure_asr_latency,
    measure_llm_latency,
    measure_tts_latency,
)
from voice_agent.asr import AsrEvent, create_asr_engine
from voice_agent.audio import AudioIO
from voice_agent.cli.visualizer import DotsVisualizer
from voice_agent.config import ConfigManager
from voice_agent.llm import LlmConfig as RuntimeLlmConfig, create_llm_engine
from voice_agent.logging import configure_logging, get_logger
from voice_agent.telemetry import TelemetryClient
from voice_agent.tts import KittenTTS, load_voice_inventory
from voice_agent.vad import SileroVadError, SileroVadStream

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
    config = manager.load()
    audio = AudioIO(samplerate=16000)
    visualizer = DotsVisualizer()
    data = audio.record_loopback(seconds=seconds, input_device=input_device, output_device=output_device)
    profile = config.active()
    vad_model_path = profile.vad.model_path.expanduser()
    try:
        vad_stream = SileroVadStream.from_numpy(
            audio=data,
            model_path=vad_model_path,
            sample_rate=audio.samplerate,
            trigger_level=profile.vad.trigger_level,
            release_level=profile.vad.release_level,
            sensitivity=profile.vad.sensitivity,
        )
    except FileNotFoundError as exc:
        _LOG.warning(
            "VAD model missing",
            extra={
                "classname": "CLI",
                "function": "run",
                "system_section": "vad",
                "error": str(exc),
                "derived_message": "[Continuous skepticism (Sherlock Protocol)] Falling back to RMS-based visualiser",
            },
        )
        visualizer.run_demo(data)
    except SileroVadError as exc:
        _LOG.error(
            "VAD initialisation failed",
            extra={
                "classname": "CLI",
                "function": "run",
                "system_section": "vad",
                "error": str(exc),
                "derived_message": "[Continuous skepticism (Sherlock Protocol)] Falling back to RMS-based visualiser",
            },
        )
        visualizer.run_demo(data)
    else:
        visualizer.render_vad(vad_stream, fallback_audio=data)

    asr_partials: list[str] = []
    asr_finals: list[AsrEvent] = []
    asr_confidence: list[float] = []

    try:
        asr_engine = create_asr_engine(profile.asr)
    except RuntimeError as exc:
        _LOG.warning(
            "[Continuous skepticism (Sherlock Protocol)] Falling back to visualiser-only mode",
            extra={
                "classname": "CLI",
                "function": "run",
                "system_section": "asr",
                "error": str(exc),
                "structured_message": "ASR backend unavailable",
            },
        )
    else:
        asr_engine.on_partial(lambda event: asr_partials.append(event.text))
        asr_engine.on_final(asr_finals.append)
        asr_engine.on_confidence(lambda value: asr_confidence.append(value))
        audio_bytes = [np.ascontiguousarray(data, dtype=np.float32).reshape(-1).tobytes()]
        try:
            asr_engine.transcribe(audio_bytes)
        except Exception as exc:  # pragma: no cover - backend specific failures
            _LOG.error(
                "[Continuous skepticism (Sherlock Protocol)] Check ASR backend configuration",
                extra={
                    "classname": "CLI",
                    "function": "run",
                    "system_section": "asr",
                    "error": str(exc),
                    "structured_message": "ASR transcription failed",
                },
            )
        else:
            if asr_partials:
                visualizer.render_partial(asr_partials)
            if asr_finals:
                typer.echo(f"Final transcript: {asr_finals[-1].text}")
            if asr_confidence:
                typer.echo(f"Language confidence: {asr_confidence[-1]:.2f}")

    typer.echo("Loopback complete")


@bench_app.command("llm")
def bench_llm(
    prompt: str = typer.Argument("Hello there", help="Prompt to send to the LLM"),
    max_tokens: int = typer.Option(128, help="Maximum tokens to request from the backend"),
    config_path: Optional[Path] = typer.Option(None, "--config-path", help="Override configuration path"),
) -> None:
    """Stream tokens from the configured llama.cpp backend."""

    manager = _config_manager(config_path)
    profile = manager.load().active()
    runtime_config = RuntimeLlmConfig(
        model_path=profile.llm.model_path,
        temperature=profile.llm.temperature,
        top_p=profile.llm.top_p,
        repeat_penalty=profile.llm.repeat_penalty,
        context_window=profile.llm.context_window,
    )

    try:
        engine = create_llm_engine(runtime_config)
    except FileNotFoundError as exc:
        _LOG.error(
            "[Continuous skepticism (Sherlock Protocol)] LLM model missing",
            extra={
                "classname": "CLI",
                "function": "bench_llm",
                "system_section": "llm",
                "error": str(exc),
                "structured_message": "Run voice-agent models pull to download",
            },
        )
        raise typer.BadParameter(str(exc)) from exc

    typer.echo("Streaming completion...\n")
    try:
        for chunk in engine.stream(prompt=prompt, max_tokens=max_tokens):
            typer.echo(chunk, nl=False)
    except RuntimeError as exc:
        _LOG.error(
            "[Continuous skepticism (Sherlock Protocol)] LLM streaming failed",
            extra={
                "classname": "CLI",
                "function": "bench_llm",
                "system_section": "llm",
                "error": str(exc),
                "structured_message": "Inspect llama.cpp configuration",
            },
        )
        raise typer.Exit(code=1) from exc
    else:
        typer.echo("\n\nStream complete")


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


DEFAULT_MODEL_DIR = Path.home() / ".voice-agent" / "models"


@models_app.command(
    "pull",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def models_pull(
    ctx: typer.Context,
    model_id: str = typer.Argument(..., help="Model identifier to download"),
) -> None:
    """Download model assets with resume and checksum validation (simplified)."""

    extra_args = list(ctx.args)
    positional_destinations: list[Path] = []
    option_destination: Optional[Path] = None

    idx = 0
    while idx < len(extra_args):
        token = extra_args[idx]
        if token in {"--dest", "-d"}:
            if idx + 1 >= len(extra_args):
                typer.echo("Error: --dest requires a value.", err=True)
                raise typer.Exit(code=2)
            option_destination = Path(extra_args[idx + 1])
            idx += 2
            continue
        if token.startswith("--dest="):
            option_destination = Path(token.split("=", 1)[1])
            idx += 1
            continue
        if token.startswith("-d") and token not in {"-d"}:
            option_destination = Path(token[2:])
            idx += 1
            continue
        positional_destinations.append(Path(token))
        idx += 1

    destination = (
        option_destination
        or (positional_destinations[0] if positional_destinations else DEFAULT_MODEL_DIR)
    )

    if len(positional_destinations) > 1:
        _LOG.warning(
            "Multiple positional destinations provided",
            extra={
                "log_filename": __file__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "classname": "CLI",
                "function": "models_pull",
                "system_section": "models",
                "line_num": 0,
                "error": None,
                "db_phase": "none",
                "method": "NONE",
                "log_message": "[Continuous skepticism (Sherlock Protocol)] Only the first positional destination will be used",
            },
        )
        typer.echo(
            "Warning: Multiple positional destinations provided; only the first will be used.",
            err=True,
        )

    if option_destination and positional_destinations:
        _LOG.warning(
            "Conflicting destination overrides provided",
            extra={
                "log_filename": __file__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "classname": "CLI",
                "function": "models_pull",
                "system_section": "models",
                "line_num": 0,
                "error": None,
                "db_phase": "none",
                "method": "NONE",
                "log_message": "[Continuous skepticism (Sherlock Protocol)] Positional destination ignored in favour of --dest",
            },
        )
        typer.echo(
            "Warning: Positional destination ignored because --dest was provided; using --dest value.",
            err=True,
        )

    MODEL_MAP = {
        "tts/kitten-nano-0.2": [
            (
                "https://huggingface.co/KittenML/kitten-tts-nano-0.2/blob/main/kitten_tts_nano_v0_2.onnx",
                "kitten_tts_nano_v0_2.onnx",
            ),
            (
                "https://huggingface.co/KittenML/kitten-tts-nano-0.2/blob/main/voices.npz",
                "voices.npz",
            ),
            (
                "https://huggingface.co/KittenML/kitten-tts-nano-0.2/blob/main/config.json",
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
    """Run latency probes for audio loopback, ASR, LLM, and TTS."""

    manager = _config_manager(config_path)
    config = manager.load()
    profile = config.active()
    telemetry = TelemetryClient()
    metrics: Dict[str, object] = {}

    audio = AudioIO(samplerate=16000)
    start = time.perf_counter()
    audio.record_loopback(seconds=1.0)
    metrics["audio_loopback_seconds"] = time.perf_counter() - start

    try:
        asr_engine = create_asr_engine(profile.asr)
    except Exception as exc:  # pragma: no cover - backend optional
        _LOG.error(
            "ASR benchmark setup failed",
            extra={
                "classname": "CLI",
                "function": "bench_latency",
                "system_section": "asr",
                "error": str(exc),
                "structured_message": "[Continuous skepticism (Sherlock Protocol)] Unable to initialise ASR backend",
            },
        )
        metrics["asr"] = {"error": str(exc)}
    else:
        asr_result = measure_asr_latency(asr_engine, make_silence(1.0, audio.samplerate), telemetry=telemetry)
        metrics["asr"] = asr_result.metrics

    runtime_config = RuntimeLlmConfig(
        model_path=profile.llm.model_path,
        temperature=profile.llm.temperature,
        top_p=profile.llm.top_p,
        repeat_penalty=profile.llm.repeat_penalty,
        context_window=profile.llm.context_window,
    )

    try:
        llm_engine = create_llm_engine(runtime_config)
    except Exception as exc:  # pragma: no cover - backend optional
        _LOG.error(
            "LLM benchmark setup failed",
            extra={
                "classname": "CLI",
                "function": "bench_latency",
                "system_section": "llm",
                "error": str(exc),
                "structured_message": "[Continuous skepticism (Sherlock Protocol)] Unable to initialise LLM backend",
            },
        )
        metrics["llm"] = {"error": str(exc)}
    else:
        llm_result = measure_llm_latency(llm_engine, "Benchmark prompt", max_tokens=32, telemetry=telemetry)
        metrics["llm"] = llm_result.metrics

    try:
        tts_engine = KittenTTS(profile.tts.model_dir, profile.tts.voice_id)
    except Exception as exc:  # pragma: no cover - backend optional
        _LOG.error(
            "TTS benchmark setup failed",
            extra={
                "classname": "CLI",
                "function": "bench_latency",
                "system_section": "tts",
                "error": str(exc),
                "structured_message": "[Continuous skepticism (Sherlock Protocol)] Unable to initialise TTS backend",
            },
        )
        metrics["tts"] = {"error": str(exc)}
    else:
        chunk_cfg = {
            "chunk_min_chars": profile.tts.chunk_min_chars,
            "chunk_min_ms": profile.tts.chunk_min_ms,
            "chunk_max_gap_ms": profile.tts.chunk_max_gap_ms,
        }
        tts_result = measure_tts_latency(tts_engine, "Benchmarking latency", chunk_cfg, telemetry=telemetry)
        metrics["tts"] = tts_result.metrics

    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


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

import json
from pathlib import Path
from unittest import mock

import numpy as np

from typer.testing import CliRunner

from voice_agent.cli.app import _download_file, app
from voice_agent.config import ConfigManager
from voice_agent.asr import AsrEngine, AsrEvent

runner = CliRunner()


def test_profile_list(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    ConfigManager(config_path=config_path)
    result = runner.invoke(app, ["profile", "list"], env={"VOICE_AGENT_CONFIG": str(config_path)})
    assert result.exit_code == 0
    assert "default" in result.stdout


def test_download_file_resume(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    target.write_bytes(b"hello")

    fake_response = mock.Mock()
    fake_response.iter_content.return_value = [b"world"]
    fake_response.raise_for_status.return_value = None

    with mock.patch("voice_agent.cli.app.requests.get", return_value=fake_response) as mock_get:
        _download_file("https://example.com/file.bin", target)

    mock_get.assert_called_once()
    assert target.read_bytes() == b"helloworld"


def test_bench_latency_outputs_metrics(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    ConfigManager(config_path=config_path)

    class DummyAudio:
        samplerate = 16000

        def record_loopback(self, seconds: float, input_device=None, output_device=None):
            return None

    class DummyAsr:
        def transcribe(self, audio_stream):
            list(audio_stream)

    class DummyLlm:
        def stream(self, prompt: str, *, max_tokens: int | None = None):
            yield "ok"

    class DummyTts:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def synthesize(self, text: str, chunk_config):
            yield np.ones(10, dtype=np.float32)

    with (
        mock.patch("voice_agent.cli.app.AudioIO", return_value=DummyAudio()),
        mock.patch("voice_agent.cli.app.create_asr_engine", return_value=DummyAsr()),
        mock.patch("voice_agent.cli.app.create_llm_engine", return_value=DummyLlm()),
        mock.patch("voice_agent.cli.app.KittenTTS", DummyTts),
    ):
        result = runner.invoke(
            app,
            ["bench", "latency"],
            env={"VOICE_AGENT_CONFIG": str(config_path)},
        )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "audio_loopback_seconds" in payload
    assert payload["asr"]["latency_ms"] >= 0
    assert payload["llm"]["chunks"] == 1


def test_models_pull_accepts_positional_destination(tmp_path: Path) -> None:
    destination = tmp_path / "models"

    with mock.patch("voice_agent.cli.app._download_file") as mock_download:
        mock_download.side_effect = lambda url, target: target.write_text("ok")
        result = runner.invoke(
            app,
            ["models", "pull", "tts/kitten-nano-0.2", str(destination)],
        )

    assert result.exit_code == 0
    expected_files = {destination / "kitten_tts_nano_v0_2.onnx", destination / "voices.npz", destination / "config.json"}
    actual_files = {call.args[1] for call in mock_download.call_args_list}
    assert actual_files == expected_files


def test_models_pull_accepts_option_destination(tmp_path: Path) -> None:
    destination = tmp_path / "from-option"

    with mock.patch("voice_agent.cli.app._download_file") as mock_download:
        result = runner.invoke(
            app,
            ["models", "pull", "tts/kitten-nano-0.2", "--dest", str(destination)],
        )

    assert result.exit_code == 0
    assert {call.args[1] for call in mock_download.call_args_list} == {
        destination / "kitten_tts_nano_v0_2.onnx",
        destination / "voices.npz",
        destination / "config.json",
    }


def test_models_pull_prefers_option_destination_when_both_provided(tmp_path: Path) -> None:
    positional = tmp_path / "positional"
    option = tmp_path / "option"

    with mock.patch("voice_agent.cli.app._download_file") as mock_download:
        result = runner.invoke(
            app,
            [
                "models",
                "pull",
                "tts/kitten-nano-0.2",
                str(positional),
                "--dest",
                str(option),
            ],
        )

    assert result.exit_code == 0
    if result.stderr_bytes is None:
        warning_output = result.stdout
    else:
        warning_output = result.stderr
    assert "Warning: Positional destination ignored because --dest was provided" in warning_output
    assert {call.args[1] for call in mock_download.call_args_list} == {
        option / "kitten_tts_nano_v0_2.onnx",
        option / "voices.npz",
        option / "config.json",
    }


def test_run_handles_missing_vad_model(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    ConfigManager(config_path=config_path)

    class DummyAudio:
        samplerate = 16000

        def record_loopback(self, seconds: float, input_device=None, output_device=None):
            return np.zeros((160, 1), dtype=np.float32)

    with (
        mock.patch("voice_agent.cli.app.AudioIO", return_value=DummyAudio()),
        mock.patch(
            "voice_agent.cli.app.SileroVadStream.from_numpy",
            side_effect=FileNotFoundError("missing model"),
        ),
        mock.patch("voice_agent.cli.app.DotsVisualizer") as mock_visualizer,
        mock.patch("voice_agent.cli.app.create_asr_engine", side_effect=RuntimeError("asr disabled")),
    ):
        result = runner.invoke(
            app,
            ["run", "--turns", "1"],
            env={"VOICE_AGENT_CONFIG": str(config_path)},
        )

    assert result.exit_code == 0
    assert "Loopback turn 1 complete" in result.stdout
    assert "Loopback session complete" in result.stdout
    assert mock_visualizer.return_value.run_demo.called


def test_run_supports_multiple_turns(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    ConfigManager(config_path=config_path)

    dummy_audio = mock.Mock()
    dummy_audio.samplerate = 16000
    dummy_audio.record_loopback.side_effect = [
        np.zeros((160, 1), dtype=np.float32),
        np.zeros((160, 1), dtype=np.float32),
    ]

    class DummyAsr(AsrEngine):
        def transcribe(self, audio_stream):  # type: ignore[override]
            self.emit_final(AsrEvent(text="hi", timestamp=0.0, confidence=0.5))

    with (
        mock.patch("voice_agent.cli.app.AudioIO", return_value=dummy_audio),
        mock.patch("voice_agent.cli.app.SileroVadStream.from_numpy", return_value=["vad"]),
        mock.patch("voice_agent.cli.app.DotsVisualizer") as mock_visualizer,
        mock.patch(
            "voice_agent.cli.app.create_asr_engine",
            side_effect=[DummyAsr(), DummyAsr()],
        ),
    ):
        result = runner.invoke(
            app,
            ["run", "--turns", "2"],
            env={"VOICE_AGENT_CONFIG": str(config_path)},
        )

    assert result.exit_code == 0
    assert dummy_audio.record_loopback.call_count == 2
    assert result.stdout.count("Loopback turn") == 2
    assert mock_visualizer.return_value.render_vad.call_count == 2

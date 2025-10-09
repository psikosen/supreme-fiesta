import json
from pathlib import Path
from unittest import mock

import numpy as np

from typer.testing import CliRunner

from voice_agent.cli.app import _download_file, app
from voice_agent.config import ConfigManager

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

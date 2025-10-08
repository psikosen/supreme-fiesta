from pathlib import Path
from unittest import mock

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

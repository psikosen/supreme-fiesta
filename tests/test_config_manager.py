from pathlib import Path

from voice_agent.config import ConfigManager


def test_config_manager_creates_default(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    manager = ConfigManager(config_path=config_path)
    assert config_path.exists()

    config = manager.load()
    assert config.active_profile == "default"
    assert "fast-mlx" in config.profiles

    manager.use_profile("fast-mlx")
    reloaded = manager.load()
    assert reloaded.active_profile == "fast-mlx"

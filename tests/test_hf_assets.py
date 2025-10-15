from pathlib import Path

from voice_agent.llm import hf_assets


class FakeHubError(Exception):
    pass


def test_infer_repo_id_with_category_segment():
    path = Path("assets/llm/Org/Repo/model.gguf")
    assert hf_assets._infer_repo_id(path) == "Org/Repo"


def test_infer_repo_id_without_category():
    path = Path("assets/Org/Repo/model.gguf")
    assert hf_assets._infer_repo_id(path) == "Org/Repo"


def test_ensure_local_gguf_missing_dependency(tmp_path, monkeypatch):
    model_path = tmp_path / "assets" / "llm" / "Org" / "Repo" / "model.gguf"

    def raise_module(name: str):  # pragma: no cover - simple stub
        raise ModuleNotFoundError(name)

    hf_assets._resolve_hf_client.cache_clear()
    hf_assets._MISSING_DEPENDENCY_LOGGED = False
    monkeypatch.setattr(hf_assets, "import_module", raise_module)

    result = hf_assets.ensure_local_gguf(model_path)

    assert result == model_path
    assert not model_path.exists()


def test_ensure_local_gguf_downloads_when_dependency_available(tmp_path, monkeypatch):
    model_path = tmp_path / "assets" / "llm" / "Org" / "Repo" / "model.gguf"
    download_calls = {}

    def fake_download(*, repo_id: str, filename: str, local_dir: str, local_dir_use_symlinks: bool, resume_download: bool):
        download_calls["args"] = (repo_id, filename, local_dir, local_dir_use_symlinks, resume_download)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF")
        return str(path)

    hf_assets._MISSING_DEPENDENCY_LOGGED = False
    hf_assets._resolve_hf_client.cache_clear()
    monkeypatch.setattr(hf_assets, "_resolve_hf_client", lambda: (fake_download, FakeHubError))

    result = hf_assets.ensure_local_gguf(model_path)

    assert result == model_path
    assert model_path.exists()
    assert download_calls["args"] == (
        "Org/Repo",
        "model.gguf",
        str(model_path.parent),
        False,
        True,
    )

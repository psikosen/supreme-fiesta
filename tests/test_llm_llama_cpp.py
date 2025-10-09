import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from voice_agent.llm import LlmConfig, LlamaCppEngine, create_llm_engine


def _write_dummy_model(path: Path) -> None:
    path.write_bytes(b"gguf")


def test_create_engine_loads_model(tmp_path: Path, mocker) -> None:
    model_path = tmp_path / "model.gguf"
    _write_dummy_model(model_path)
    mock_llama = mocker.Mock()
    mock_llama.return_value.create_completion.return_value = iter(())

    config = LlmConfig(
        model_path=model_path,
        temperature=0.5,
        top_p=0.8,
        repeat_penalty=1.05,
        context_window=1024,
    )

    with mock.patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=mock_llama)}):
        engine = create_llm_engine(config)

    assert isinstance(engine, LlamaCppEngine)
    mock_llama.assert_called_once_with(model_path=str(model_path), n_ctx=1024)


def test_stream_yields_chunks(tmp_path: Path, mocker) -> None:
    model_path = tmp_path / "model.gguf"
    _write_dummy_model(model_path)

    chunks = iter(
        [
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": " world"}]},
            {"choices": [{"text": "!"}]},
        ]
    )

    mock_llama = mocker.Mock()
    mock_llama.return_value.create_completion.return_value = chunks

    config = LlmConfig(model_path=model_path)
    with mock.patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=mock_llama)}):
        engine = create_llm_engine(config)

    result = list(engine.stream(prompt="Hi", max_tokens=5))

    assert result == ["Hello", " world", "!"]
    mock_llama.return_value.create_completion.assert_called_once_with(
        prompt="Hi",
        stream=True,
        temperature=config.temperature,
        top_p=config.top_p,
        repeat_penalty=config.repeat_penalty,
        max_tokens=5,
    )


def test_missing_model_raises(tmp_path: Path) -> None:
    config = LlmConfig(model_path=tmp_path / "missing.gguf")

    with pytest.raises(FileNotFoundError):
        create_llm_engine(config)

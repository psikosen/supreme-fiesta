"""Utilities for handling Hugging Face GGUF assets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from voice_agent.logging import get_logger

_LOG = get_logger(__name__)

_DERIVED_MESSAGE: Final[str] = "[Continuous skepticism (Sherlock Protocol)] Attempted Hugging Face GGUF repair"


def ensure_local_gguf(model_path: Path) -> Path:
    """Ensure a llama.cpp GGUF file exists locally, downloading from Hugging Face if needed."""

    if model_path.exists():
        return model_path

    repo_id = _infer_repo_id(model_path)
    if repo_id is None:
        _log_skip(model_path)
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=model_path.name,
            local_dir=str(model_path.parent),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except HfHubHTTPError as exc:  # pragma: no cover - network failures depend on runtime
        _LOG.warning(
            "Hugging Face GGUF download failed",
            extra={
                "filename": __file__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "classname": __name__,
                "function": "ensure_local_gguf",
                "system_section": "llm",
                "line_num": 0,
                "error": str(exc),
                "db_phase": "none",
                "method": "NONE",
                "message": f"Quality review: Unable to download {model_path.name} from {repo_id}",
                "derived_message": _DERIVED_MESSAGE,
            },
        )
        return model_path

    _LOG.info(
        "Downloaded GGUF model from Hugging Face",
        extra={
            "filename": __file__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classname": __name__,
            "function": "ensure_local_gguf",
            "system_section": "llm",
            "line_num": 0,
            "error": None,
            "db_phase": "none",
            "method": "NONE",
            "message": f"Quality review: Downloaded {model_path.name} from {repo_id}",
            "derived_message": _DERIVED_MESSAGE,
        },
    )
    return model_path


def _infer_repo_id(model_path: Path) -> str | None:
    parts = model_path.parts
    try:
        assets_index = parts.index("assets")
    except ValueError:
        return None

    repo_parts = parts[assets_index + 2 : -1]
    if len(repo_parts) != 1:
        return None

    namespace = parts[assets_index + 1]
    repo = repo_parts[0]
    if not namespace or not repo:
        return None

    return f"{namespace}/{repo}"


def _log_skip(model_path: Path) -> None:
    _LOG.debug(
        "Skipping Hugging Face download for GGUF path",
        extra={
            "filename": __file__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classname": __name__,
            "function": "_log_skip",
            "system_section": "llm",
            "line_num": 0,
            "error": None,
            "db_phase": "none",
            "method": "NONE",
            "message": f"Quality review: Path {model_path} does not map to Hugging Face repo",
            "derived_message": _DERIVED_MESSAGE,
        },
    )

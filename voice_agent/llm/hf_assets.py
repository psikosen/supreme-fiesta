"""Utilities for handling Hugging Face GGUF assets."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Callable, Final, Optional, Tuple, Type

from voice_agent.logging import get_logger

_LOG = get_logger(__name__)

_DERIVED_MESSAGE: Final[str] = (
    "[Continuous skepticism (Sherlock Protocol)]\n"
    "Quality review checklist\n"
    "* Could this change affect unexpected files/systems?\n"
    "* Are all dependencies accounted for without hidden couplings?\n"
    "* Which edge cases or failure modes still need coverage?\n"
    "* If blocked, what outcome are we working backward from?"
)

_HfDownload = Callable[..., str]
_HfHttpError = Type[Exception]

_MISSING_DEPENDENCY_LOGGED: bool = False


def ensure_local_gguf(model_path: Path) -> Path:
    """Ensure a llama.cpp GGUF file exists locally, downloading from Hugging Face if needed."""

    if model_path.exists():
        return model_path

    repo_id = _infer_repo_id(model_path)
    if repo_id is None:
        _log_skip(model_path)
        return model_path

    client = _resolve_hf_client()
    if client is None:
        _log_dependency_skip(model_path)
        return model_path

    hf_hub_download, http_error = client

    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=model_path.name,
            local_dir=str(model_path.parent),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except http_error as exc:  # pragma: no cover - network failures depend on runtime
        _LOG.warning(
            "Hugging Face GGUF download failed",
            extra={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "classname": __name__,
                "function": "ensure_local_gguf",
                "system_section": "llm",
                "line_num": 0,
                "error": str(exc),
                "db_phase": "none",
                "method": "NONE",
                "structured_message": f"Quality review: Unable to download {model_path.name} from {repo_id}",
                "derived_message": _DERIVED_MESSAGE,
            },
        )
        return model_path

    _LOG.info(
        "Downloaded GGUF model from Hugging Face",
        extra={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classname": __name__,
            "function": "ensure_local_gguf",
            "system_section": "llm",
            "line_num": 0,
            "error": None,
            "db_phase": "none",
            "method": "NONE",
            "structured_message": f"Quality review: Downloaded {model_path.name} from {repo_id}",
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

    repo_parts = parts[assets_index + 1 : -1]
    if len(repo_parts) < 2:
        return None

    namespace, repo = repo_parts[-2], repo_parts[-1]
    if not namespace or not repo:
        return None

    return f"{namespace}/{repo}"


def _log_skip(model_path: Path) -> None:
    _LOG.debug(
        "Skipping Hugging Face download for GGUF path",
        extra={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classname": __name__,
            "function": "_log_skip",
            "system_section": "llm",
            "line_num": 0,
            "error": None,
            "db_phase": "none",
            "method": "NONE",
            "structured_message": f"Quality review: Path {model_path} does not map to Hugging Face repo",
            "derived_message": _DERIVED_MESSAGE,
        },
    )


@lru_cache(maxsize=1)
def _resolve_hf_client() -> Optional[Tuple[_HfDownload, _HfHttpError]]:
    try:
        hub_module = import_module("huggingface_hub")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        _log_dependency_missing(str(exc))
        return None

    download = getattr(hub_module, "hf_hub_download", None)
    if not callable(download):  # pragma: no cover - defensive
        _log_dependency_missing("hf_hub_download attribute missing or not callable")
        return None

    try:
        utils_module = import_module("huggingface_hub.utils")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        _log_dependency_missing(str(exc))
        return None

    http_error = getattr(utils_module, "HfHubHTTPError", None)
    if not isinstance(http_error, type) or not issubclass(http_error, Exception):  # pragma: no cover - defensive
        _log_dependency_missing("HfHubHTTPError missing from huggingface_hub.utils")
        return None

    return download, http_error


def _log_dependency_missing(reason: str) -> None:
    global _MISSING_DEPENDENCY_LOGGED
    if _MISSING_DEPENDENCY_LOGGED:
        return
    _MISSING_DEPENDENCY_LOGGED = True
    _LOG.warning(
        "huggingface_hub dependency unavailable",
        extra={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classname": __name__,
            "function": "_resolve_hf_client",
            "system_section": "llm",
            "line_num": 0,
            "error": reason,
            "db_phase": "none",
            "method": "NONE",
            "structured_message": "Quality review: huggingface_hub not installed; skipping automatic GGUF downloads",
            "derived_message": _DERIVED_MESSAGE,
        },
    )


def _log_dependency_skip(model_path: Path) -> None:
    _LOG.warning(
        "Skipping Hugging Face download; dependency unavailable",
        extra={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classname": __name__,
            "function": "ensure_local_gguf",
            "system_section": "llm",
            "line_num": 0,
            "error": None,
            "db_phase": "none",
            "method": "NONE",
            "structured_message": f"Quality review: huggingface_hub missing; cannot download {model_path.name}",
            "derived_message": _DERIVED_MESSAGE,
        },
    )

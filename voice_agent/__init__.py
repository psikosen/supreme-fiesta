"""Voice agent runtime package."""

from importlib.metadata import version

__all__ = ["__version__"]


def __getattr__(name: str) -> str:
    if name == "__version__":
        try:
            return version("voice-agent")
        except Exception:  # pragma: no cover - fallback when metadata missing
            return "0.0.0"
    raise AttributeError(name)

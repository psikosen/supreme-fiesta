"""Utilities shared across ASR backends."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def stream_to_numpy(audio_stream: Iterable[bytes], dtype: np.dtype = np.float32) -> np.ndarray:
    """Combine a stream of PCM byte chunks into a single NumPy array.

    The voice agent currently exchanges audio between components as little-endian float32
    PCM frames.  Both Faster-Whisper and MLX Whisper expect NumPy arrays in that format, so
    this helper consolidates a byte-stream into a contiguous array that downstream engines
    can consume.
    """

    if isinstance(audio_stream, (bytes, bytearray)):
        return np.frombuffer(audio_stream, dtype=dtype)
    chunks = list(audio_stream)
    if not chunks:
        return np.array([], dtype=dtype)
    return np.frombuffer(b"".join(chunks), dtype=dtype)


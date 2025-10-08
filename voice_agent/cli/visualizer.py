"""Terminal-based dot visualizer for runtime state."""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class VisualizerConfig:
    rms_threshold: float = 0.02
    chunk_size: int = 1024
    refresh_rate: float = 0.05


class DotsVisualizer:
    """Render ASCII indicators for VAD, ASR partials, and LLM tokens."""

    def __init__(self, config: VisualizerConfig | None = None) -> None:
        self.config = config or VisualizerConfig()

    def render_audio(self, audio: np.ndarray) -> None:
        sys.stdout.write("VAD State:\n")
        sys.stdout.flush()
        chunk_size = self.config.chunk_size
        total_chunks = max(1, audio.size // chunk_size)
        for idx in range(total_chunks):
            chunk = audio[idx * chunk_size : (idx + 1) * chunk_size]
            if not chunk.size:
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            bar = "█" * min(10, int(rms / self.config.rms_threshold))
            if rms > self.config.rms_threshold:
                state = "speaking"
            else:
                state = "silence"
            sys.stdout.write(f"[{state:>8}] {bar}\n")
            sys.stdout.flush()
            time.sleep(self.config.refresh_rate)

    def render_tokens(self, token_stream: Iterable[str], duration: float = 0.5) -> None:
        sys.stdout.write("LLM Tokens:\n")
        sys.stdout.flush()
        for token in token_stream:
            sys.stdout.write("•")
            sys.stdout.flush()
            time.sleep(duration)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def render_partial(self, text_stream: Iterable[str]) -> None:
        sys.stdout.write("ASR Partial:\n")
        sys.stdout.flush()
        for text in text_stream:
            sys.stdout.write(f"\r{text:60s}")
            sys.stdout.flush()
            time.sleep(self.config.refresh_rate)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def run_demo(self, audio: np.ndarray) -> None:
        self.render_audio(audio)
        fake_tokens = itertools.islice(itertools.cycle(["."]), 12)
        self.render_tokens(fake_tokens, duration=0.1)
        self.render_partial(["transcribing..."])

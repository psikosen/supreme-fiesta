"""Telemetry helpers for emitting structured latency events."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List

from voice_agent.logging import get_logger


@dataclass(slots=True)
class TelemetryEvent:
    """Represents a telemetry datapoint recorded by the runtime."""

    name: str
    timestamp: float
    fields: Dict[str, Any]


class TelemetryClient:
    """Collect and emit telemetry events to the structured logger."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self._clock = clock or time.perf_counter
        self._time_source = time_source or time.time
        self._logger = get_logger(f"{__name__}.TelemetryClient")
        self._events: List[TelemetryEvent] = []

    def emit(self, name: str, **fields: Any) -> TelemetryEvent:
        """Emit a telemetry event and record it for later inspection."""

        event = TelemetryEvent(name=name, timestamp=self._time_source(), fields=dict(fields))
        self._events.append(event)
        payload = {"name": name, **fields}
        self._logger.info(
            "Telemetry event",
            extra={
                "classname": self.__class__.__name__,
                "function": "emit",
                "system_section": "telemetry",
                "structured_message": json.dumps(payload, ensure_ascii=False),
            },
        )
        return event

    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[None]:
        """Measure the duration of a block and emit it as telemetry."""

        start = self._clock()
        try:
            yield
        finally:
            duration_ms = (self._clock() - start) * 1000.0
            self.emit(name, duration_ms=duration_ms, **fields)

    @property
    def events(self) -> List[TelemetryEvent]:
        """Return a copy of emitted telemetry events."""

        return list(self._events)


__all__ = ["TelemetryClient", "TelemetryEvent"]


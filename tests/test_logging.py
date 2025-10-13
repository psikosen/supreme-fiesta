import json
import logging

from voice_agent.logging import SHERLOCK_PROMPT, StructuredFormatter


def test_structured_formatter_emits_derived_message() -> None:
    record = logging.LogRecord(
        name="voice_agent.tests",
        level=logging.INFO,
        pathname="/tmp/test.py",
        lineno=42,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.classname = "TestCase"
    record.function = "test_structured_formatter_emits_derived_message"
    record.system_section = "logging"
    record.derived_message = "[Continuous skepticism (Sherlock Protocol)] Test derived line"

    formatter = StructuredFormatter()
    formatted = formatter.format(record)
    lines = formatted.splitlines()

    assert len(lines) == 7
    payload = json.loads(lines[0])
    assert payload["message"] == "hello"
    assert payload["derived_message"] == record.derived_message
    assert lines[1] == record.derived_message
    prompt_lines = SHERLOCK_PROMPT.splitlines()
    assert lines[2:] == prompt_lines

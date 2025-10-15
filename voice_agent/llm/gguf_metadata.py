"""Utilities for reading GGUF metadata required by the llama.cpp runner."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

_GGUF_MAGIC = b"GGUF"

# GGUF metadata value type identifiers. The numeric values match the enum in
# `gguf/constants.py` from the upstream llama.cpp project. Only the cases we
# need to skip or parse are implemented here to keep the dependency surface
# small.
_TYPE_UINT8 = 0
_TYPE_INT8 = 1
_TYPE_UINT16 = 2
_TYPE_INT16 = 3
_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10
_TYPE_INT64 = 11
_TYPE_FLOAT64 = 12

_SCALAR_BYTE_SIZES: dict[int, int] = {
    _TYPE_UINT8: 1,
    _TYPE_INT8: 1,
    _TYPE_UINT16: 2,
    _TYPE_INT16: 2,
    _TYPE_UINT32: 4,
    _TYPE_INT32: 4,
    _TYPE_FLOAT32: 4,
    _TYPE_BOOL: 1,
    _TYPE_UINT64: 8,
    _TYPE_INT64: 8,
    _TYPE_FLOAT64: 8,
}


def read_general_architecture(path: Path) -> str | None:
    """Return the GGUF ``general.architecture`` value if present.

    The parser focuses on the minimal subset of GGUF needed to reach metadata
    entries without introducing an additional runtime dependency on the full
    gguf Python package.
    """

    try:
        with path.open("rb") as handle:
            return _read_general_architecture_from_stream(handle)
    except OSError:
        return None


def _read_general_architecture_from_stream(handle: BinaryIO) -> str | None:
    if handle.read(len(_GGUF_MAGIC)) != _GGUF_MAGIC:
        return None

    # GGUF header layout:
    # uint32 version, uint64 tensor_count, uint64 kv_count
    version_bytes = handle.read(4)
    if len(version_bytes) != 4:
        return None

    tensor_count_bytes = handle.read(8)
    kv_count_bytes = handle.read(8)
    if len(tensor_count_bytes) != 8 or len(kv_count_bytes) != 8:
        return None

    kv_count = int.from_bytes(kv_count_bytes, "little", signed=False)

    for _ in range(kv_count):
        key_length_bytes = handle.read(8)
        if len(key_length_bytes) != 8:
            return None
        key_length = int.from_bytes(key_length_bytes, "little", signed=False)

        key_bytes = handle.read(key_length)
        if len(key_bytes) != key_length:
            return None
        key = key_bytes.decode("utf-8", errors="ignore")

        value_type_bytes = handle.read(4)
        if len(value_type_bytes) != 4:
            return None
        value_type = int.from_bytes(value_type_bytes, "little", signed=False)

        capture_value = key == "general.architecture"
        try:
            value = _read_value(handle, value_type, capture_string=capture_value)
        except ValueError:
            value = None

        if capture_value:
            return value

    return None


def _read_value(handle: BinaryIO, value_type: int, *, capture_string: bool) -> str | None:
    if value_type == _TYPE_STRING:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            return None
        length = int.from_bytes(length_bytes, "little", signed=False)
        value_bytes = handle.read(length)
        if len(value_bytes) != length:
            return None
        if capture_string:
            return value_bytes.decode("utf-8", errors="ignore")
        return None

    if value_type == _TYPE_ARRAY:
        element_type_bytes = handle.read(4)
        length_bytes = handle.read(8)
        if len(element_type_bytes) != 4 or len(length_bytes) != 8:
            return None
        element_type = int.from_bytes(element_type_bytes, "little", signed=False)
        element_count = int.from_bytes(length_bytes, "little", signed=False)
        for _ in range(element_count):
            _read_value(handle, element_type, capture_string=False)
        return None

    size = _SCALAR_BYTE_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"Unsupported GGUF metadata type: {value_type}")

    skipped = handle.read(size)
    if len(skipped) != size:
        return None
    return None

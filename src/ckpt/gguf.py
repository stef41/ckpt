"""GGUF binary format parser for checkpoint inspection."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ckpt._types import CheckpointFormat, CheckpointInfo, DType, FormatError, TensorInfo

# GGUF value type constants
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

# GGUF tensor dtype mapping (ggml_type enum)
_GGUF_TENSOR_DTYPE: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
}

# Map GGUF tensor dtypes to ckpt DType where applicable
_GGUF_TO_CKPT_DTYPE: dict[str, DType] = {
    "F32": DType.F32,
    "F16": DType.F16,
    "BF16": DType.BF16,
    "F64": DType.F64,
    "I8": DType.I8,
    "I16": DType.I16,
    "I32": DType.I32,
    "I64": DType.I64,
}

GGUF_MAGIC = b"GGUF"


@dataclass
class GGUFTensorEntry:
    """Parsed tensor info from a GGUF file."""

    name: str
    shape: list[int]
    dtype: str
    offset: int


@dataclass
class GGUFInfo:
    """Parsed info from a GGUF file header."""

    version: int
    tensor_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: list[GGUFTensorEntry] = field(default_factory=list)


class _GGUFReader:
    """Low-level GGUF binary reader."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def _check(self, n: int) -> None:
        if self._pos + n > len(self._data):
            raise FormatError(
                f"Unexpected end of GGUF data at offset {self._pos}, need {n} bytes"
            )

    def read_bytes(self, n: int) -> bytes:
        self._check(n)
        result = self._data[self._pos : self._pos + n]
        self._pos += n
        return result

    def read_uint8(self) -> int:
        self._check(1)
        val = struct.unpack_from("<B", self._data, self._pos)[0]
        self._pos += 1
        return val

    def read_int8(self) -> int:
        self._check(1)
        val = struct.unpack_from("<b", self._data, self._pos)[0]
        self._pos += 1
        return val

    def read_uint16(self) -> int:
        self._check(2)
        val = struct.unpack_from("<H", self._data, self._pos)[0]
        self._pos += 2
        return val

    def read_int16(self) -> int:
        self._check(2)
        val = struct.unpack_from("<h", self._data, self._pos)[0]
        self._pos += 2
        return val

    def read_uint32(self) -> int:
        self._check(4)
        val = struct.unpack_from("<I", self._data, self._pos)[0]
        self._pos += 4
        return val

    def read_int32(self) -> int:
        self._check(4)
        val = struct.unpack_from("<i", self._data, self._pos)[0]
        self._pos += 4
        return val

    def read_uint64(self) -> int:
        self._check(8)
        val = struct.unpack_from("<Q", self._data, self._pos)[0]
        self._pos += 8
        return val

    def read_int64(self) -> int:
        self._check(8)
        val = struct.unpack_from("<q", self._data, self._pos)[0]
        self._pos += 8
        return val

    def read_float32(self) -> float:
        self._check(4)
        val = struct.unpack_from("<f", self._data, self._pos)[0]
        self._pos += 4
        return val

    def read_float64(self) -> float:
        self._check(8)
        val = struct.unpack_from("<d", self._data, self._pos)[0]
        self._pos += 8
        return val

    def read_bool(self) -> bool:
        return self.read_uint8() != 0

    def read_string(self) -> str:
        length = self.read_uint64()
        raw = self.read_bytes(length)
        return raw.decode("utf-8")

    def read_value(self, value_type: int) -> Any:
        """Read a GGUF metadata value of the given type."""
        if value_type == _GGUF_TYPE_UINT8:
            return self.read_uint8()
        elif value_type == _GGUF_TYPE_INT8:
            return self.read_int8()
        elif value_type == _GGUF_TYPE_UINT16:
            return self.read_uint16()
        elif value_type == _GGUF_TYPE_INT16:
            return self.read_int16()
        elif value_type == _GGUF_TYPE_UINT32:
            return self.read_uint32()
        elif value_type == _GGUF_TYPE_INT32:
            return self.read_int32()
        elif value_type == _GGUF_TYPE_FLOAT32:
            return self.read_float32()
        elif value_type == _GGUF_TYPE_BOOL:
            return self.read_bool()
        elif value_type == _GGUF_TYPE_STRING:
            return self.read_string()
        elif value_type == _GGUF_TYPE_ARRAY:
            elem_type = self.read_uint32()
            count = self.read_uint64()
            return [self.read_value(elem_type) for _ in range(count)]
        elif value_type == _GGUF_TYPE_UINT64:
            return self.read_uint64()
        elif value_type == _GGUF_TYPE_INT64:
            return self.read_int64()
        elif value_type == _GGUF_TYPE_FLOAT64:
            return self.read_float64()
        else:
            raise FormatError(f"Unknown GGUF value type: {value_type}")

    @property
    def pos(self) -> int:
        return self._pos


def _parse_header(reader: _GGUFReader) -> tuple[int, int, int]:
    """Parse GGUF magic, version, tensor_count, metadata_kv_count."""
    magic = reader.read_bytes(4)
    if magic != GGUF_MAGIC:
        raise FormatError(f"Not a GGUF file: invalid magic {magic!r}")

    version = reader.read_uint32()
    if version not in (2, 3):
        raise FormatError(f"Unsupported GGUF version: {version}")

    tensor_count = reader.read_uint64()
    metadata_kv_count = reader.read_uint64()
    return version, tensor_count, metadata_kv_count


def parse_gguf(path: str | Path) -> GGUFInfo:
    """Parse a GGUF file header and return structured info.

    Reads the magic bytes, version, metadata key-value pairs,
    and tensor info entries from the GGUF binary format.
    """
    path = Path(path)
    with open(path, "rb") as f:
        data = f.read()

    return parse_gguf_bytes(data)


def parse_gguf_bytes(data: bytes) -> GGUFInfo:
    """Parse GGUF info from raw bytes (useful for testing)."""
    reader = _GGUFReader(data)

    version, tensor_count, kv_count = _parse_header(reader)

    # Read metadata key-value pairs
    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = reader.read_string()
        value_type = reader.read_uint32()
        value = reader.read_value(value_type)
        metadata[key] = value

    # Read tensor info entries
    tensors: list[GGUFTensorEntry] = []
    for _ in range(tensor_count):
        name = reader.read_string()
        n_dims = reader.read_uint32()
        dims = [reader.read_uint64() for _ in range(n_dims)]
        dtype_id = reader.read_uint32()
        offset = reader.read_uint64()
        dtype_str = _GGUF_TENSOR_DTYPE.get(dtype_id, f"UNKNOWN({dtype_id})")
        tensors.append(GGUFTensorEntry(
            name=name,
            shape=dims,
            dtype=dtype_str,
            offset=offset,
        ))

    return GGUFInfo(
        version=version,
        tensor_count=tensor_count,
        metadata=metadata,
        tensors=tensors,
    )


def inspect_gguf(path: str | Path) -> CheckpointInfo:
    """Inspect a GGUF file and return a standard CheckpointInfo."""
    path = Path(path)
    file_size = path.stat().st_size
    info = parse_gguf(path)

    tensors: list[TensorInfo] = []
    for t in info.tensors:
        dtype = _GGUF_TO_CKPT_DTYPE.get(t.dtype, DType.UNKNOWN)
        tensors.append(TensorInfo(
            name=t.name,
            shape=t.shape,
            dtype=dtype,
            offset_start=t.offset,
        ))

    return CheckpointInfo(
        path=str(path),
        format=CheckpointFormat.UNKNOWN,  # GGUF not in the enum yet
        file_size=file_size,
        tensors=tensors,
        metadata=info.metadata,
    )


def format_gguf_info(info: GGUFInfo) -> str:
    """Format GGUFInfo as a human-readable summary string."""
    lines: list[str] = []
    lines.append(f"GGUF v{info.version}")
    lines.append(f"Tensors: {info.tensor_count}")

    if info.metadata:
        lines.append(f"Metadata entries: {len(info.metadata)}")
        # Show architecture info if available
        for key in ("general.architecture", "general.name", "general.file_type"):
            if key in info.metadata:
                lines.append(f"  {key}: {info.metadata[key]}")

    if info.tensors:
        # Summarise dtypes
        dtype_counts: dict[str, int] = {}
        total_elements = 0
        for t in info.tensors:
            dtype_counts[t.dtype] = dtype_counts.get(t.dtype, 0) + 1
            elements = 1
            for d in t.shape:
                elements *= d
            total_elements += elements

        lines.append(f"Total elements: {total_elements:,}")
        dtype_summary = ", ".join(f"{dt}: {c}" for dt, c in sorted(dtype_counts.items()))
        lines.append(f"Dtype distribution: {dtype_summary}")

        # Show first few tensors
        show = min(5, len(info.tensors))
        lines.append(f"First {show} tensors:")
        for t in info.tensors[:show]:
            shape_str = "×".join(str(d) for d in t.shape)
            lines.append(f"  {t.name} [{shape_str}] {t.dtype}")
        if len(info.tensors) > show:
            lines.append(f"  ... and {len(info.tensors) - show} more")

    return "\n".join(lines)

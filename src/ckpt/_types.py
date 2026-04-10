"""Core types for ckpt."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CheckpointFormat(str, Enum):
    """Supported checkpoint formats."""

    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    NUMPY = "numpy"
    UNKNOWN = "unknown"


class DType(str, Enum):
    """Common tensor data types."""

    F32 = "F32"
    F16 = "F16"
    BF16 = "BF16"
    F64 = "F64"
    I64 = "I64"
    I32 = "I32"
    I16 = "I16"
    I8 = "I8"
    U8 = "U8"
    BOOL = "BOOL"
    UNKNOWN = "UNKNOWN"


# Bytes per element for each dtype
DTYPE_SIZES: dict[DType, int] = {
    DType.F32: 4, DType.F16: 2, DType.BF16: 2, DType.F64: 8,
    DType.I64: 8, DType.I32: 4, DType.I16: 2, DType.I8: 1,
    DType.U8: 1, DType.BOOL: 1,
}


@dataclass
class TensorInfo:
    """Metadata for a single tensor in a checkpoint."""

    name: str
    shape: list[int]
    dtype: DType
    offset_start: int = 0
    offset_end: int = 0

    @property
    def numel(self) -> int:
        """Number of elements."""
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def size_bytes(self) -> int:
        """Size in bytes."""
        return self.numel * DTYPE_SIZES.get(self.dtype, 0)

    @property
    def shape_str(self) -> str:
        return "×".join(str(s) for s in self.shape)


@dataclass
class CheckpointInfo:
    """Metadata for an entire checkpoint."""

    path: str
    format: CheckpointFormat
    file_size: int
    tensors: list[TensorInfo]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tensors(self) -> int:
        return len(self.tensors)

    @property
    def n_parameters(self) -> int:
        return sum(t.numel for t in self.tensors)

    @property
    def total_bytes(self) -> int:
        return sum(t.size_bytes for t in self.tensors)

    def dtype_summary(self) -> dict[str, int]:
        """Count parameters per dtype."""
        counts: dict[str, int] = {}
        for t in self.tensors:
            key = t.dtype.value
            counts[key] = counts.get(key, 0) + t.numel
        return counts

    def layer_groups(self) -> dict[str, list[TensorInfo]]:
        """Group tensors by layer prefix (e.g., 'model.layers.0')."""
        groups: dict[str, list[TensorInfo]] = {}
        for t in self.tensors:
            parts = t.name.split(".")
            # Take first 3 parts as group key, or full name if shorter
            key = ".".join(parts[:3]) if len(parts) > 3 else t.name
            groups.setdefault(key, []).append(t)
        return groups


@dataclass
class DiffEntry:
    """A difference between two checkpoints."""

    tensor_name: str
    change_type: str  # "added", "removed", "shape_changed", "dtype_changed", "values_changed"
    details: str = ""


@dataclass
class DiffResult:
    """Result of comparing two checkpoints."""

    path_a: str
    path_b: str
    entries: list[DiffEntry]
    n_shared: int = 0
    n_identical: int = 0

    @property
    def n_changes(self) -> int:
        return len(self.entries)

    @property
    def has_changes(self) -> bool:
        return len(self.entries) > 0


@dataclass
class MergeConfig:
    """Configuration for LoRA adapter merging."""

    base_path: str
    adapter_path: str
    output_path: str
    alpha: float = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 2.0):
            raise ValueError(f"alpha must be between 0.0 and 2.0, got {self.alpha}")


class CkptError(Exception):
    """Base exception for ckpt."""


class FormatError(CkptError):
    """Unsupported or corrupt file format."""


class MergeError(CkptError):
    """Error during checkpoint merging."""

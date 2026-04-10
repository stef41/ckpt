"""Checkpoint size reduction estimator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ckpt._types import CheckpointInfo, DType, DTYPE_SIZES, TensorInfo
from ckpt.inspect import format_size


# Bytes per element — extends DTYPE_SIZES with sub-byte and string aliases
_DTYPE_BYTES: Dict[str, float] = {
    "float32": 4, "fp32": 4, "f32": 4,
    "float16": 2, "fp16": 2, "f16": 2,
    "bfloat16": 2, "bf16": 2,
    "int8": 1, "i8": 1,
    "int4": 0.5, "i4": 0.5,
}

# Also accept DType enum values
for _dt, _sz in DTYPE_SIZES.items():
    _DTYPE_BYTES[_dt.value.lower()] = _sz


def _bytes_per_element(dtype: str) -> float:
    """Resolve dtype string to bytes-per-element."""
    key = dtype.strip().lower()
    if key not in _DTYPE_BYTES:
        raise ValueError(
            f"Unknown dtype '{dtype}'. Supported: {sorted(set(_DTYPE_BYTES.keys()))}"
        )
    return _DTYPE_BYTES[key]


@dataclass
class TensorEstimate:
    """Per-tensor size estimation."""

    name: str
    original_dtype: str
    target_dtype: str
    numel: int
    original_bytes: int
    estimated_bytes: int

    @property
    def reduction_bytes(self) -> int:
        return self.original_bytes - self.estimated_bytes

    @property
    def reduction_percent(self) -> float:
        if self.original_bytes == 0:
            return 0.0
        return (self.reduction_bytes / self.original_bytes) * 100


@dataclass
class EstimationResult:
    """Result of a size-reduction estimation."""

    original_size: int
    estimated_size: int
    target_dtype: str
    per_tensor: List[TensorEstimate] = field(default_factory=list)

    @property
    def reduction_bytes(self) -> int:
        return self.original_size - self.estimated_size

    @property
    def reduction_percent(self) -> float:
        if self.original_size == 0:
            return 0.0
        return (self.reduction_bytes / self.original_size) * 100


def estimate_reduction(
    info: CheckpointInfo,
    target_dtype: str = "float16",
) -> EstimationResult:
    """Estimate file-size reduction from casting every tensor to *target_dtype*.

    Parameters
    ----------
    info:
        A :class:`CheckpointInfo` obtained via :func:`ckpt.inspect`.
    target_dtype:
        Target dtype string (e.g. ``"float16"``, ``"bfloat16"``, ``"int8"``).

    Returns
    -------
    EstimationResult
    """
    target_bpe = _bytes_per_element(target_dtype)
    per_tensor: List[TensorEstimate] = []
    total_original = 0
    total_estimated = 0

    for t in info.tensors:
        orig = t.size_bytes
        est = int(math.ceil(t.numel * target_bpe))
        total_original += orig
        total_estimated += est
        per_tensor.append(
            TensorEstimate(
                name=t.name,
                original_dtype=t.dtype.value,
                target_dtype=target_dtype,
                numel=t.numel,
                original_bytes=orig,
                estimated_bytes=est,
            )
        )

    return EstimationResult(
        original_size=total_original,
        estimated_size=total_estimated,
        target_dtype=target_dtype,
        per_tensor=per_tensor,
    )


@dataclass
class QuantEstimationResult:
    """Result of a quantisation size estimation."""

    original_size: int
    estimated_size: int
    bits: int
    total_params: int

    @property
    def reduction_bytes(self) -> int:
        return self.original_size - self.estimated_size

    @property
    def reduction_percent(self) -> float:
        if self.original_size == 0:
            return 0.0
        return (self.reduction_bytes / self.original_size) * 100


def estimate_quantized_size(
    info: CheckpointInfo,
    bits: int = 4,
) -> QuantEstimationResult:
    """Estimate model size if every parameter were quantised to *bits*-bit.

    The formula is simply ``total_params * bits / 8``.

    Parameters
    ----------
    info:
        A :class:`CheckpointInfo`.
    bits:
        Target quantisation bit-width (e.g. 4, 8).

    Returns
    -------
    QuantEstimationResult
    """
    if bits <= 0:
        raise ValueError("bits must be a positive integer")
    total_params = info.n_parameters
    est = int(math.ceil(total_params * bits / 8))
    return QuantEstimationResult(
        original_size=info.total_bytes,
        estimated_size=est,
        bits=bits,
        total_params=total_params,
    )


def format_estimation(result: EstimationResult) -> str:
    """Format an :class:`EstimationResult` as human-readable text."""
    lines: List[str] = []
    lines.append(f"Target dtype: {result.target_dtype}")
    lines.append(f"Original size: {format_size(result.original_size)}")
    lines.append(f"Estimated size: {format_size(result.estimated_size)}")
    lines.append(
        f"Reduction: {format_size(result.reduction_bytes)} "
        f"({result.reduction_percent:.1f}%)"
    )
    if result.per_tensor:
        lines.append("")
        lines.append(f"{'Tensor':<50} {'From':<6} {'To':<10} {'Saved':>12}")
        lines.append("-" * 80)
        for te in result.per_tensor:
            lines.append(
                f"{te.name:<50} {te.original_dtype:<6} {te.target_dtype:<10} "
                f"{format_size(te.reduction_bytes):>12}"
            )
    return "\n".join(lines)

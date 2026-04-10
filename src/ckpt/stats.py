"""Checkpoint weight statistics — per-layer analysis without full loading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ckpt._types import CheckpointInfo, TensorInfo


@dataclass
class TensorStats:
    """Statistics for a single tensor."""

    name: str
    shape: List[int]
    numel: int
    size_bytes: int
    dtype: str
    # Value stats (only if tensor data is available)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    sparsity: Optional[float] = None  # fraction of zeros
    abs_mean: Optional[float] = None


@dataclass
class CheckpointStats:
    """Aggregate statistics for a checkpoint."""

    path: str
    n_tensors: int
    n_parameters: int
    total_bytes: int
    dtype_counts: Dict[str, int]
    layer_groups: Dict[str, int]  # group -> n_params
    tensor_stats: List[TensorStats]

    @property
    def total_size_human(self) -> str:
        size = self.total_bytes
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024  # type: ignore[assignment]
        return f"{size:.1f} PB"


def compute_tensor_stats(
    name: str,
    values: Sequence[float],
    shape: List[int],
    dtype: str,
    size_bytes: int,
) -> TensorStats:
    """Compute statistics for a tensor's values."""
    numel = len(values)
    if numel == 0:
        return TensorStats(
            name=name, shape=shape, numel=0,
            size_bytes=size_bytes, dtype=dtype,
        )

    mean = sum(values) / numel
    variance = sum((v - mean) ** 2 for v in values) / numel
    std = math.sqrt(variance) if variance > 0 else 0.0
    n_zeros = sum(1 for v in values if v == 0.0)
    abs_vals = [abs(v) for v in values]

    return TensorStats(
        name=name,
        shape=shape,
        numel=numel,
        size_bytes=size_bytes,
        dtype=dtype,
        mean=mean,
        std=std,
        min_val=min(values),
        max_val=max(values),
        sparsity=n_zeros / numel if numel > 0 else 0.0,
        abs_mean=sum(abs_vals) / numel,
    )


def stats_from_info(info: CheckpointInfo) -> CheckpointStats:
    """Compute structural stats from CheckpointInfo (no values needed)."""
    layer_groups: Dict[str, int] = {}
    for group_name, tensors in info.layer_groups().items():
        layer_groups[group_name] = sum(t.numel for t in tensors)

    tensor_stats = [
        TensorStats(
            name=t.name,
            shape=t.shape,
            numel=t.numel,
            size_bytes=t.size_bytes,
            dtype=t.dtype.value,
        )
        for t in info.tensors
    ]

    return CheckpointStats(
        path=info.path,
        n_tensors=info.n_tensors,
        n_parameters=info.n_parameters,
        total_bytes=info.total_bytes,
        dtype_counts=info.dtype_summary(),
        layer_groups=layer_groups,
        tensor_stats=tensor_stats,
    )


def format_stats(stats: CheckpointStats) -> str:
    """Format stats as a human-readable string."""
    lines: list[str] = []
    lines.append(f"Checkpoint: {stats.path}")
    lines.append(f"Parameters: {stats.n_parameters:,} ({_format_params(stats.n_parameters)})")
    lines.append(f"Tensors: {stats.n_tensors}")
    lines.append(f"Size: {stats.total_size_human}")
    lines.append("")

    # Dtype breakdown
    lines.append("Data types:")
    for dtype, count in sorted(stats.dtype_counts.items()):
        pct = count / stats.n_parameters * 100 if stats.n_parameters > 0 else 0
        lines.append(f"  {dtype}: {count:,} ({pct:.1f}%)")
    lines.append("")

    # Top tensors by size
    sorted_tensors = sorted(stats.tensor_stats, key=lambda t: t.size_bytes, reverse=True)
    lines.append("Largest tensors:")
    for t in sorted_tensors[:10]:
        lines.append(f"  {t.name}: {t.shape} {t.dtype} ({t.numel:,} params)")

    return "\n".join(lines)


def _format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)

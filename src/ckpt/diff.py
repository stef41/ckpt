"""Diff two checkpoints — find structural and value changes."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from ckpt._types import (
    CheckpointInfo,
    DiffEntry,
    DiffResult,
    TensorInfo,
)
from ckpt.inspect import inspect


def diff_infos(info_a: CheckpointInfo, info_b: CheckpointInfo) -> DiffResult:
    """Compare two CheckpointInfo objects structurally.

    Checks for:
    - Added tensors (in B but not A)
    - Removed tensors (in A but not B)
    - Shape changes
    - Dtype changes
    """
    tensors_a = {t.name: t for t in info_a.tensors}
    tensors_b = {t.name: t for t in info_b.tensors}

    names_a = set(tensors_a.keys())
    names_b = set(tensors_b.keys())

    entries: list[DiffEntry] = []
    n_shared = 0
    n_identical = 0

    # Removed (in A, not in B)
    for name in sorted(names_a - names_b):
        t = tensors_a[name]
        entries.append(DiffEntry(
            tensor_name=name,
            change_type="removed",
            details=f"shape={t.shape_str} dtype={t.dtype.value}",
        ))

    # Added (in B, not in A)
    for name in sorted(names_b - names_a):
        t = tensors_b[name]
        entries.append(DiffEntry(
            tensor_name=name,
            change_type="added",
            details=f"shape={t.shape_str} dtype={t.dtype.value}",
        ))

    # Shared — check for changes
    for name in sorted(names_a & names_b):
        ta = tensors_a[name]
        tb = tensors_b[name]
        n_shared += 1
        changed = False

        if ta.shape != tb.shape:
            entries.append(DiffEntry(
                tensor_name=name,
                change_type="shape_changed",
                details=f"{ta.shape_str} → {tb.shape_str}",
            ))
            changed = True

        if ta.dtype != tb.dtype:
            entries.append(DiffEntry(
                tensor_name=name,
                change_type="dtype_changed",
                details=f"{ta.dtype.value} → {tb.dtype.value}",
            ))
            changed = True

        if not changed:
            n_identical += 1

    return DiffResult(
        path_a=info_a.path,
        path_b=info_b.path,
        entries=entries,
        n_shared=n_shared,
        n_identical=n_identical,
    )


def diff(
    path_a: Union[str, Path],
    path_b: Union[str, Path],
) -> DiffResult:
    """Diff two checkpoint files."""
    info_a = inspect(path_a)
    info_b = inspect(path_b)
    return diff_infos(info_a, info_b)


def format_diff(result: DiffResult) -> str:
    """Format a DiffResult as a human-readable string."""
    lines: list[str] = []
    lines.append(f"Comparing:")
    lines.append(f"  A: {result.path_a}")
    lines.append(f"  B: {result.path_b}")
    lines.append(f"")
    lines.append(f"Shared tensors: {result.n_shared}")
    lines.append(f"Identical: {result.n_identical}")
    lines.append(f"Changes: {result.n_changes}")

    if result.entries:
        lines.append("")
        for entry in result.entries:
            symbol = {"added": "+", "removed": "-", "shape_changed": "~", "dtype_changed": "~", "values_changed": "≠"}
            s = symbol.get(entry.change_type, "?")
            lines.append(f"  [{s}] {entry.tensor_name}: {entry.change_type} ({entry.details})")

    return "\n".join(lines)

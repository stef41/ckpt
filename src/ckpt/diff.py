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


# ANSI escape codes
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_SYMBOL_MAP = {
    "added": ("+", _GREEN),
    "removed": ("-", _RED),
    "shape_changed": ("~", _YELLOW),
    "dtype_changed": ("~", _YELLOW),
    "values_changed": ("≠", _YELLOW),
}


def format_diff_rich(result: DiffResult) -> str:
    """Format a DiffResult with ANSI color codes for terminal display.

    - Added tensors: green with ``+`` prefix
    - Removed tensors: red with ``-`` prefix
    - Shape/dtype changes: yellow with ``~`` prefix
    """
    lines: list[str] = []
    lines.append(f"{_BOLD}Comparing:{_RESET}")
    lines.append(f"  A: {result.path_a}")
    lines.append(f"  B: {result.path_b}")
    lines.append("")
    lines.append(f"Shared tensors: {result.n_shared}")
    lines.append(f"Identical: {result.n_identical}")
    lines.append(f"Changes: {result.n_changes}")

    if result.entries:
        lines.append("")
        for entry in result.entries:
            symbol, color = _SYMBOL_MAP.get(entry.change_type, ("?", ""))
            lines.append(
                f"  {color}{symbol} {entry.tensor_name}: "
                f"{entry.change_type} ({entry.details}){_RESET}"
            )

    # Summary line
    added = sum(1 for e in result.entries if e.change_type == "added")
    removed = sum(1 for e in result.entries if e.change_type == "removed")
    changed = sum(1 for e in result.entries if e.change_type not in ("added", "removed"))
    lines.append("")
    lines.append(
        f"{_BOLD}Summary:{_RESET} "
        f"{_GREEN}+{added} added{_RESET}, "
        f"{_RED}-{removed} removed{_RESET}, "
        f"{_YELLOW}~{changed} changed{_RESET}"
    )
    return "\n".join(lines)


def format_diff_table(result: DiffResult) -> str:
    """Format a DiffResult as a plain-text table.

    Columns: Status | Tensor | Shape A | Shape B | Dtype
    """
    symbol_map = {
        "added": "added",
        "removed": "removed",
        "shape_changed": "shape_changed",
        "dtype_changed": "dtype_changed",
        "values_changed": "values_changed",
    }

    rows: list[tuple[str, str, str, str, str]] = []
    for entry in result.entries:
        status = symbol_map.get(entry.change_type, entry.change_type)
        shape_a = ""
        shape_b = ""
        dtype = ""

        if entry.change_type == "added":
            # details: "shape=... dtype=..."
            parts = entry.details.split(" dtype=")
            if len(parts) == 2:
                shape_b = parts[0].replace("shape=", "")
                dtype = parts[1]
        elif entry.change_type == "removed":
            parts = entry.details.split(" dtype=")
            if len(parts) == 2:
                shape_a = parts[0].replace("shape=", "")
                dtype = parts[1]
        elif entry.change_type == "shape_changed":
            # details: "AxB → CxD"
            halves = entry.details.split(" → ")
            if len(halves) == 2:
                shape_a = halves[0]
                shape_b = halves[1]
        elif entry.change_type == "dtype_changed":
            # details: "F32 → F16"
            dtype = entry.details

        rows.append((status, entry.tensor_name, shape_a, shape_b, dtype))

    headers = ("Status", "Tensor", "Shape A", "Shape B", "Dtype")
    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_row(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    lines: list[str] = []
    header_line = _fmt_row(headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in rows:
        lines.append(_fmt_row(row))

    return "\n".join(lines)

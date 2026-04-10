"""Validate checkpoint integrity."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from ckpt._types import CheckpointFormat, CkptError, FormatError, DTYPE_SIZES, DType
from ckpt.inspect import detect_format


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: str  # "error" or "warning"
    message: str


@dataclass
class ValidationResult:
    """Result of validating a checkpoint."""

    path: str
    valid: bool
    issues: List[ValidationIssue]
    format: CheckpointFormat


def validate_safetensors(path: Union[str, Path]) -> ValidationResult:
    """Validate a SafeTensors file for integrity.

    Checks:
    - File is large enough for header
    - Header length is reasonable
    - Header is valid JSON
    - Tensor offsets don't overlap or exceed file size
    - Tensor data_offsets are consistent with dtype and shape
    """
    path = Path(path)
    issues: list[ValidationIssue] = []
    file_size = path.stat().st_size

    if file_size < 8:
        issues.append(ValidationIssue("error", "File too small (< 8 bytes)"))
        return ValidationResult(str(path), False, issues, CheckpointFormat.SAFETENSORS)

    with open(path, "rb") as f:
        raw_len = f.read(8)
        header_len = struct.unpack("<Q", raw_len)[0]

        if header_len > file_size - 8:
            issues.append(ValidationIssue("error", f"Header length ({header_len}) exceeds file size ({file_size})"))
            return ValidationResult(str(path), False, issues, CheckpointFormat.SAFETENSORS)

        if header_len > 100_000_000:
            issues.append(ValidationIssue("warning", f"Unusually large header: {header_len} bytes"))

        header_bytes = f.read(header_len)

    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as e:
        issues.append(ValidationIssue("error", f"Invalid JSON header: {e}"))
        return ValidationResult(str(path), False, issues, CheckpointFormat.SAFETENSORS)

    if not isinstance(header, dict):
        issues.append(ValidationIssue("error", "Header is not a JSON object"))
        return ValidationResult(str(path), False, issues, CheckpointFormat.SAFETENSORS)

    data_start = 8 + header_len
    data_size = file_size - data_start

    _st_dtype_sizes = {
        "F32": 4, "F16": 2, "BF16": 2, "F64": 8,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    }

    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            issues.append(ValidationIssue("warning", f"Tensor '{name}': info is not a dict"))
            continue

        dtype = info.get("dtype", "")
        shape = info.get("shape", [])
        offsets = info.get("data_offsets", [])

        if dtype not in _st_dtype_sizes:
            issues.append(ValidationIssue("warning", f"Tensor '{name}': unknown dtype '{dtype}'"))

        if len(offsets) != 2:
            issues.append(ValidationIssue("error", f"Tensor '{name}': missing data_offsets"))
            continue

        start, end = offsets
        if end > data_size:
            issues.append(ValidationIssue("error", f"Tensor '{name}': offset end ({end}) exceeds data region ({data_size})"))
        if start > end:
            issues.append(ValidationIssue("error", f"Tensor '{name}': start offset > end offset"))

        # Check size matches dtype * numel
        if dtype in _st_dtype_sizes and shape:
            numel = 1
            for s in shape:
                numel *= s
            expected_bytes = numel * _st_dtype_sizes[dtype]
            actual_bytes = end - start
            if actual_bytes != expected_bytes:
                issues.append(ValidationIssue(
                    "warning",
                    f"Tensor '{name}': expected {expected_bytes} bytes but region is {actual_bytes}",
                ))

    has_errors = any(i.severity == "error" for i in issues)
    return ValidationResult(str(path), not has_errors, issues, CheckpointFormat.SAFETENSORS)


def validate(path: Union[str, Path]) -> ValidationResult:
    """Auto-detect format and validate a checkpoint."""
    path = Path(path)
    if not path.exists():
        return ValidationResult(
            str(path), False,
            [ValidationIssue("error", "File does not exist")],
            CheckpointFormat.UNKNOWN,
        )

    fmt = detect_format(path)
    if fmt == CheckpointFormat.SAFETENSORS:
        return validate_safetensors(path)

    # For other formats, basic checks only
    file_size = path.stat().st_size
    issues: list[ValidationIssue] = []
    if file_size == 0:
        issues.append(ValidationIssue("error", "File is empty"))
    return ValidationResult(str(path), len(issues) == 0, issues, fmt)

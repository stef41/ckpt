"""Inspect checkpoint files — parse headers without loading full tensors."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from ckpt._types import (
    CheckpointFormat,
    CheckpointInfo,
    DType,
    FormatError,
    TensorInfo,
)

# SafeTensors dtype mapping
_ST_DTYPE_MAP: dict[str, DType] = {
    "F32": DType.F32, "F16": DType.F16, "BF16": DType.BF16, "F64": DType.F64,
    "I64": DType.I64, "I32": DType.I32, "I16": DType.I16, "I8": DType.I8,
    "U8": DType.U8, "BOOL": DType.BOOL,
}


def detect_format(path: str | Path) -> CheckpointFormat:
    """Detect checkpoint format from file extension and magic bytes."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".safetensors":
        return CheckpointFormat.SAFETENSORS
    if suffix in (".bin", ".pt", ".pth"):
        return CheckpointFormat.PYTORCH
    if suffix == ".npy" or suffix == ".npz":
        return CheckpointFormat.NUMPY

    # Check magic bytes
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        # SafeTensors starts with 8-byte little-endian header length
        if len(header) == 8:
            header_len = struct.unpack("<Q", header)[0]
            if 1 < header_len < 100_000_000:
                return CheckpointFormat.SAFETENSORS
    except (OSError, struct.error):
        pass

    return CheckpointFormat.UNKNOWN


def inspect_safetensors(path: str | Path) -> CheckpointInfo:
    """Parse a SafeTensors file header without loading tensor data.

    SafeTensors format:
    - 8 bytes: header length (little-endian u64)
    - header_length bytes: JSON header describing tensors
    - remaining: raw tensor data
    """
    path = Path(path)
    file_size = path.stat().st_size

    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise FormatError(f"File too small to be SafeTensors: {path}")

        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len > file_size or header_len > 100_000_000:
            raise FormatError(f"Invalid SafeTensors header length: {header_len}")

        header_bytes = f.read(header_len)

    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as e:
        raise FormatError(f"Invalid SafeTensors header JSON: {e}") from e

    # Extract metadata (stored under __metadata__ key)
    metadata: dict[str, Any] = {}
    if "__metadata__" in header:
        metadata = header.pop("__metadata__")

    tensors: list[TensorInfo] = []
    for name, info in sorted(header.items()):
        dtype_str = info.get("dtype", "UNKNOWN")
        dtype = _ST_DTYPE_MAP.get(dtype_str, DType.UNKNOWN)
        shape = info.get("shape", [])
        offsets = info.get("data_offsets", [0, 0])

        tensors.append(TensorInfo(
            name=name,
            shape=shape,
            dtype=dtype,
            offset_start=offsets[0] if len(offsets) > 0 else 0,
            offset_end=offsets[1] if len(offsets) > 1 else 0,
        ))

    return CheckpointInfo(
        path=str(path),
        format=CheckpointFormat.SAFETENSORS,
        file_size=file_size,
        tensors=tensors,
        metadata=metadata,
    )


def inspect_pytorch(path: str | Path) -> CheckpointInfo:
    """Inspect a PyTorch checkpoint (.bin/.pt/.pth).

    Requires torch to be installed. Falls back to basic file info if not available.
    """
    path = Path(path)
    file_size = path.stat().st_size

    try:
        import torch
    except ImportError:
        return CheckpointInfo(
            path=str(path),
            format=CheckpointFormat.PYTORCH,
            file_size=file_size,
            tensors=[],
            metadata={"error": "torch not installed — install with: pip install ckpt[torch]"},
        )

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    _torch_dtype_map = {
        torch.float32: DType.F32, torch.float16: DType.F16,
        torch.bfloat16: DType.BF16, torch.float64: DType.F64,
        torch.int64: DType.I64, torch.int32: DType.I32,
        torch.int16: DType.I16, torch.int8: DType.I8,
        torch.uint8: DType.U8, torch.bool: DType.BOOL,
    }

    tensors: list[TensorInfo] = []
    for name, tensor in sorted(state_dict.items()):
        if not hasattr(tensor, "shape"):
            continue
        dtype = _torch_dtype_map.get(tensor.dtype, DType.UNKNOWN)
        tensors.append(TensorInfo(
            name=name,
            shape=list(tensor.shape),
            dtype=dtype,
        ))

    return CheckpointInfo(
        path=str(path),
        format=CheckpointFormat.PYTORCH,
        file_size=file_size,
        tensors=tensors,
    )


def inspect(path: str | Path) -> CheckpointInfo:
    """Auto-detect format and inspect a checkpoint file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    fmt = detect_format(path)
    if fmt == CheckpointFormat.SAFETENSORS:
        return inspect_safetensors(path)
    if fmt == CheckpointFormat.PYTORCH:
        return inspect_pytorch(path)
    raise FormatError(f"Unsupported format for {path} (detected: {fmt.value})")


def format_size(n_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} PB"


def format_params(n: int) -> str:
    """Human-readable parameter count."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)

"""Checkpoint format conversion utilities."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConversionFormat:
    """Supported conversion formats (string constants)."""

    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    NUMPY = "numpy"
    GGUF = "gguf"

    ALL = (SAFETENSORS, PYTORCH, NUMPY, GGUF)

    _EXTENSIONS: dict[str, str] = {
        ".safetensors": SAFETENSORS,
        ".pt": PYTORCH,
        ".pth": PYTORCH,
        ".bin": PYTORCH,
        ".npy": NUMPY,
        ".npz": NUMPY,
        ".gguf": GGUF,
    }


@dataclass
class ConversionConfig:
    """Configuration for a checkpoint conversion."""

    source_format: str
    target_format: str
    dtype: str | None = None
    shard_size_mb: int | None = None

    def __post_init__(self) -> None:
        if self.source_format not in ConversionFormat.ALL:
            raise ValueError(
                f"Unsupported source format: {self.source_format!r}. "
                f"Choose from {ConversionFormat.ALL}"
            )
        if self.target_format not in ConversionFormat.ALL:
            raise ValueError(
                f"Unsupported target format: {self.target_format!r}. "
                f"Choose from {ConversionFormat.ALL}"
            )
        if self.source_format == self.target_format:
            raise ValueError("Source and target formats must differ")


@dataclass
class ConversionResult:
    """Result of a checkpoint conversion."""

    source_format: str
    target_format: str
    source_size: int
    target_size: int
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        if self.source_size == 0:
            return 0.0
        return self.target_size / self.source_size

    @property
    def size_change_pct(self) -> float:
        if self.source_size == 0:
            return 0.0
        return (self.target_size - self.source_size) / self.source_size * 100


# ---------------------------------------------------------------------------
# Internal serialisers / deserialisers operating on plain dicts
# ---------------------------------------------------------------------------

def _serialize_safetensors(tensors: dict[str, dict[str, Any]]) -> bytes:
    """Produce a minimal safetensors-like binary blob from *tensors*.

    Each value in *tensors* is ``{"dtype": str, "shape": list, "data": bytes}``.
    """
    header: dict[str, Any] = {}
    data_parts: list[bytes] = []
    offset = 0
    for name, info in tensors.items():
        raw = info["data"]
        header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + len(raw)],
        }
        data_parts.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_len = len(header_bytes).to_bytes(8, "little")
    return header_len + header_bytes + b"".join(data_parts)


def _deserialize_safetensors(blob: bytes) -> dict[str, dict[str, Any]]:
    header_len = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8: 8 + header_len])
    data_start = 8 + header_len
    result: dict[str, dict[str, Any]] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        result[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data": blob[data_start + start: data_start + end],
        }
    return result


def _serialize_pytorch(tensors: dict[str, dict[str, Any]]) -> bytes:
    """Produce a simple JSON-based blob mimicking a PyTorch state dict."""
    import base64

    payload: dict[str, Any] = {}
    for name, info in tensors.items():
        payload[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_b64": base64.b64encode(info["data"]).decode(),
        }
    return json.dumps(payload).encode()


def _deserialize_pytorch(blob: bytes) -> dict[str, dict[str, Any]]:
    import base64

    payload = json.loads(blob)
    result: dict[str, dict[str, Any]] = {}
    for name, info in payload.items():
        result[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data": base64.b64decode(info["data_b64"]),
        }
    return result


def _serialize_numpy(tensors: dict[str, dict[str, Any]]) -> bytes:
    import base64

    payload: dict[str, Any] = {}
    for name, info in tensors.items():
        payload[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_b64": base64.b64encode(info["data"]).decode(),
        }
    header = json.dumps(payload).encode()
    # Simple framing: 8-byte header length + header
    return len(header).to_bytes(8, "little") + header


def _deserialize_numpy(blob: bytes) -> dict[str, dict[str, Any]]:
    import base64

    header_len = int.from_bytes(blob[:8], "little")
    payload = json.loads(blob[8: 8 + header_len])
    result: dict[str, dict[str, Any]] = {}
    for name, info in payload.items():
        result[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data": base64.b64decode(info["data_b64"]),
        }
    return result


def _serialize_gguf(tensors: dict[str, dict[str, Any]]) -> bytes:
    """Minimal GGUF-like serialisation."""
    import base64

    payload = {
        "magic": "GGUF",
        "tensors": {
            name: {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "data_b64": base64.b64encode(info["data"]).decode(),
            }
            for name, info in tensors.items()
        },
    }
    return json.dumps(payload).encode()


def _deserialize_gguf(blob: bytes) -> dict[str, dict[str, Any]]:
    import base64

    payload = json.loads(blob)
    result: dict[str, dict[str, Any]] = {}
    inner = payload.get("tensors", payload)
    for name, info in inner.items():
        result[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data": base64.b64decode(info["data_b64"]),
        }
    return result


_SERIALIZERS = {
    ConversionFormat.SAFETENSORS: _serialize_safetensors,
    ConversionFormat.PYTORCH: _serialize_pytorch,
    ConversionFormat.NUMPY: _serialize_numpy,
    ConversionFormat.GGUF: _serialize_gguf,
}

_DESERIALIZERS = {
    ConversionFormat.SAFETENSORS: _deserialize_safetensors,
    ConversionFormat.PYTORCH: _deserialize_pytorch,
    ConversionFormat.NUMPY: _deserialize_numpy,
    ConversionFormat.GGUF: _deserialize_gguf,
}


# ---------------------------------------------------------------------------
# FormatConverter
# ---------------------------------------------------------------------------

class FormatConverter:
    """Convert checkpoint files between formats."""

    def __init__(self, config: ConversionConfig) -> None:
        self.config = config

    # -- public API ----------------------------------------------------------

    def convert(self, source_path: str, target_path: str) -> ConversionResult:
        """Read *source_path*, convert, and write *target_path*.

        Returns a :class:`ConversionResult` with statistics.
        """
        t0 = time.monotonic()

        src = Path(source_path)
        tgt = Path(target_path)

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")

        source_blob = src.read_bytes()
        source_size = len(source_blob)

        # Deserialise from source format
        deserialize = _DESERIALIZERS.get(self.config.source_format)
        if deserialize is None:
            raise ValueError(f"No deserialiser for {self.config.source_format}")
        tensors = deserialize(source_blob)

        # Optional dtype annotation rewrite
        if self.config.dtype is not None:
            for info in tensors.values():
                info["dtype"] = self.config.dtype

        # Serialise to target format (potentially sharded)
        serialize = _SERIALIZERS.get(self.config.target_format)
        if serialize is None:
            raise ValueError(f"No serialiser for {self.config.target_format}")

        if self.config.shard_size_mb is not None and self.config.shard_size_mb > 0:
            shards = self._shard(tensors, self.config.shard_size_mb)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            total_target = 0
            for idx, shard_tensors in enumerate(shards):
                shard_blob = serialize(shard_tensors)
                shard_path = tgt.parent / f"{tgt.stem}-{idx:05d}{tgt.suffix}"
                shard_path.write_bytes(shard_blob)
                total_target += len(shard_blob)
            target_size = total_target
            metadata = {"n_shards": len(shards)}
        else:
            target_blob = serialize(tensors)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_bytes(target_blob)
            target_size = len(target_blob)
            metadata = {"n_shards": 1}

        duration = time.monotonic() - t0
        metadata["n_tensors"] = len(tensors)

        return ConversionResult(
            source_format=self.config.source_format,
            target_format=self.config.target_format,
            source_size=source_size,
            target_size=target_size,
            duration_s=duration,
            metadata=metadata,
        )

    @staticmethod
    def detect_format(path: str) -> str:
        """Detect checkpoint format from file extension."""
        ext = Path(path).suffix.lower()
        fmt = ConversionFormat._EXTENSIONS.get(ext)
        if fmt is None:
            raise ValueError(f"Cannot detect format from extension: {ext!r}")
        return fmt

    @staticmethod
    def supported_conversions() -> list[tuple[str, str]]:
        """Return all supported (source, target) format pairs."""
        pairs: list[tuple[str, str]] = []
        for src in ConversionFormat.ALL:
            for tgt in ConversionFormat.ALL:
                if src != tgt:
                    pairs.append((src, tgt))
        return pairs

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _shard(
        tensors: dict[str, dict[str, Any]],
        shard_size_mb: int,
    ) -> list[dict[str, dict[str, Any]]]:
        limit = shard_size_mb * 1024 * 1024
        shards: list[dict[str, dict[str, Any]]] = []
        current: dict[str, dict[str, Any]] = {}
        current_size = 0
        for name, info in tensors.items():
            tensor_size = len(info["data"])
            if current and current_size + tensor_size > limit:
                shards.append(current)
                current = {}
                current_size = 0
            current[name] = info
            current_size += tensor_size
        if current:
            shards.append(current)
        return shards


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def convert_checkpoint(
    source: str,
    target: str,
    config: ConversionConfig | None = None,
) -> ConversionResult:
    """High-level convenience: detect formats from extensions and convert.

    If *config* is ``None``, formats are inferred from file extensions.
    """
    if config is None:
        src_fmt = FormatConverter.detect_format(source)
        tgt_fmt = FormatConverter.detect_format(target)
        config = ConversionConfig(source_format=src_fmt, target_format=tgt_fmt)

    converter = FormatConverter(config)
    return converter.convert(source, target)

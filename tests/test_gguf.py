"""Tests for GGUF format parser."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import pytest

from ckpt._types import CheckpointFormat, DType, FormatError
from ckpt.gguf import (
    GGUFInfo,
    GGUFTensorEntry,
    format_gguf_info,
    inspect_gguf,
    parse_gguf,
    parse_gguf_bytes,
)


def _gguf_string(s: str) -> bytes:
    """Encode a string in GGUF format: uint64 length + raw bytes."""
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_kv(key: str, value_type: int, value_bytes: bytes) -> bytes:
    """Build a GGUF metadata key-value pair."""
    return _gguf_string(key) + struct.pack("<I", value_type) + value_bytes


def _build_gguf(
    version: int = 3,
    tensor_count: int = 0,
    kv_pairs: list[bytes] | None = None,
    tensor_infos: list[bytes] | None = None,
) -> bytes:
    """Build a minimal GGUF binary blob."""
    kv_pairs = kv_pairs or []
    tensor_infos = tensor_infos or []
    header = b"GGUF"
    header += struct.pack("<I", version)
    header += struct.pack("<Q", tensor_count)
    header += struct.pack("<Q", len(kv_pairs))
    data = header
    for kv in kv_pairs:
        data += kv
    for ti in tensor_infos:
        data += ti
    return data


def _build_tensor_info(name: str, dims: list[int], dtype_id: int = 0, offset: int = 0) -> bytes:
    """Build a GGUF tensor info entry."""
    data = _gguf_string(name)
    data += struct.pack("<I", len(dims))
    for d in dims:
        data += struct.pack("<Q", d)
    data += struct.pack("<I", dtype_id)
    data += struct.pack("<Q", offset)
    return data


class TestParseGGUFBytes:
    """Tests for the low-level GGUF byte parser."""

    def test_minimal_valid(self) -> None:
        data = _build_gguf(version=3, tensor_count=0)
        info = parse_gguf_bytes(data)
        assert info.version == 3
        assert info.tensor_count == 0
        assert info.metadata == {}
        assert info.tensors == []

    def test_version_2(self) -> None:
        data = _build_gguf(version=2)
        info = parse_gguf_bytes(data)
        assert info.version == 2

    def test_invalid_magic(self) -> None:
        data = b"XXXX" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
        with pytest.raises(FormatError, match="Not a GGUF file"):
            parse_gguf_bytes(data)

    def test_unsupported_version(self) -> None:
        data = b"GGUF" + struct.pack("<I", 99) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
        with pytest.raises(FormatError, match="Unsupported GGUF version"):
            parse_gguf_bytes(data)

    def test_truncated_file(self) -> None:
        data = b"GGUF" + struct.pack("<I", 3)
        with pytest.raises(FormatError, match="Unexpected end"):
            parse_gguf_bytes(data)

    def test_metadata_uint32(self) -> None:
        kv = _gguf_kv("general.file_type", 4, struct.pack("<I", 7))
        data = _build_gguf(kv_pairs=[kv])
        info = parse_gguf_bytes(data)
        assert info.metadata["general.file_type"] == 7

    def test_metadata_string(self) -> None:
        value_bytes = _gguf_string("llama")
        kv = _gguf_kv("general.architecture", 8, value_bytes)
        data = _build_gguf(kv_pairs=[kv])
        info = parse_gguf_bytes(data)
        assert info.metadata["general.architecture"] == "llama"

    def test_metadata_float32(self) -> None:
        kv = _gguf_kv("score", 6, struct.pack("<f", 3.14))
        data = _build_gguf(kv_pairs=[kv])
        info = parse_gguf_bytes(data)
        assert abs(info.metadata["score"] - 3.14) < 1e-5

    def test_metadata_bool(self) -> None:
        kv = _gguf_kv("flag", 7, struct.pack("<B", 1))
        data = _build_gguf(kv_pairs=[kv])
        info = parse_gguf_bytes(data)
        assert info.metadata["flag"] is True

    def test_metadata_array(self) -> None:
        # Array of 3 uint32 values
        arr_data = struct.pack("<I", 4)  # elem type = uint32
        arr_data += struct.pack("<Q", 3)  # count = 3
        arr_data += struct.pack("<III", 10, 20, 30)
        kv = _gguf_kv("dims", 9, arr_data)
        data = _build_gguf(kv_pairs=[kv])
        info = parse_gguf_bytes(data)
        assert info.metadata["dims"] == [10, 20, 30]

    def test_metadata_int64(self) -> None:
        kv = _gguf_kv("big_num", 11, struct.pack("<q", -12345678901234))
        data = _build_gguf(kv_pairs=[kv])
        info = parse_gguf_bytes(data)
        assert info.metadata["big_num"] == -12345678901234

    def test_metadata_float64(self) -> None:
        kv = _gguf_kv("precise", 12, struct.pack("<d", 2.718281828))
        data = _build_gguf(kv_pairs=[kv])
        info = parse_gguf_bytes(data)
        assert abs(info.metadata["precise"] - 2.718281828) < 1e-9

    def test_single_tensor(self) -> None:
        ti = _build_tensor_info("blk.0.attn.weight", [4096, 4096], dtype_id=1, offset=0)
        data = _build_gguf(tensor_count=1, tensor_infos=[ti])
        info = parse_gguf_bytes(data)
        assert len(info.tensors) == 1
        assert info.tensors[0].name == "blk.0.attn.weight"
        assert info.tensors[0].shape == [4096, 4096]
        assert info.tensors[0].dtype == "F16"
        assert info.tensors[0].offset == 0

    def test_multiple_tensors(self) -> None:
        t1 = _build_tensor_info("weight", [768, 768], dtype_id=0, offset=0)
        t2 = _build_tensor_info("bias", [768], dtype_id=0, offset=768 * 768 * 4)
        data = _build_gguf(tensor_count=2, tensor_infos=[t1, t2])
        info = parse_gguf_bytes(data)
        assert len(info.tensors) == 2
        assert info.tensors[0].name == "weight"
        assert info.tensors[1].name == "bias"
        assert info.tensors[1].shape == [768]

    def test_quantized_dtype(self) -> None:
        ti = _build_tensor_info("q", [1024], dtype_id=2, offset=0)
        data = _build_gguf(tensor_count=1, tensor_infos=[ti])
        info = parse_gguf_bytes(data)
        assert info.tensors[0].dtype == "Q4_0"

    def test_unknown_dtype(self) -> None:
        ti = _build_tensor_info("x", [10], dtype_id=999, offset=0)
        data = _build_gguf(tensor_count=1, tensor_infos=[ti])
        info = parse_gguf_bytes(data)
        assert "UNKNOWN" in info.tensors[0].dtype

    def test_metadata_with_tensors(self) -> None:
        kv = _gguf_kv("general.architecture", 8, _gguf_string("llama"))
        ti = _build_tensor_info("tok_embd.weight", [32000, 4096], dtype_id=1, offset=0)
        data = _build_gguf(tensor_count=1, kv_pairs=[kv], tensor_infos=[ti])
        info = parse_gguf_bytes(data)
        assert info.metadata["general.architecture"] == "llama"
        assert info.tensors[0].name == "tok_embd.weight"

    def test_unknown_value_type_raises(self) -> None:
        kv = _gguf_string("bad_key") + struct.pack("<I", 255) + b"\x00"
        data = _build_gguf(kv_pairs=[kv])
        with pytest.raises(FormatError, match="Unknown GGUF value type"):
            parse_gguf_bytes(data)


class TestParseGGUFFile:
    """Test file-based parsing."""

    def test_roundtrip_via_file(self, tmp_path: Path) -> None:
        kv = _gguf_kv("general.name", 8, _gguf_string("testmodel"))
        ti = _build_tensor_info("w", [64, 64], dtype_id=0, offset=0)
        raw = _build_gguf(version=3, tensor_count=1, kv_pairs=[kv], tensor_infos=[ti])

        path = tmp_path / "model.gguf"
        path.write_bytes(raw)

        info = parse_gguf(path)
        assert info.version == 3
        assert info.metadata["general.name"] == "testmodel"
        assert len(info.tensors) == 1


class TestInspectGGUF:
    """Test conversion to CheckpointInfo."""

    def test_inspect_produces_checkpoint_info(self, tmp_path: Path) -> None:
        ti = _build_tensor_info("layer.weight", [256, 128], dtype_id=0, offset=0)
        raw = _build_gguf(version=3, tensor_count=1, tensor_infos=[ti])

        path = tmp_path / "model.gguf"
        path.write_bytes(raw)

        ci = inspect_gguf(path)
        assert ci.path == str(path)
        assert ci.n_tensors == 1
        assert ci.tensors[0].dtype == DType.F32
        assert ci.tensors[0].shape == [256, 128]
        assert ci.tensors[0].name == "layer.weight"
        assert ci.file_size == len(raw)

    def test_inspect_maps_f16(self, tmp_path: Path) -> None:
        ti = _build_tensor_info("x", [10], dtype_id=1, offset=0)
        raw = _build_gguf(version=3, tensor_count=1, tensor_infos=[ti])
        path = tmp_path / "f16.gguf"
        path.write_bytes(raw)

        ci = inspect_gguf(path)
        assert ci.tensors[0].dtype == DType.F16

    def test_inspect_quantized_becomes_unknown(self, tmp_path: Path) -> None:
        ti = _build_tensor_info("x", [10], dtype_id=2, offset=0)  # Q4_0
        raw = _build_gguf(version=3, tensor_count=1, tensor_infos=[ti])
        path = tmp_path / "q4.gguf"
        path.write_bytes(raw)

        ci = inspect_gguf(path)
        assert ci.tensors[0].dtype == DType.UNKNOWN

    def test_inspect_preserves_metadata(self, tmp_path: Path) -> None:
        kv = _gguf_kv("general.architecture", 8, _gguf_string("mistral"))
        raw = _build_gguf(version=3, kv_pairs=[kv])
        path = tmp_path / "meta.gguf"
        path.write_bytes(raw)

        ci = inspect_gguf(path)
        assert ci.metadata["general.architecture"] == "mistral"


class TestFormatGGUFInfo:
    """Test human-readable formatting."""

    def test_minimal_format(self) -> None:
        info = GGUFInfo(version=3, tensor_count=0)
        text = format_gguf_info(info)
        assert "GGUF v3" in text
        assert "Tensors: 0" in text

    def test_format_with_metadata(self) -> None:
        info = GGUFInfo(
            version=3,
            tensor_count=1,
            metadata={"general.architecture": "llama", "general.name": "MyModel"},
            tensors=[GGUFTensorEntry("w", [4096, 4096], "F16", 0)],
        )
        text = format_gguf_info(info)
        assert "llama" in text
        assert "MyModel" in text
        assert "w" in text
        assert "4096×4096" in text

    def test_format_dtype_distribution(self) -> None:
        info = GGUFInfo(
            version=2,
            tensor_count=3,
            tensors=[
                GGUFTensorEntry("a", [10], "F16", 0),
                GGUFTensorEntry("b", [20], "F16", 0),
                GGUFTensorEntry("c", [30], "Q4_0", 0),
            ],
        )
        text = format_gguf_info(info)
        assert "F16: 2" in text
        assert "Q4_0: 1" in text

    def test_format_truncates_tensor_list(self) -> None:
        tensors = [GGUFTensorEntry(f"t{i}", [100], "F32", 0) for i in range(10)]
        info = GGUFInfo(version=3, tensor_count=10, tensors=tensors)
        text = format_gguf_info(info)
        assert "5 more" in text

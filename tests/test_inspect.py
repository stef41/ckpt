"""Tests for ckpt.inspect — SafeTensors parsing and format detection."""

import json
import struct

import pytest

from ckpt._types import CheckpointFormat, DType, FormatError
from ckpt.inspect import (
    detect_format,
    format_params,
    format_size,
    inspect,
    inspect_safetensors,
)


def _make_safetensors(path, tensors=None, metadata=None):
    """Create a minimal valid SafeTensors file for testing."""
    if tensors is None:
        tensors = {
            "weight": {"dtype": "F32", "shape": [4, 3], "data_offsets": [0, 48]},
            "bias": {"dtype": "F32", "shape": [4], "data_offsets": [48, 64]},
        }
    header = dict(tensors)
    if metadata:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)

    # Calculate total data size
    max_offset = 0
    for info in tensors.values():
        if "data_offsets" in info:
            max_offset = max(max_offset, info["data_offsets"][1])
    data = b"\x00" * max_offset

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        f.write(data)


class TestDetectFormat:
    def test_safetensors_extension(self, tmp_path):
        p = tmp_path / "model.safetensors"
        _make_safetensors(p)
        assert detect_format(p) == CheckpointFormat.SAFETENSORS

    def test_pytorch_extension(self, tmp_path):
        p = tmp_path / "model.bin"
        p.write_bytes(b"\x00" * 100)
        assert detect_format(p) == CheckpointFormat.PYTORCH

    def test_pt_extension(self, tmp_path):
        p = tmp_path / "model.pt"
        p.write_bytes(b"\x00" * 100)
        assert detect_format(p) == CheckpointFormat.PYTORCH

    def test_numpy_extension(self, tmp_path):
        p = tmp_path / "data.npy"
        p.write_bytes(b"\x00" * 100)
        assert detect_format(p) == CheckpointFormat.NUMPY

    def test_unknown_extension(self, tmp_path):
        p = tmp_path / "data.xyz"
        p.write_bytes(b"\x00" * 100)
        assert detect_format(p) == CheckpointFormat.UNKNOWN

    def test_safetensors_by_magic(self, tmp_path):
        p = tmp_path / "model"  # no extension
        _make_safetensors(p)
        assert detect_format(p) == CheckpointFormat.SAFETENSORS


class TestInspectSafetensors:
    def test_basic(self, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_safetensors(p)
        info = inspect_safetensors(p)
        assert info.n_tensors == 2
        assert info.format == CheckpointFormat.SAFETENSORS

    def test_tensor_names(self, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_safetensors(p)
        info = inspect_safetensors(p)
        names = [t.name for t in info.tensors]
        assert "weight" in names
        assert "bias" in names

    def test_tensor_shapes(self, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_safetensors(p)
        info = inspect_safetensors(p)
        weight = next(t for t in info.tensors if t.name == "weight")
        assert weight.shape == [4, 3]
        assert weight.dtype == DType.F32

    def test_metadata(self, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_safetensors(p, metadata={"format": "pt"})
        info = inspect_safetensors(p)
        assert info.metadata.get("format") == "pt"

    def test_file_size(self, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_safetensors(p)
        info = inspect_safetensors(p)
        assert info.file_size == p.stat().st_size

    def test_too_small(self, tmp_path):
        p = tmp_path / "m.safetensors"
        p.write_bytes(b"\x00\x00")
        with pytest.raises(FormatError, match="too small"):
            inspect_safetensors(p)

    def test_invalid_header_len(self, tmp_path):
        p = tmp_path / "m.safetensors"
        # Write header len that exceeds file
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", 999999999))
            f.write(b"\x00" * 16)
        with pytest.raises(FormatError, match="Invalid SafeTensors header"):
            inspect_safetensors(p)

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "m.safetensors"
        bad_header = b"{not valid json"
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", len(bad_header)))
            f.write(bad_header)
        with pytest.raises(FormatError, match="Invalid SafeTensors header JSON"):
            inspect_safetensors(p)

    def test_multiple_dtypes(self, tmp_path):
        p = tmp_path / "m.safetensors"
        tensors = {
            "w_f16": {"dtype": "F16", "shape": [10, 10], "data_offsets": [0, 200]},
            "w_f32": {"dtype": "F32", "shape": [10, 10], "data_offsets": [200, 600]},
            "w_bf16": {"dtype": "BF16", "shape": [5], "data_offsets": [600, 610]},
        }
        _make_safetensors(p, tensors=tensors)
        info = inspect_safetensors(p)
        dtypes = {t.dtype for t in info.tensors}
        assert DType.F16 in dtypes
        assert DType.F32 in dtypes
        assert DType.BF16 in dtypes

    def test_large_tensor_count(self, tmp_path):
        p = tmp_path / "m.safetensors"
        offset = 0
        tensors = {}
        for i in range(100):
            size = 4 * 10  # 10 floats
            tensors[f"layer.{i}.weight"] = {
                "dtype": "F32", "shape": [10], "data_offsets": [offset, offset + size],
            }
            offset += size
        _make_safetensors(p, tensors=tensors)
        info = inspect_safetensors(p)
        assert info.n_tensors == 100


class TestInspectAutoDetect:
    def test_safetensors(self, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_safetensors(p)
        info = inspect(p)
        assert info.format == CheckpointFormat.SAFETENSORS

    def test_not_found(self):
        with pytest.raises(FileNotFoundError):
            inspect("/nonexistent/model.safetensors")

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "m.xyz"
        p.write_bytes(b"\x00" * 100)
        with pytest.raises(FormatError, match="Unsupported"):
            inspect(p)


class TestFormatHelpers:
    def test_format_size_bytes(self):
        assert "B" in format_size(100)

    def test_format_size_kb(self):
        assert "KB" in format_size(2048)

    def test_format_size_mb(self):
        assert "MB" in format_size(5 * 1024 * 1024)

    def test_format_size_gb(self):
        assert "GB" in format_size(2 * 1024 * 1024 * 1024)

    def test_format_params_billions(self):
        assert "B" in format_params(7_000_000_000)

    def test_format_params_millions(self):
        assert "M" in format_params(125_000_000)

    def test_format_params_thousands(self):
        assert "K" in format_params(5000)

    def test_format_params_small(self):
        assert format_params(42) == "42"

"""Tests for ckpt.validate."""

import json
import struct

import pytest

from ckpt._types import CheckpointFormat
from ckpt.validate import validate, validate_safetensors


def _make_valid_st(path):
    """Create a valid SafeTensors test file."""
    header = {
        "weight": {"dtype": "F32", "shape": [4, 3], "data_offsets": [0, 48]},
        "bias": {"dtype": "F32", "shape": [4], "data_offsets": [48, 64]},
    }
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * 64)


class TestValidateSafetensors:
    def test_valid_file(self, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_valid_st(p)
        result = validate_safetensors(p)
        assert result.valid
        assert len(result.issues) == 0

    def test_too_small(self, tmp_path):
        p = tmp_path / "m.safetensors"
        p.write_bytes(b"\x00\x00")
        result = validate_safetensors(p)
        assert not result.valid

    def test_header_exceeds_file(self, tmp_path):
        p = tmp_path / "m.safetensors"
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", 999999))
            f.write(b"\x00" * 16)
        result = validate_safetensors(p)
        assert not result.valid

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "m.safetensors"
        header = b"not json"
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
        result = validate_safetensors(p)
        assert not result.valid

    def test_offset_exceeds_data(self, tmp_path):
        p = tmp_path / "m.safetensors"
        header = {
            "weight": {"dtype": "F32", "shape": [4, 3], "data_offsets": [0, 99999]},
        }
        header_bytes = json.dumps(header).encode()
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\x00" * 48)
        result = validate_safetensors(p)
        # Should have an error about offsets
        assert any("exceeds" in i.message for i in result.issues)

    def test_start_gt_end(self, tmp_path):
        p = tmp_path / "m.safetensors"
        header = {
            "weight": {"dtype": "F32", "shape": [4], "data_offsets": [100, 50]},
        }
        header_bytes = json.dumps(header).encode()
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\x00" * 200)
        result = validate_safetensors(p)
        assert any("start offset > end" in i.message for i in result.issues)

    def test_size_mismatch(self, tmp_path):
        p = tmp_path / "m.safetensors"
        # 4 floats = 16 bytes, but we say offsets span 20 bytes
        header = {
            "weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 20]},
        }
        header_bytes = json.dumps(header).encode()
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\x00" * 20)
        result = validate_safetensors(p)
        assert any("expected" in i.message for i in result.issues)


class TestValidateAutoDetect:
    def test_safetensors(self, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_valid_st(p)
        result = validate(p)
        assert result.valid

    def test_file_not_found(self):
        result = validate("/nonexistent/file.safetensors")
        assert not result.valid
        assert result.issues[0].message == "File does not exist"

    def test_empty_file(self, tmp_path):
        p = tmp_path / "m.bin"
        p.write_bytes(b"")
        result = validate(p)
        assert not result.valid

    def test_pytorch_basic(self, tmp_path):
        p = tmp_path / "m.bin"
        p.write_bytes(b"\x00" * 100)
        result = validate(p)
        assert result.valid  # basic check passes for non-empty files
        assert result.format == CheckpointFormat.PYTORCH

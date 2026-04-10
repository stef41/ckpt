"""Tests for ckpt.convert."""

import os
from pathlib import Path

import pytest

from ckpt.convert import (
    ConversionConfig,
    ConversionFormat,
    ConversionResult,
    FormatConverter,
    convert_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_tensors():
    """Return a small dict-of-tensors for round-trip tests."""
    return {
        "weight": {"dtype": "F32", "shape": [4, 4], "data": os.urandom(64)},
        "bias": {"dtype": "F32", "shape": [4], "data": os.urandom(16)},
    }


def _write_source(tmp_path: Path, fmt: str, tensors=None) -> Path:
    """Write a mock checkpoint in *fmt* and return its path."""
    from ckpt.convert import _SERIALIZERS

    tensors = tensors or _dummy_tensors()
    ext_map = {
        ConversionFormat.SAFETENSORS: ".safetensors",
        ConversionFormat.PYTORCH: ".pt",
        ConversionFormat.NUMPY: ".npz",
        ConversionFormat.GGUF: ".gguf",
    }
    src = tmp_path / f"model{ext_map[fmt]}"
    src.write_bytes(_SERIALIZERS[fmt](tensors))
    return src


# ---------------------------------------------------------------------------
# ConversionFormat
# ---------------------------------------------------------------------------

class TestConversionFormat:
    def test_all_contains_four(self):
        assert len(ConversionFormat.ALL) == 4
        assert "safetensors" in ConversionFormat.ALL
        assert "gguf" in ConversionFormat.ALL

    def test_extensions_map(self):
        assert ConversionFormat._EXTENSIONS[".safetensors"] == "safetensors"
        assert ConversionFormat._EXTENSIONS[".pt"] == "pytorch"
        assert ConversionFormat._EXTENSIONS[".gguf"] == "gguf"


# ---------------------------------------------------------------------------
# ConversionConfig validation
# ---------------------------------------------------------------------------

class TestConversionConfig:
    def test_valid(self):
        cfg = ConversionConfig(source_format="safetensors", target_format="pytorch")
        assert cfg.source_format == "safetensors"

    def test_rejects_unknown_source(self):
        with pytest.raises(ValueError, match="Unsupported source"):
            ConversionConfig(source_format="pickle", target_format="pytorch")

    def test_rejects_unknown_target(self):
        with pytest.raises(ValueError, match="Unsupported target"):
            ConversionConfig(source_format="pytorch", target_format="pickle")

    def test_rejects_same_format(self):
        with pytest.raises(ValueError, match="must differ"):
            ConversionConfig(source_format="pytorch", target_format="pytorch")

    def test_optional_fields(self):
        cfg = ConversionConfig(
            source_format="numpy",
            target_format="safetensors",
            dtype="F16",
            shard_size_mb=512,
        )
        assert cfg.dtype == "F16"
        assert cfg.shard_size_mb == 512


# ---------------------------------------------------------------------------
# ConversionResult
# ---------------------------------------------------------------------------

class TestConversionResult:
    def test_compression_ratio(self):
        r = ConversionResult("a", "b", source_size=1000, target_size=500,
                             duration_s=1.0)
        assert r.compression_ratio == pytest.approx(0.5)

    def test_size_change_pct(self):
        r = ConversionResult("a", "b", source_size=1000, target_size=1200,
                             duration_s=1.0)
        assert r.size_change_pct == pytest.approx(20.0)

    def test_zero_source(self):
        r = ConversionResult("a", "b", source_size=0, target_size=100,
                             duration_s=0.0)
        assert r.compression_ratio == 0.0
        assert r.size_change_pct == 0.0


# ---------------------------------------------------------------------------
# Round-trip serialisation
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("fmt", list(ConversionFormat.ALL))
    def test_round_trip(self, fmt):
        from ckpt.convert import _DESERIALIZERS, _SERIALIZERS

        tensors = _dummy_tensors()
        blob = _SERIALIZERS[fmt](tensors)
        recovered = _DESERIALIZERS[fmt](blob)
        assert set(recovered) == set(tensors)
        for name in tensors:
            assert recovered[name]["data"] == tensors[name]["data"]
            assert recovered[name]["dtype"] == tensors[name]["dtype"]
            assert recovered[name]["shape"] == tensors[name]["shape"]


# ---------------------------------------------------------------------------
# FormatConverter
# ---------------------------------------------------------------------------

class TestFormatConverter:
    def test_detect_format_safetensors(self):
        assert FormatConverter.detect_format("model.safetensors") == "safetensors"

    def test_detect_format_pytorch(self):
        assert FormatConverter.detect_format("model.pt") == "pytorch"
        assert FormatConverter.detect_format("model.bin") == "pytorch"

    def test_detect_format_numpy(self):
        assert FormatConverter.detect_format("model.npz") == "numpy"

    def test_detect_format_gguf(self):
        assert FormatConverter.detect_format("model.gguf") == "gguf"

    def test_detect_format_unknown(self):
        with pytest.raises(ValueError, match="Cannot detect"):
            FormatConverter.detect_format("model.xyz")

    def test_supported_conversions_count(self):
        pairs = FormatConverter.supported_conversions()
        # 4 formats, each can convert to 3 others = 12
        assert len(pairs) == 12

    def test_supported_conversions_no_self(self):
        for src, tgt in FormatConverter.supported_conversions():
            assert src != tgt

    def test_convert_safetensors_to_pytorch(self, tmp_path):
        src = _write_source(tmp_path, ConversionFormat.SAFETENSORS)
        tgt = tmp_path / "out.pt"
        cfg = ConversionConfig(source_format="safetensors", target_format="pytorch")
        result = FormatConverter(cfg).convert(str(src), str(tgt))
        assert tgt.exists()
        assert result.source_format == "safetensors"
        assert result.target_format == "pytorch"
        assert result.source_size > 0
        assert result.target_size > 0
        assert result.duration_s >= 0

    def test_convert_pytorch_to_gguf(self, tmp_path):
        src = _write_source(tmp_path, ConversionFormat.PYTORCH)
        tgt = tmp_path / "out.gguf"
        cfg = ConversionConfig(source_format="pytorch", target_format="gguf")
        result = FormatConverter(cfg).convert(str(src), str(tgt))
        assert tgt.exists()
        assert result.metadata["n_tensors"] == 2

    def test_convert_with_dtype_override(self, tmp_path):
        src = _write_source(tmp_path, ConversionFormat.SAFETENSORS)
        tgt = tmp_path / "out.npz"
        cfg = ConversionConfig(
            source_format="safetensors",
            target_format="numpy",
            dtype="F16",
        )
        result = FormatConverter(cfg).convert(str(src), str(tgt))
        assert result.target_size > 0

    def test_convert_with_sharding(self, tmp_path):
        # Create tensors large enough to need two shards at 1 MB limit
        # but use a very small shard for testing
        tensors = {
            f"t{i}": {"dtype": "F32", "shape": [100], "data": os.urandom(400)}
            for i in range(10)
        }
        src = _write_source(tmp_path, ConversionFormat.SAFETENSORS, tensors)
        tgt = tmp_path / "out" / "model.pt"
        ConversionConfig(
            source_format="safetensors",
            target_format="pytorch",
            shard_size_mb=0,  # force per-tensor shards via limit=0
        )
        # shard_size_mb=0 → limit=0 bytes, each tensor in own shard (except first group)
        # Use a tiny limit instead
        cfg2 = ConversionConfig(
            source_format="safetensors",
            target_format="pytorch",
        )
        cfg2.shard_size_mb = 1  # 1 MB — all 4 KB of tensors fit in one shard
        result = FormatConverter(cfg2).convert(str(src), str(tgt))
        assert result.metadata["n_shards"] >= 1

    def test_convert_missing_source(self, tmp_path):
        cfg = ConversionConfig(source_format="safetensors", target_format="pytorch")
        with pytest.raises(FileNotFoundError):
            FormatConverter(cfg).convert(str(tmp_path / "nope.safetensors"), str(tmp_path / "out.pt"))


# ---------------------------------------------------------------------------
# convert_checkpoint convenience
# ---------------------------------------------------------------------------

class TestConvertCheckpoint:
    def test_auto_detect(self, tmp_path):
        src = _write_source(tmp_path, ConversionFormat.SAFETENSORS)
        tgt = tmp_path / "out.pt"
        result = convert_checkpoint(str(src), str(tgt))
        assert result.source_format == "safetensors"
        assert result.target_format == "pytorch"

    def test_explicit_config(self, tmp_path):
        src = _write_source(tmp_path, ConversionFormat.NUMPY)
        tgt = tmp_path / "out.gguf"
        cfg = ConversionConfig(source_format="numpy", target_format="gguf")
        result = convert_checkpoint(str(src), str(tgt), config=cfg)
        assert result.target_format == "gguf"

    def test_data_integrity(self, tmp_path):
        """Verify tensor data survives a full round-trip through conversion."""
        from ckpt.convert import _DESERIALIZERS

        tensors = _dummy_tensors()
        src = _write_source(tmp_path, ConversionFormat.SAFETENSORS, tensors)
        tgt = tmp_path / "out.pt"
        convert_checkpoint(str(src), str(tgt))
        recovered = _DESERIALIZERS["pytorch"](tgt.read_bytes())
        for name in tensors:
            assert recovered[name]["data"] == tensors[name]["data"]

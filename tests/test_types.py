"""Tests for ckpt._types."""

import pytest

from ckpt._types import (
    DTYPE_SIZES,
    CheckpointFormat,
    CheckpointInfo,
    CkptError,
    DiffEntry,
    DiffResult,
    DType,
    FormatError,
    MergeConfig,
    MergeError,
    TensorInfo,
)


class TestTensorInfo:
    def test_numel(self):
        t = TensorInfo(name="w", shape=[768, 3072], dtype=DType.F16)
        assert t.numel == 768 * 3072

    def test_numel_scalar(self):
        t = TensorInfo(name="b", shape=[], dtype=DType.F32)
        assert t.numel == 1

    def test_size_bytes(self):
        t = TensorInfo(name="w", shape=[1024, 1024], dtype=DType.F32)
        assert t.size_bytes == 1024 * 1024 * 4

    def test_size_bytes_f16(self):
        t = TensorInfo(name="w", shape=[100], dtype=DType.F16)
        assert t.size_bytes == 200

    def test_shape_str(self):
        t = TensorInfo(name="w", shape=[2, 3, 4], dtype=DType.F32)
        assert t.shape_str == "2×3×4"


class TestCheckpointInfo:
    def _make(self):
        tensors = [
            TensorInfo(name="layer.0.weight", shape=[768, 768], dtype=DType.F16),
            TensorInfo(name="layer.0.bias", shape=[768], dtype=DType.F32),
            TensorInfo(name="layer.1.weight", shape=[768, 3072], dtype=DType.F16),
        ]
        return CheckpointInfo(
            path="/tmp/model.safetensors",
            format=CheckpointFormat.SAFETENSORS,
            file_size=1024 * 1024,
            tensors=tensors,
        )

    def test_n_tensors(self):
        assert self._make().n_tensors == 3

    def test_n_parameters(self):
        info = self._make()
        expected = 768 * 768 + 768 + 768 * 3072
        assert info.n_parameters == expected

    def test_dtype_summary(self):
        s = self._make().dtype_summary()
        assert "F16" in s
        assert "F32" in s

    def test_layer_groups(self):
        groups = self._make().layer_groups()
        assert "layer.0.weight" in groups or "layer.0" in groups

    def test_total_bytes(self):
        assert self._make().total_bytes > 0


class TestDiffResult:
    def test_no_changes(self):
        r = DiffResult(path_a="a", path_b="b", entries=[], n_shared=5, n_identical=5)
        assert not r.has_changes
        assert r.n_changes == 0

    def test_with_changes(self):
        entries = [DiffEntry(tensor_name="w", change_type="added", details="new")]
        r = DiffResult(path_a="a", path_b="b", entries=entries, n_shared=3, n_identical=3)
        assert r.has_changes
        assert r.n_changes == 1


class TestMergeConfig:
    def test_valid(self):
        c = MergeConfig(base_path="a", adapter_path="b", output_path="c")
        assert c.alpha == 1.0

    def test_invalid_alpha_high(self):
        with pytest.raises(ValueError, match="alpha"):
            MergeConfig(base_path="a", adapter_path="b", output_path="c", alpha=3.0)

    def test_invalid_alpha_low(self):
        with pytest.raises(ValueError, match="alpha"):
            MergeConfig(base_path="a", adapter_path="b", output_path="c", alpha=-0.5)


class TestDType:
    def test_sizes(self):
        assert DTYPE_SIZES[DType.F32] == 4
        assert DTYPE_SIZES[DType.F16] == 2
        assert DTYPE_SIZES[DType.BF16] == 2
        assert DTYPE_SIZES[DType.I8] == 1

    def test_all_dtypes_have_sizes(self):
        for dt in DType:
            if dt != DType.UNKNOWN:
                assert dt in DTYPE_SIZES


class TestExceptions:
    def test_ckpt_error(self):
        e = CkptError("bad")
        assert isinstance(e, Exception)

    def test_format_error(self):
        e = FormatError("corrupt")
        assert isinstance(e, CkptError)

    def test_merge_error(self):
        e = MergeError("mismatch")
        assert isinstance(e, CkptError)

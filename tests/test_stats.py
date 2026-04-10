"""Tests for ckpt.stats."""

from ckpt._types import CheckpointFormat, CheckpointInfo, DType, TensorInfo
from ckpt.stats import (
    compute_tensor_stats,
    format_stats,
    stats_from_info,
)


def _make_info():
    tensors = [
        TensorInfo(name="model.layers.0.weight", shape=[768, 768], dtype=DType.F16),
        TensorInfo(name="model.layers.0.bias", shape=[768], dtype=DType.F32),
        TensorInfo(name="model.layers.1.weight", shape=[768, 3072], dtype=DType.F16),
        TensorInfo(name="model.embed.weight", shape=[50257, 768], dtype=DType.F16),
    ]
    return CheckpointInfo(
        path="/tmp/model.safetensors",
        format=CheckpointFormat.SAFETENSORS,
        file_size=80_000_000,
        tensors=tensors,
    )


class TestStatsFromInfo:
    def test_basic(self):
        s = stats_from_info(_make_info())
        assert s.n_tensors == 4
        assert s.n_parameters > 0
        assert s.total_bytes > 0

    def test_dtype_counts(self):
        s = stats_from_info(_make_info())
        assert "F16" in s.dtype_counts
        assert "F32" in s.dtype_counts

    def test_layer_groups(self):
        s = stats_from_info(_make_info())
        assert len(s.layer_groups) > 0

    def test_tensor_stats(self):
        s = stats_from_info(_make_info())
        assert len(s.tensor_stats) == 4
        for ts in s.tensor_stats:
            assert ts.numel > 0

    def test_total_size_human(self):
        s = stats_from_info(_make_info())
        assert any(u in s.total_size_human for u in ("B", "KB", "MB", "GB"))


class TestComputeTensorStats:
    def test_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ts = compute_tensor_stats("w", values, [5], "F32", 20)
        assert ts.mean == 3.0
        assert ts.min_val == 1.0
        assert ts.max_val == 5.0
        assert ts.numel == 5

    def test_empty(self):
        ts = compute_tensor_stats("w", [], [0], "F32", 0)
        assert ts.numel == 0
        assert ts.mean is None

    def test_sparsity(self):
        values = [0.0, 0.0, 1.0, 0.0, 2.0]
        ts = compute_tensor_stats("w", values, [5], "F32", 20)
        assert ts.sparsity == 0.6

    def test_std(self):
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        ts = compute_tensor_stats("w", values, [8], "F32", 32)
        assert ts.std is not None
        assert ts.std > 0

    def test_abs_mean(self):
        values = [-1.0, 1.0, -2.0, 2.0]
        ts = compute_tensor_stats("w", values, [4], "F32", 16)
        assert ts.abs_mean == 1.5


class TestFormatStats:
    def test_contains_path(self):
        s = stats_from_info(_make_info())
        text = format_stats(s)
        assert "/tmp/model.safetensors" in text

    def test_contains_params(self):
        s = stats_from_info(_make_info())
        text = format_stats(s)
        assert "Parameters:" in text

    def test_contains_dtypes(self):
        s = stats_from_info(_make_info())
        text = format_stats(s)
        assert "F16" in text

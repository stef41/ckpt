"""Tests for ckpt.estimator — checkpoint size reduction estimator."""

from __future__ import annotations

import math

import pytest

from ckpt._types import CheckpointFormat, CheckpointInfo, DType, TensorInfo
from ckpt.estimator import (
    EstimationResult,
    QuantEstimationResult,
    TensorEstimate,
    estimate_quantized_size,
    estimate_reduction,
    format_estimation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_info(tensors: list[TensorInfo] | None = None, file_size: int = 0) -> CheckpointInfo:
    if tensors is None:
        tensors = [
            TensorInfo(name="weight", shape=[1024, 1024], dtype=DType.F32),
            TensorInfo(name="bias", shape=[1024], dtype=DType.F32),
        ]
    if file_size == 0:
        file_size = sum(t.size_bytes for t in tensors)
    return CheckpointInfo(
        path="model.safetensors",
        format=CheckpointFormat.SAFETENSORS,
        file_size=file_size,
        tensors=tensors,
    )


# ---------------------------------------------------------------------------
# estimate_reduction
# ---------------------------------------------------------------------------

class TestEstimateReduction:
    def test_f32_to_f16_halves_size(self):
        info = _make_info()
        result = estimate_reduction(info, target_dtype="float16")
        assert result.estimated_size == info.total_bytes // 2

    def test_reduction_bytes_and_percent(self):
        info = _make_info()
        result = estimate_reduction(info, target_dtype="float16")
        assert result.reduction_bytes == info.total_bytes - result.estimated_size
        assert result.reduction_percent == pytest.approx(50.0, abs=0.1)

    def test_per_tensor_breakdown_length(self):
        info = _make_info()
        result = estimate_reduction(info, target_dtype="float16")
        assert len(result.per_tensor) == len(info.tensors)

    def test_per_tensor_names_match(self):
        info = _make_info()
        result = estimate_reduction(info, target_dtype="float16")
        assert [te.name for te in result.per_tensor] == [t.name for t in info.tensors]

    def test_to_int8_quarter_size(self):
        info = _make_info()
        result = estimate_reduction(info, target_dtype="int8")
        assert result.estimated_size == info.total_bytes // 4

    def test_to_int4_eighth_size(self):
        info = _make_info()
        result = estimate_reduction(info, target_dtype="int4")
        # F32 is 4 bytes, int4 is 0.5 bytes → 1/8
        assert result.estimated_size == int(math.ceil(info.total_bytes / 8))

    def test_same_dtype_no_change(self):
        info = _make_info()
        result = estimate_reduction(info, target_dtype="float32")
        assert result.reduction_bytes == 0
        assert result.reduction_percent == pytest.approx(0.0)

    def test_unknown_dtype_raises(self):
        info = _make_info()
        with pytest.raises(ValueError, match="Unknown dtype"):
            estimate_reduction(info, target_dtype="float3")

    def test_empty_checkpoint(self):
        info = _make_info(tensors=[])
        result = estimate_reduction(info, target_dtype="float16")
        assert result.original_size == 0
        assert result.estimated_size == 0
        assert result.reduction_percent == 0.0

    def test_bfloat16_alias(self):
        info = _make_info()
        r1 = estimate_reduction(info, target_dtype="bfloat16")
        r2 = estimate_reduction(info, target_dtype="bf16")
        assert r1.estimated_size == r2.estimated_size

    def test_mixed_dtypes(self):
        tensors = [
            TensorInfo(name="a", shape=[100], dtype=DType.F32),
            TensorInfo(name="b", shape=[100], dtype=DType.F16),
        ]
        info = _make_info(tensors=tensors)
        result = estimate_reduction(info, target_dtype="float16")
        # a: 100*4 → 100*2, b: 100*2 → 100*2
        assert result.estimated_size == 200 + 200
        assert result.original_size == 400 + 200


# ---------------------------------------------------------------------------
# estimate_quantized_size
# ---------------------------------------------------------------------------

class TestEstimateQuantizedSize:
    def test_4bit(self):
        info = _make_info()
        result = estimate_quantized_size(info, bits=4)
        expected = int(math.ceil(info.n_parameters * 4 / 8))
        assert result.estimated_size == expected

    def test_8bit(self):
        info = _make_info()
        result = estimate_quantized_size(info, bits=8)
        expected = int(math.ceil(info.n_parameters * 8 / 8))
        assert result.estimated_size == expected

    def test_reduction_percent(self):
        info = _make_info()
        result = estimate_quantized_size(info, bits=4)
        assert result.reduction_percent > 0

    def test_invalid_bits_raises(self):
        info = _make_info()
        with pytest.raises(ValueError, match="bits must be a positive"):
            estimate_quantized_size(info, bits=0)

    def test_total_params(self):
        info = _make_info()
        result = estimate_quantized_size(info, bits=4)
        assert result.total_params == info.n_parameters


# ---------------------------------------------------------------------------
# format_estimation
# ---------------------------------------------------------------------------

class TestFormatEstimation:
    def test_contains_key_fields(self):
        info = _make_info()
        result = estimate_reduction(info, target_dtype="float16")
        text = format_estimation(result)
        assert "float16" in text
        assert "Original size" in text
        assert "Estimated size" in text
        assert "Reduction" in text

    def test_per_tensor_lines(self):
        info = _make_info()
        result = estimate_reduction(info, target_dtype="float16")
        text = format_estimation(result)
        for t in info.tensors:
            assert t.name in text

    def test_empty_no_crash(self):
        result = EstimationResult(
            original_size=0, estimated_size=0, target_dtype="fp16", per_tensor=[],
        )
        text = format_estimation(result)
        assert "0.0%" in text

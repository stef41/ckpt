"""Tests for ckpt.diff."""


from ckpt._types import (
    CheckpointFormat,
    CheckpointInfo,
    DType,
    TensorInfo,
)
from ckpt.diff import diff_infos, format_diff


def _info(tensors, path="test.safetensors"):
    return CheckpointInfo(
        path=path, format=CheckpointFormat.SAFETENSORS,
        file_size=1024, tensors=tensors,
    )


class TestDiffInfos:
    def test_identical(self):
        tensors = [
            TensorInfo(name="w", shape=[10, 10], dtype=DType.F32),
            TensorInfo(name="b", shape=[10], dtype=DType.F32),
        ]
        result = diff_infos(_info(tensors, "a"), _info(tensors, "b"))
        assert not result.has_changes
        assert result.n_shared == 2
        assert result.n_identical == 2

    def test_added_tensor(self):
        a = [TensorInfo(name="w", shape=[10], dtype=DType.F32)]
        b = [
            TensorInfo(name="w", shape=[10], dtype=DType.F32),
            TensorInfo(name="b", shape=[10], dtype=DType.F32),
        ]
        result = diff_infos(_info(a, "a"), _info(b, "b"))
        assert result.n_changes == 1
        added = [e for e in result.entries if e.change_type == "added"]
        assert len(added) == 1
        assert added[0].tensor_name == "b"

    def test_removed_tensor(self):
        a = [
            TensorInfo(name="w", shape=[10], dtype=DType.F32),
            TensorInfo(name="b", shape=[10], dtype=DType.F32),
        ]
        b = [TensorInfo(name="w", shape=[10], dtype=DType.F32)]
        result = diff_infos(_info(a, "a"), _info(b, "b"))
        removed = [e for e in result.entries if e.change_type == "removed"]
        assert len(removed) == 1
        assert removed[0].tensor_name == "b"

    def test_shape_changed(self):
        a = [TensorInfo(name="w", shape=[10, 10], dtype=DType.F32)]
        b = [TensorInfo(name="w", shape=[10, 20], dtype=DType.F32)]
        result = diff_infos(_info(a, "a"), _info(b, "b"))
        changes = [e for e in result.entries if e.change_type == "shape_changed"]
        assert len(changes) == 1
        assert "10×10" in changes[0].details
        assert "10×20" in changes[0].details

    def test_dtype_changed(self):
        a = [TensorInfo(name="w", shape=[10], dtype=DType.F32)]
        b = [TensorInfo(name="w", shape=[10], dtype=DType.F16)]
        result = diff_infos(_info(a, "a"), _info(b, "b"))
        changes = [e for e in result.entries if e.change_type == "dtype_changed"]
        assert len(changes) == 1

    def test_multiple_changes(self):
        a = [
            TensorInfo(name="w1", shape=[10], dtype=DType.F32),
            TensorInfo(name="w2", shape=[10, 10], dtype=DType.F32),
            TensorInfo(name="w3", shape=[5], dtype=DType.F16),
        ]
        b = [
            TensorInfo(name="w1", shape=[10], dtype=DType.F32),  # identical
            TensorInfo(name="w2", shape=[10, 20], dtype=DType.F32),  # shape changed
            TensorInfo(name="w4", shape=[5], dtype=DType.F16),  # w3 removed, w4 added
        ]
        result = diff_infos(_info(a, "a"), _info(b, "b"))
        assert result.n_shared == 2  # w1 and w2
        assert result.n_identical == 1  # only w1
        types = {e.change_type for e in result.entries}
        assert "added" in types
        assert "removed" in types
        assert "shape_changed" in types

    def test_empty_checkpoints(self):
        result = diff_infos(_info([], "a"), _info([], "b"))
        assert not result.has_changes
        assert result.n_shared == 0


class TestFormatDiff:
    def test_no_changes(self):
        a = [TensorInfo(name="w", shape=[10], dtype=DType.F32)]
        result = diff_infos(_info(a, "a"), _info(a, "b"))
        text = format_diff(result)
        assert "Changes: 0" in text

    def test_with_changes(self):
        a = [TensorInfo(name="w", shape=[10], dtype=DType.F32)]
        b = [TensorInfo(name="w", shape=[20], dtype=DType.F32)]
        result = diff_infos(_info(a, "a"), _info(b, "b"))
        text = format_diff(result)
        assert "shape_changed" in text
        assert "~" in text

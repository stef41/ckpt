"""Tests for format_diff_rich and format_diff_table."""

from ckpt._types import DiffEntry, DiffResult
from ckpt.diff import format_diff_rich, format_diff_table


def _make_result(entries=None, n_shared=0, n_identical=0):
    return DiffResult(
        path_a="model_a.safetensors",
        path_b="model_b.safetensors",
        entries=entries or [],
        n_shared=n_shared,
        n_identical=n_identical,
    )


# ── format_diff_rich tests ──────────────────────────────────────


def test_rich_empty_diff():
    result = _make_result(n_shared=3, n_identical=3)
    out = format_diff_rich(result)
    assert "model_a.safetensors" in out
    assert "model_b.safetensors" in out
    assert "+0 added" in out
    assert "-0 removed" in out
    assert "~0 changed" in out


def test_rich_added_tensor():
    entries = [
        DiffEntry(
            tensor_name="layer.1.weight",
            change_type="added",
            details="shape=768×3072 dtype=F16",
        )
    ]
    out = format_diff_rich(_make_result(entries))
    # Green color code
    assert "\033[32m" in out
    assert "+ layer.1.weight" in out
    assert "+1 added" in out


def test_rich_removed_tensor():
    entries = [
        DiffEntry(
            tensor_name="embed.weight",
            change_type="removed",
            details="shape=32000×4096 dtype=F32",
        )
    ]
    out = format_diff_rich(_make_result(entries))
    # Red color code
    assert "\033[31m" in out
    assert "- embed.weight" in out
    assert "-1 removed" in out


def test_rich_shape_changed():
    entries = [
        DiffEntry(
            tensor_name="head.weight",
            change_type="shape_changed",
            details="768×768 → 1024×1024",
        )
    ]
    out = format_diff_rich(_make_result(entries, n_shared=1))
    # Yellow color code
    assert "\033[33m" in out
    assert "~ head.weight" in out
    assert "~1 changed" in out


def test_rich_dtype_changed():
    entries = [
        DiffEntry(
            tensor_name="norm.weight",
            change_type="dtype_changed",
            details="F32 → BF16",
        )
    ]
    out = format_diff_rich(_make_result(entries, n_shared=1))
    assert "\033[33m" in out
    assert "~ norm.weight" in out


def test_rich_mixed():
    entries = [
        DiffEntry("a.weight", "removed", "shape=10×10 dtype=F32"),
        DiffEntry("b.weight", "added", "shape=20×20 dtype=F16"),
        DiffEntry("c.weight", "shape_changed", "10×10 → 20×20"),
    ]
    out = format_diff_rich(_make_result(entries, n_shared=1))
    assert "+1 added" in out
    assert "-1 removed" in out
    assert "~1 changed" in out


def test_rich_summary_at_bottom():
    out = format_diff_rich(_make_result())
    lines = out.strip().split("\n")
    assert "Summary:" in lines[-1]


# ── format_diff_table tests ─────────────────────────────────────


def test_table_empty():
    out = format_diff_table(_make_result())
    assert "Status" in out
    assert "Tensor" in out
    # Only header + separator
    lines = out.strip().split("\n")
    assert len(lines) == 2


def test_table_added():
    entries = [
        DiffEntry("layer.0.bias", "added", "shape=3072 dtype=BF16"),
    ]
    out = format_diff_table(_make_result(entries))
    lines = out.strip().split("\n")
    assert len(lines) == 3  # header + sep + 1 row
    row = lines[2]
    assert "added" in row
    assert "layer.0.bias" in row
    assert "3072" in row
    assert "BF16" in row


def test_table_removed():
    entries = [
        DiffEntry("embed.weight", "removed", "shape=32000×4096 dtype=F32"),
    ]
    out = format_diff_table(_make_result(entries))
    row = out.strip().split("\n")[2]
    assert "removed" in row
    assert "32000×4096" in row
    assert "F32" in row


def test_table_shape_changed():
    entries = [
        DiffEntry("fc.weight", "shape_changed", "768×768 → 1024×1024"),
    ]
    out = format_diff_table(_make_result(entries))
    row = out.strip().split("\n")[2]
    assert "shape_changed" in row
    assert "768×768" in row
    assert "1024×1024" in row


def test_table_dtype_changed():
    entries = [
        DiffEntry("norm.weight", "dtype_changed", "F32 → BF16"),
    ]
    out = format_diff_table(_make_result(entries))
    row = out.strip().split("\n")[2]
    assert "dtype_changed" in row
    assert "F32 → BF16" in row


def test_table_multiple_rows():
    entries = [
        DiffEntry("a.w", "removed", "shape=10 dtype=F32"),
        DiffEntry("b.w", "added", "shape=20 dtype=F16"),
        DiffEntry("c.w", "shape_changed", "5×5 → 10×10"),
        DiffEntry("d.w", "dtype_changed", "F32 → BF16"),
    ]
    out = format_diff_table(_make_result(entries))
    lines = out.strip().split("\n")
    assert len(lines) == 6  # header + sep + 4 rows


def test_table_no_ansi_codes():
    entries = [
        DiffEntry("x.weight", "added", "shape=100 dtype=F16"),
    ]
    out = format_diff_table(_make_result(entries))
    assert "\033[" not in out

"""Tests for ckpt.cli."""

import json
import struct
import pytest

try:
    from click.testing import CliRunner
    _HAS_CLICK = True
except ImportError:
    _HAS_CLICK = False

from ckpt.cli import _build_cli

pytestmark = pytest.mark.skipif(not _HAS_CLICK, reason="click not installed")


def _make_st(path):
    header = {
        "model.weight": {"dtype": "F16", "shape": [768, 768], "data_offsets": [0, 1179648]},
        "model.bias": {"dtype": "F32", "shape": [768], "data_offsets": [1179648, 1182720]},
    }
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * 1182720)


@pytest.fixture
def cli():
    c = _build_cli()
    assert c is not None
    return c


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIInfo:
    def test_inspect(self, cli, runner, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_st(p)
        result = runner.invoke(cli, ["info", str(p)])
        assert result.exit_code == 0, result.output
        assert "model.weight" in result.output

    def test_inspect_json(self, cli, runner, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_st(p)
        result = runner.invoke(cli, ["info", str(p), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["n_tensors"] == 2


class TestCLIDiff:
    def test_identical(self, cli, runner, tmp_path):
        p1 = tmp_path / "a.safetensors"
        p2 = tmp_path / "b.safetensors"
        _make_st(p1)
        _make_st(p2)
        result = runner.invoke(cli, ["diff", str(p1), str(p2)])
        assert result.exit_code == 0


class TestCLIStats:
    def test_stats(self, cli, runner, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_st(p)
        result = runner.invoke(cli, ["stats", str(p)])
        assert result.exit_code == 0
        assert "Parameters:" in result.output


class TestCLIValidate:
    def test_valid(self, cli, runner, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_st(p)
        result = runner.invoke(cli, ["validate", str(p)])
        assert result.exit_code == 0
        assert "✓" in result.output or "valid" in result.output


class TestCLIGroup:
    def test_help(self, cli, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "ckpt" in result.output
